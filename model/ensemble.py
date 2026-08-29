"""
Signal ensemble using HistGradientBoostingClassifier.
Replaces MLPClassifier — gradient boosting is significantly better on
tabular financial data and doesn't suffer from partial_fit degradation.

Full refit on the replay buffer every RETRAIN_INTERVAL new outcomes.
"""

import os
import time
import numpy as np
import joblib
from collections import deque
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance as sk_perm_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import _SigmoidCalibration
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, KFold
from utils.logger import get_logger
from config import (
    SIGNAL_COUNT, REPLAY_BUFFER_SIZE, RETRAIN_INTERVAL,
    MIN_TRAIN_SAMPLES, MODEL_WEIGHTS_PATH,
)

AUTO_TUNE_MIN_SAMPLES = 200   # minimum samples before grid search runs
AUTO_TUNE_EVERY       = 5     # run every N retrains

log = get_logger("ensemble")

class SignalEnsemble:

    def __init__(self, name: str = "ensemble", keep_features: list[str] | None = None):
        self._name   = name
        self._model_path  = MODEL_WEIGHTS_PATH.parent / f"{name}.joblib"
        self._buffer_path = MODEL_WEIGHTS_PATH.parent / f"replay_buffer_{name}.npz"

        # Optional feature pruning: zero every column NOT in keep_features so the
        # model ignores it (validated equivalent to dropping for this tree model).
        # Buffer still stores the full 53-vector, so the set can be re-tuned later.
        self._zero_cols: np.ndarray | None = None
        if keep_features:
            from signals.aggregator import SIGNAL_NAMES
            keep = {SIGNAL_NAMES.index(f) for f in keep_features}
            self._zero_cols = np.array(
                [i for i in range(SIGNAL_COUNT) if i not in keep], dtype=int
            )

        self.model        = self._build_model()
        self._fitted      = False
        self._calibrator: "IsotonicRegression | _SigmoidCalibration | None" = None
        self._calibrator_path = MODEL_WEIGHTS_PATH.parent / f"{name}.calibrator.joblib"

        self._replay_X: deque[np.ndarray] = deque(maxlen=REPLAY_BUFFER_SIZE)
        self._replay_y: deque[float]       = deque(maxlen=REPLAY_BUFFER_SIZE)
        self._new_since_train: int = 0
        self._train_count: int     = 0
        self._cached_importance: np.ndarray | None = None

        self._load()
        self._load_buffer()

    def _build_model(self) -> HistGradientBoostingClassifier:
        return HistGradientBoostingClassifier(
            max_iter=300,              # higher ceiling; early stopping decides when to stop
            max_depth=2,
            learning_rate=0.05,
            min_samples_leaf=10,
            l2_regularization=0.1,
            early_stopping=True,       # stop before memorising training data
            validation_fraction=0.15,  # hold out 15% for early-stopping validation
            n_iter_no_change=20,       # stop after 20 rounds with no improvement
            tol=1e-4,
            random_state=42,
        )

    # ── Inference ──────────────────────────────────────────────────────────────

    def _apply_mask(self, X: np.ndarray) -> np.ndarray:
        """Zero the pruned feature columns (no-op when no mask configured)."""
        if self._zero_cols is None:
            return X
        X = np.array(X, dtype=np.float32, copy=True)
        X[..., self._zero_cols] = 0.0
        return X

    def predict(self, signal_vec: np.ndarray) -> float:
        """Returns calibrated P(BTC_UP) in [0, 1]. Returns 0.5 if model not trained yet."""
        if not self._fitted:
            return 0.5
        raw = float(self.model.predict_proba(self._apply_mask(signal_vec.reshape(1, -1)))[0][1])
        if self._calibrator is not None:
            return float(np.clip(self._calibrator.predict([raw])[0], 0.05, 0.95))
        return float(np.clip(raw, 0.05, 0.95))

    # ── Online Learning ────────────────────────────────────────────────────────

    def add_outcome(self, signal_vec: np.ndarray, btc_went_up: bool) -> None:
        self._replay_X.append(signal_vec.copy())
        self._replay_y.append(1.0 if btc_went_up else 0.0)
        self._new_since_train += 1

        # The first train fires as soon as we have enough data — don't make it
        # wait for RETRAIN_INTERVAL *new* outcomes. _new_since_train resets on
        # restart, so slow models (15m settle a few times an hour) never reach
        # 25-in-a-session between restarts and would stay untrained forever.
        first_train = self._train_count == 0
        if (len(self._replay_X) >= MIN_TRAIN_SAMPLES
                and (first_train or self._new_since_train >= RETRAIN_INTERVAL)):
            self.retrain()          # retrain() persists weights + buffer
            self._new_since_train = 0
        else:
            # Persist on every outcome. Slow models (15m settle only a few
            # times an hour) lost up to 9 outcomes on restart with the old
            # every-10 save, and stayed at 0 on the dashboard for hours.
            self._save_buffer()

    def retrain(self, increment_count: bool = True) -> None:
        # Mask once here so fit, calibrator, and importance all see the pruned X.
        X = self._apply_mask(np.array(self._replay_X, dtype=np.float32))
        y = np.array(self._replay_y, dtype=np.float32)

        pos_rate = y.mean()
        log.info(
            f"[{self._name}] Retraining on {len(X)} samples "
            f"(pos_rate={pos_rate:.2%}, train_count={self._train_count})"
        )

        # Exponential recency weighting — newest sample weight=1.0, oldest≈0.14
        n       = len(X)
        weights = np.exp(np.linspace(-2.0, 0.0, n)).astype(np.float32)
        weights = weights / weights.mean()  # keep mean weight at 1.0

        # Auto-tune hyperparameters every AUTO_TUNE_EVERY retrains if enough data
        if (len(X) >= AUTO_TUNE_MIN_SAMPLES
                and self._train_count > 0
                and self._train_count % AUTO_TUNE_EVERY == 0):
            self._auto_tune(X, y, weights)
        else:
            self.model.fit(X, y, sample_weight=weights)

        # Probability calibration via out-of-fold isotonic regression
        self._fit_calibrator(X, y)

        self._fitted      = True
        # Startup refits (increment_count=False) shouldn't inflate the retrain
        # counter — otherwise every restart bumps it and drifts the auto-tune cadence.
        if increment_count:
            self._train_count += 1

        self._compute_importance(X, y)
        log.info(f"[{self._name}] Retrain complete (train_count={self._train_count})")
        self.log_importance(list(range(SIGNAL_COUNT)))
        self._save()

    def _auto_tune(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> None:
        param_grid = {
            "max_depth":        [2, 3],
            "learning_rate":    [0.03, 0.05, 0.1],
            "min_samples_leaf": [10, 20],
        }
        tscv = TimeSeriesSplit(n_splits=3)
        gs   = GridSearchCV(
            self._build_model(),   # use current base hyperparams as starting point
            param_grid, cv=tscv, scoring="roc_auc", n_jobs=-1, refit=True,
        )
        try:
            gs.fit(X, y, sample_weight=weights)
            log.info(
                f"[{self._name}] Auto-tune best params: {gs.best_params_} "
                f"(CV AUC={gs.best_score_:.3f})"
            )
            self.model = gs.best_estimator_
        except Exception as e:
            log.warning(f"[{self._name}] Auto-tune failed ({e}), using current params.")
            self.model.fit(X, y, sample_weight=weights)

    def _fit_calibrator(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit a probability calibrator, choosing the scheme by sample size.

        The serving model (self.model) is trained on 100% of the buffer, so the
        calibrator should see predictions from the same model class. Validated on
        the live buffers via temporal-holdout ECE (the split that matches serving —
        the bot always predicts the next-in-time market):

          * Large buffers (>=1000): 5-fold OOF calibration wins clearly (5m_eth ECE
            0.110 -> 0.056). Full coverage gives the isotonic fit enough honest
            points to map the model's confidence accurately.
          * Small buffers (<1000): shuffled-fold OOF mixes regimes and generalises
            WORSE to the recent tail (15m ECE 0.111 -> 0.176), because temporal
            drift dominates at ~250 samples. The recent-tail Platt fit stays closer
            to the next market, so it wins there.

        Hence: OOF+isotonic at scale, recent-tail+sigmoid below it."""
        if len(X) < 80:
            log.debug(f"[{self._name}] Skipping calibrator — too few samples ({len(X)}).")
            return
        try:
            n = len(X)
            w = np.exp(np.linspace(-2.0, 0.0, n)).astype(np.float32)
            w /= w.mean()

            if n >= 1000:
                # Out-of-fold predictions from the serving model class over all data.
                # Shuffled K-fold (not TimeSeriesSplit) so every sample lands in
                # exactly one test fold — calibration needs full coverage, and at
                # this sample size regime drift no longer dominates the fit.
                oof = np.zeros(n, dtype=np.float64)
                for tr_idx, te_idx in KFold(5, shuffle=True, random_state=42).split(X):
                    m = self._build_model()
                    m.fit(X[tr_idx], y[tr_idx], sample_weight=w[tr_idx])
                    oof[te_idx] = m.predict_proba(X[te_idx])[:, 1]
                cal_probs, y_cal = oof, y
                cal    = IsotonicRegression(out_of_bounds="clip")
                method = "isotonic-OOF"
            else:
                # Recent-tail Platt: fit a tmp model on the older 80% and calibrate on
                # the most-recent 20%, which sits closest in time to the next market.
                # Sigmoid (not isotonic) — too few points for isotonic to be smooth.
                split      = max(50, int(n * 0.80))
                X_tr, y_tr = X[:split], y[:split]
                X_cal, y_cal = X[split:], y[split:]
                tmp = self._build_model()
                tmp.fit(X_tr, y_tr, sample_weight=w[:split])
                cal_probs = tmp.predict_proba(X_cal)[:, 1]
                cal    = _SigmoidCalibration()
                method = "sigmoid-tail"

            cal.fit(cal_probs, y_cal)
            self._calibrator = cal
            log.debug(
                f"[{self._name}] {method} calibrator fitted on {len(cal_probs)} points."
            )
        except Exception as e:
            log.warning(f"[{self._name}] Calibrator fit failed ({e}) — using raw probs.")

    # ── Signal Importance ──────────────────────────────────────────────────────

    def _compute_importance(self, X: np.ndarray, y: np.ndarray) -> None:
        """Compute permutation importance on a held-out tail and cache result."""
        try:
            n_val = min(200, len(X))
            X_val, y_val = X[-n_val:], y[-n_val:]
            result = sk_perm_importance(
                self.model, X_val, y_val,
                n_repeats=5, random_state=42, n_jobs=1, scoring="roc_auc",
            )
            imp = result.importances_mean
            imp = np.maximum(imp, 0.0)
            total = imp.sum()
            self._cached_importance = imp / (total + 1e-10)
        except Exception as e:
            log.warning(f"[{self._name}] Permutation importance failed ({e}) — using uniform.")
            self._cached_importance = None

    def signal_importance(self) -> np.ndarray:
        if not self._fitted:
            return np.ones(SIGNAL_COUNT) / SIGNAL_COUNT
        if self._cached_importance is not None:
            return self._cached_importance
        return np.ones(SIGNAL_COUNT) / SIGNAL_COUNT

    def log_importance(self, signal_names: list) -> None:
        from signals.aggregator import SIGNAL_NAMES
        imp    = self.signal_importance()
        ranked = sorted(zip(SIGNAL_NAMES, imp), key=lambda x: -x[1])
        log.info("Top 10 signals by importance:")
        for name, score in ranked[:10]:
            log.info(f"  {name:<25} {score:.4f}")

    # ── Persistence ────────────────────────────────────────────────────────────

    def _save(self) -> None:
        try:
            self._model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.model, self._model_path)
            if self._calibrator is not None:
                joblib.dump(self._calibrator, self._calibrator_path)
            log.debug(f"[{self._name}] Model + calibrator saved.")
        except Exception as e:
            log.warning(f"[{self._name}] Model save failed: {e}")
        self._save_buffer()

    def _save_buffer(self) -> None:
        try:
            self._buffer_path.parent.mkdir(parents=True, exist_ok=True)
            # Atomic write: the dashboard reads this file from another process,
            # so write to a temp file and rename to avoid a torn read.
            tmp = self._buffer_path.with_name(self._buffer_path.name + ".tmp")
            with open(tmp, "wb") as f:
                np.savez_compressed(
                    f,
                    X=np.array(self._replay_X, dtype=np.float32),
                    y=np.array(self._replay_y, dtype=np.float32),
                    meta=np.array([self._train_count, SIGNAL_COUNT]),
                )
            # On Windows os.replace fails with WinError 5 if another process (the
            # dashboard reading this .npz, or Defender scanning the fresh .tmp) has
            # a handle open at that instant. Retry over a few seconds to ride out
            # the transient lock instead of dropping the outcome.
            for attempt in range(12):
                try:
                    os.replace(tmp, self._buffer_path)
                    break
                except PermissionError:
                    if attempt == 11:
                        raise
                    time.sleep(0.25)
            log.debug(f"[{self._name}] Replay buffer saved ({len(self._replay_X)} samples).")
        except Exception as e:
            log.warning(f"[{self._name}] Buffer save failed: {e}")
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

    def _load(self) -> None:
        try:
            if self._model_path.exists():
                candidate = joblib.load(self._model_path)
                n_feat = getattr(candidate, "n_features_in_", None)
                if n_feat is not None and n_feat != SIGNAL_COUNT:
                    log.warning(
                        f"[{self._name}] Saved model expects {n_feat} features "
                        f"but SIGNAL_COUNT={SIGNAL_COUNT} — discarding stale weights."
                    )
                    self._model_path.unlink(missing_ok=True)
                    return
                self.model   = candidate
                self._fitted = True
                log.info(f"[{self._name}] Loaded saved model ({SIGNAL_COUNT} features).")
                if self._calibrator_path.exists():
                    try:
                        self._calibrator = joblib.load(self._calibrator_path)
                        log.info(f"[{self._name}] Loaded probability calibrator.")
                    except Exception:
                        pass
        except Exception as e:
            log.warning(f"[{self._name}] Could not load model: {e} — starting fresh.")

    def _load_buffer(self) -> None:
        try:
            if not self._buffer_path.exists():
                return
            # Read fully inside the context manager and copy out, so the file
            # handle is closed before retrain() below calls _save_buffer() →
            # os.replace on this same path. Holding it open here was locking our
            # own file and failing every startup refit's save (WinError 5).
            with np.load(self._buffer_path) as data:
                meta = data["meta"]
                saved_signal_count = int(meta[1])
                if saved_signal_count != SIGNAL_COUNT:
                    log.warning(
                        f"[{self._name}] Replay buffer signal count mismatch "
                        f"({saved_signal_count} saved vs {SIGNAL_COUNT} current) — discarding."
                    )
                    self._buffer_path.unlink(missing_ok=True)
                    return
                X = np.array(data["X"])
                y = np.array(data["y"])
                train_count = int(meta[0])

            for xi, yi in zip(X, y):
                self._replay_X.append(xi)
                self._replay_y.append(float(yi))
            self._train_count = train_count
            log.info(
                f"[{self._name}] Replay buffer restored: {len(self._replay_X)} samples, "
                f"{self._train_count} prior retrains."
            )
            # Always refit on startup so hyperparameter changes take effect
            # immediately. Count it as the inaugural train when train_count is
            # still 0, so models that crossed MIN_TRAIN_SAMPLES while we were
            # down (slow 15m models) flip is_trained=True instead of staying
            # stuck. Later startup refits stay uncounted so restarts don't
            # inflate the counter or drift the auto-tune cadence.
            if len(self._replay_X) >= MIN_TRAIN_SAMPLES:
                log.info(f"[{self._name}] Refitting model with current hyperparameters...")
                self.retrain(increment_count=(self._train_count == 0))
        except Exception as e:
            log.warning(f"[{self._name}] Could not load replay buffer: {e} — starting fresh.")

    # ── Stats ──────────────────────────────────────────────────────────────────

    @property
    def n_samples(self) -> int:
        return len(self._replay_X)

    @property
    def is_trained(self) -> bool:
        return self._train_count > 0
