"""
Expected Value framework.

EV = P(win) × profit_if_win - P(lose) × stake

Where:
  P(win)        = model's predicted probability, adjusted for direction
  profit_if_win = (1 - market_price) / market_price × stake  (for YES)
  P(lose)       = 1 - P(win)
  stake         = position size in USDC
"""

from dataclasses import dataclass
from data.polymarket_api import OrderBookState, PolymarketMarket
from config import EV_THRESHOLD_5M, EV_THRESHOLD_15M, MAX_POSITION_USDC, MIN_POSITION_USDC, KELLY_FRACTION, KELLY_MAX_PCT, MAX_OPEN_EXPOSURE_PCT, POLYMARKET_FEE, MODEL_CONVICTION_FLOOR, MODEL_CONVICTION_FLOOR_5M, BLOCK_5M_CONTRARIAN, CORR_HAIRCUT_RHO, MAX_ENTRY_PRICE_15M, BLOCK_15M_TOSSUP_LO, BLOCK_15M_TOSSUP_HI
from utils.logger import get_logger

log = get_logger("ev_calculator")


@dataclass
class TradeDecision:
    should_trade: bool
    direction: str        # "YES" or "NO"
    token_id: str
    market_price: float   # price of the token we're buying (0–1 range)
    model_prob: float     # model's estimated P(BTC_UP)
    ev_fraction: float    # EV as fraction of stake
    stake_usdc: float
    market_id: str
    reason: str = ""


def compute_ev(
    model_prob_up: float,
    market: PolymarketMarket,
    pm_book: OrderBookState,
    interval: str,
    ev_threshold_override: float | None = None,
    conviction_floor_override: float | None = None,
) -> TradeDecision:
    """
    Evaluate whether a trade has positive expected value.

    model_prob_up: P(BTC goes UP) from the ensemble, in [0, 1].
    ev_threshold_override: if set, replaces the config-based threshold.
    conviction_floor_override: if set, replaces the per-interval conviction floor
      (e.g. 0.0 to disable it — used by the "no-conviction" shadow probe).
    """
    base      = EV_THRESHOLD_5M if interval == "5m" else EV_THRESHOLD_15M
    threshold = ev_threshold_override if ev_threshold_override is not None else base

    yes_price = pm_book.yes_ask   # cost to buy YES (we pay ask)
    no_price  = pm_book.no_ask    # cost to buy NO

    if yes_price <= 0 or no_price <= 0:
        return TradeDecision(
            should_trade=False, direction="", token_id="", market_price=0,
            model_prob=model_prob_up, ev_fraction=0, stake_usdc=0,
            market_id=market.condition_id, reason="invalid_book",
        )

    # Spreads too wide → skip
    if pm_book.spread_yes() > 0.04:
        return TradeDecision(
            should_trade=False, direction="", token_id="", market_price=0,
            model_prob=model_prob_up, ev_fraction=0, stake_usdc=0,
            market_id=market.condition_id, reason="spread_too_wide",
        )

    # ── Evaluate YES direction ─────────────────────────────────────────────────
    # Buying YES at yes_price: pays yes_price, wins (1 - yes_price) if correct
    # Polymarket deducts POLYMARKET_FEE from the profit (not the stake).
    yes_profit       = (1.0 - yes_price) * (1.0 - POLYMARKET_FEE)
    ev_yes_fraction  = (model_prob_up * yes_profit) - ((1.0 - model_prob_up) * yes_price)
    ev_yes_fraction /= yes_price  # normalise to fraction of stake

    # ── Evaluate NO direction ──────────────────────────────────────────────────
    model_prob_down  = 1.0 - model_prob_up
    no_profit        = (1.0 - no_price) * (1.0 - POLYMARKET_FEE)
    ev_no_fraction   = (model_prob_down * no_profit) - ((1.0 - model_prob_down) * no_price)
    ev_no_fraction  /= no_price

    log.debug(
        f"EV YES: {ev_yes_fraction:.3f}  EV NO: {ev_no_fraction:.3f}  "
        f"Threshold: {threshold:.3f}  model_p={model_prob_up:.3f}"
    )

    # ── Select best direction ──────────────────────────────────────────────────
    best_ev   = max(ev_yes_fraction, ev_no_fraction)
    direction = "YES" if ev_yes_fraction >= ev_no_fraction else "NO"
    token_id  = market.token_id_yes if direction == "YES" else market.token_id_no
    price     = yes_price if direction == "YES" else no_price
    model_p   = model_prob_up if direction == "YES" else model_prob_down

    # ── Conviction floor (per-interval) ────────────────────────────────────────
    # The model only has demonstrated edge above this threshold; below it, win rate
    # collapses to a coin flip or worse. This also blocks against-the-model longshots
    # (model_p < 0.5), where EV picks the cheap side opposite the model's own call.
    # 5m uses a higher floor — its 0.70-0.80 band is a net loser (model overconfident).
    floor = MODEL_CONVICTION_FLOOR_5M if interval == "5m" else MODEL_CONVICTION_FLOOR
    if conviction_floor_override is not None:
        floor = conviction_floor_override
    if model_p < floor:
        return TradeDecision(
            should_trade=False, direction=direction, token_id=token_id,
            market_price=price, model_prob=model_p, ev_fraction=best_ev,
            stake_usdc=0, market_id=market.condition_id,
            reason=f"low_conviction ({model_p:.3f} < {floor:.2f})",
        )

    # ── 5m contrarian block ─────────────────────────────────────────────────────
    # Block 5m bets on the cheap (<0.5) side: against-the-model longshots EV picks for
    # their payout but that lose ~27% of the time. A persistent 5m bleeder.
    if interval == "5m" and BLOCK_5M_CONTRARIAN and price < 0.5:
        return TradeDecision(
            should_trade=False, direction=direction, token_id=token_id,
            market_price=price, model_prob=model_p, ev_fraction=best_ev,
            stake_usdc=0, market_id=market.condition_id,
            reason=f"contrarian_5m (price {price:.3f} < 0.50)",
        )

    # ── 15m price ceiling ────────────────────────────────────────────────────────
    # Expensive favorites (entry ≥ ceiling) are net-negative on two independent
    # ~100-trade samples (-0.07 and -0.089 per $1): the market has already priced
    # the move, so there's no edge left and we slightly overpay. The real edge
    # lives in the cheap/mid side. Block only 15m; 5m already order-disabled.
    if interval == "15m" and price >= MAX_ENTRY_PRICE_15M:
        return TradeDecision(
            should_trade=False, direction=direction, token_id=token_id,
            market_price=price, model_prob=model_p, ev_fraction=best_ev,
            stake_usdc=0, market_id=market.condition_id,
            reason=f"price_ceiling_15m ({price:.3f} >= {MAX_ENTRY_PRICE_15M:.2f})",
        )

    # ── 15m toss-up band ─────────────────────────────────────────────────────────
    # Coin-flip prices where the model has no directional edge — pure drag
    # (~-0.09/$1). Blocking [lo,hi) concentrates capital on the contrarian side
    # that carries the P&L, ~halving drawdown for ~8% less return. lo>=hi disables.
    if interval == "15m" and BLOCK_15M_TOSSUP_LO <= price < BLOCK_15M_TOSSUP_HI:
        return TradeDecision(
            should_trade=False, direction=direction, token_id=token_id,
            market_price=price, model_prob=model_p, ev_fraction=best_ev,
            stake_usdc=0, market_id=market.condition_id,
            reason=f"tossup_15m ({BLOCK_15M_TOSSUP_LO:.2f}<={price:.3f}<{BLOCK_15M_TOSSUP_HI:.2f})",
        )

    if best_ev < threshold:
        return TradeDecision(
            should_trade=False, direction=direction, token_id=token_id,
            market_price=price, model_prob=model_p, ev_fraction=best_ev,
            stake_usdc=0, market_id=market.condition_id,
            reason=f"ev_below_threshold ({best_ev:.3f} < {threshold:.3f})",
        )

    # ── Kelly position sizing ──────────────────────────────────────────────────
    # Full Kelly: f = (p * b - q) / b  where b = (1-price)/price
    b       = (1.0 - price) / price
    q       = 1.0 - model_p
    kelly_f = (model_p * b - q) / b
    kelly_f = max(0.0, kelly_f) * KELLY_FRACTION   # fractional Kelly

    # Auto-scaling max + drawdown brake
    try:
        import bankroll_state as _br
        _state    = _br.load()
        _bankroll = _state["bankroll"]
        dd_mult   = _br.drawdown_multiplier(_state)
        # Per-bet ceiling is the LOWER of: 15% of bankroll (lets bets compound with
        # edge) and the absolute MAX_POSITION_USDC liquidity cap (a thin 15m book
        # can't fill an arbitrarily large order no matter how big the bankroll gets).
        dyn_max   = min(MAX_POSITION_USDC, max(MIN_POSITION_USDC, _bankroll * KELLY_MAX_PCT))
    except Exception:
        dd_mult = 1.0
        dyn_max = MAX_POSITION_USDC
        _bankroll = MAX_POSITION_USDC   # fall back to the cap if state unavailable

    # Correlation haircut: shrink this bet in proportion to how many other open
    # bets already lean the same way (UP/DOWN). Correlated same-window crypto bets
    # act as one larger position, so full Kelly on each over-bets.
    import correlation as _corr
    corr_mult = _corr.haircut(direction, CORR_HAIRCUT_RHO)

    # kelly_f is a fraction of bankroll, so size off bankroll and let dyn_max
    # act as the per-bet cap (KELLY_MAX_PCT of bankroll). Multiplying kelly_f by
    # dyn_max instead shrank every bet to ~15% of its intended fractional-Kelly size.
    stake = min(kelly_f * _bankroll, dyn_max) * dd_mult * corr_mult

    # Open-exposure guard: never let the capital committed across all currently
    # open bets exceed MAX_OPEN_EXPOSURE_PCT of the bankroll. Each bet sizes off
    # the full bankroll, so without this the ~dozens of concurrent 5m/15m
    # positions (×4 assets) could risk several times the bankroll at once.
    import exposure as _exp
    room  = max(0.0, _bankroll * MAX_OPEN_EXPOSURE_PCT - _exp.current())
    stake = min(stake, room)
    if stake < MIN_POSITION_USDC:
        return TradeDecision(
            should_trade=False, direction=direction, token_id=token_id,
            market_price=price, model_prob=model_p, ev_fraction=best_ev,
            stake_usdc=0, market_id=market.condition_id, reason="exposure_cap",
        )
    stake = max(stake, MIN_POSITION_USDC)

    log.debug(
        f"Kelly sizing: f={kelly_f:.3f} dyn_max=${dyn_max:.2f} "
        f"dd_mult={dd_mult:.2f} corr_mult={corr_mult:.2f} stake=${stake:.2f}"
    )

    return TradeDecision(
        should_trade=True,
        direction=direction,
        token_id=token_id,
        market_price=price,
        model_prob=model_p,
        ev_fraction=best_ev,
        stake_usdc=round(stake, 2),
        market_id=market.condition_id,
        reason="ev_positive",
    )
