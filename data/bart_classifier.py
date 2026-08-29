"""
Shared BART MNLI zero-shot classifier singleton.
Loads the model once — all sentiment modules import from here.
"""

from utils.logger import get_logger

log = get_logger("bart_classifier")

_classifier = None


def get_classifier():
    global _classifier
    if _classifier is not None:
        return _classifier
    try:
        from transformers import pipeline
        log.info(
            "Loading BART MNLI model — first run downloads ~1.6 GB, please wait..."
        )
        _classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1,
            batch_size=8,
        )
        log.info("BART MNLI model ready.")
    except ImportError:
        log.warning(
            "transformers/torch not installed — BART sentiment disabled. "
            "Run: pip install transformers torch"
        )
    except Exception as e:
        log.warning(f"BART MNLI load failed: {e}")
    return _classifier
