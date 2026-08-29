import logging
import sys
import colorlog
from logging.handlers import RotatingFileHandler
from config import LOG_LEVEL, BASE_DIR

LOG_FILE = BASE_DIR / "bot.log"

# Shared file handler so every logger writes the same rotating file. Plain text
# (no color codes) and a full date so the file is greppable after the fact —
# e.g. feature-importance dumps survive a terminal scrollback.
_file_handler = RotatingFileHandler(
    LOG_FILE, maxBytes=10_000_000, backupCount=3, encoding="utf-8"
)
_file_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))


def get_logger(name: str) -> logging.Logger:
    handler = colorlog.StreamHandler(sys.stdout)
    handler.setFormatter(colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        log_colors={
            "DEBUG":    "cyan",
            "INFO":     "green",
            "WARNING":  "yellow",
            "ERROR":    "red",
            "CRITICAL": "bold_red",
        },
    ))

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    if not logger.handlers:
        logger.addHandler(handler)
        logger.addHandler(_file_handler)
    logger.propagate = False
    return logger
