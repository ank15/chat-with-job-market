"""Application logging: one place to configure where logs go and how they look.

Logs stream to the console *and* to ``logs/app.log`` (rotating, so it never grows
unbounded). Call ``get_logger()`` anywhere to get the shared logger; configuration
runs once and is idempotent.
"""

import logging
from logging.handlers import RotatingFileHandler

from . import config

LOG_DIR = config.PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "app.log"
_LOGGER_NAME = "ragchat"
_configured = False


def configure_logging(level=logging.INFO):
    """Set up the shared logger (console + rotating file). Safe to call repeatedly."""
    global _configured
    logger = logging.getLogger(_LOGGER_NAME)
    if _configured:
        return logger

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=3)
    file_handler.setFormatter(formatter)

    logger.setLevel(level)
    logger.handlers.clear()
    logger.addHandler(console)
    logger.addHandler(file_handler)
    logger.propagate = False  # don't double-log via the root logger

    _configured = True
    return logger


def get_logger():
    """Return the configured application logger."""
    return configure_logging()
