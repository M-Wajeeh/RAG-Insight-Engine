import logging
import sys


_DEFAULT_NAME = "rag"
_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"


def setup_logger(name: str = _DEFAULT_NAME, level: int = logging.INFO) -> logging.Logger:
    """Return a configured stdout logger.

    Idempotent: calling with the same name reuses handlers instead of duplicating them.
    """
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="replace")

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        logger.addHandler(handler)

    return logger
