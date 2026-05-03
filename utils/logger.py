import logging
import sys


def setup_logger(name="rag_logger", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # Format
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger