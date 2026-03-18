from __future__ import annotations

import logging
import os

from rich.logging import RichHandler


def get_logger(name: str) -> logging.Logger:
    """Return a logger with consistent formatting using rich console output."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        level = os.environ.get("LOG_LEVEL", "INFO").upper()
        logger.setLevel(getattr(logging, level, logging.INFO))

        handler = RichHandler(
            rich_tracebacks=True,
            show_path=False,
            markup=True,
        )
        handler.setLevel(logger.level)

        fmt = logging.Formatter("%(message)s", datefmt="[%X]")
        handler.setFormatter(fmt)

        logger.addHandler(handler)
        logger.propagate = False

    return logger
