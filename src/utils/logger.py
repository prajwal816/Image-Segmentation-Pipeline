"""Centralized logging configuration for the segmentation pipeline."""

import logging
import os
import sys
from datetime import datetime


def get_logger(
    name: str,
    log_dir: str = "experiments/logs",
    level: str = "INFO",
    console: bool = True,
    file: bool = True,
) -> logging.Logger:
    """Create and configure a logger instance.

    Args:
        name: Logger name (typically module name).
        log_dir: Directory to save log files.
        level: Logging level string.
        console: Whether to add console handler.
        file: Whether to add file handler.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if file:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f"{name}_{timestamp}.log"),
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
