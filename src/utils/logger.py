"""
Shared logger factory for the whole project.

Using one helper instead of repeated logging.basicConfig() calls keeps
formatting consistent across scripts and source modules. It also makes it
easy to write logs both to stdout and to a file when running on HPC.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path


def get_logger(
    name: str,
    log_dir: str | Path | None = None,
    level: str = "INFO",
) -> logging.Logger:
    """
    Create or return a configured logger.

    Parameters
    ----------
    name:
        Logger name, usually __name__ from the calling module.
    log_dir:
        Optional directory for timestamped log files.
    level:
        Logging level as a string, e.g. "INFO" or "DEBUG".
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers if the logger was already created earlier
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file logging for long local runs or SLURM jobs
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{name.replace('.', '_')}_{timestamp}.log"

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger