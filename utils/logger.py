from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path


LOGGER_FORMAT = "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s"
LOGGER_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(log_dir: str) -> Path:
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    logfile = log_path / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    formatter = logging.Formatter(LOGGER_FORMAT, datefmt=LOGGER_DATE_FORMAT)
    file_handler = logging.FileHandler(logfile, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
    return logfile


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
