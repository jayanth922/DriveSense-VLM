"""Structured logging setup for DriveSense-VLM training and inference.

Configures Python's standard logging with a consistent format that includes
timestamps, module name, and log level. Also provides a W&B-aware logger
that forwards structured metric dicts to W&B when a run is active.

Usage:
    from drivesense.utils.logging import get_logger

    logger = get_logger(__name__)
    logger.info("Starting training epoch %d", epoch)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

# W&B — optional at import time; only used when a run is active
try:
    import wandb  # type: ignore[import]
except ImportError:
    wandb = None  # type: ignore[assignment]

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def setup_logging(
    level: int = logging.INFO,
    log_file: Path | None = None,
) -> None:
    """Configure root logger with console and optional file handlers.

    Idempotent — calling multiple times has no effect after first call.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO).
        log_file: Optional path to a log file. If None, logs only to stdout.
    """
    global _configured
    if _configured:
        return

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    root = logging.getLogger()
    root.setLevel(level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Get a named logger, initialising logging with defaults if not yet done.

    Args:
        name: Logger name — use ``__name__`` for module-level loggers.

    Returns:
        Configured Logger instance.
    """
    setup_logging()
    return logging.getLogger(name)


def log_metrics(metrics: dict[str, Any], step: int | None = None) -> None:
    """Log a dict of metrics to both Python logging and W&B (if active).

    Args:
        metrics: Dict mapping metric name to scalar value.
        step: Optional global step counter for W&B x-axis alignment.
    """
    logger = get_logger(__name__)
    metric_str = " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                             for k, v in metrics.items())
    logger.info("Metrics: %s", metric_str)

    if wandb is not None and wandb.run is not None:
        wandb.log(metrics, step=step)
