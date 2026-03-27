"""Entry point for ``python -m drivesense.training``.

Usage:
    python -m drivesense.training --config configs/training.yaml
    python -m drivesense.training --config configs/training.yaml --resume
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("drivesense.training")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DriveSense-VLM LoRA SFT training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="configs/training.yaml", help="Path to training.yaml")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint in the output dir",
    )
    return p.parse_args()


def main() -> None:
    """Parse args, optionally patch resume flag, and call :func:`train`."""
    args = _parse_args()

    try:
        from drivesense.training.sft_trainer import _load_all_configs, train
    except ImportError as exc:
        logger.error(
            "Training dependencies not installed: %s\n"
            "Install with: pip install -e '.[training]'",
            exc,
        )
        sys.exit(1)

    if args.resume:
        # Patch the config to use "latest" resume mode
        config = _load_all_configs(Path(args.config))
        training_cfg = config.get("training", {})
        training_cfg["resume_from_checkpoint"] = "latest"

    metrics = train(args.config)
    logger.info("Training complete: %s", metrics)


if __name__ == "__main__":
    main()
