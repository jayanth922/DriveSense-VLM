"""Entry point for ``python -m drivesense.training``.

Usage:
    python -m drivesense.training --config configs/training.yaml
    python -m drivesense.training --config configs/training.yaml --resume
"""

from __future__ import annotations

import argparse
import logging
import sys

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
    """Parse args and call :func:`train`, passing --resume as an explicit override."""
    args = _parse_args()

    try:
        from drivesense.training.sft_trainer import train
    except ImportError as exc:
        logger.error(
            "Training dependencies not installed: %s\n"
            "Install with: pip install -e '.[training]'",
            exc,
        )
        sys.exit(1)

    # train() ALWAYS reloads config from args.config, so mutating a locally-loaded
    # config dict here would be silently discarded (this was the --resume bug —
    # it never actually resumed). Pass the resume choice as an explicit override.
    metrics = train(args.config, resume_override="latest" if args.resume else None)
    logger.info("Training complete: %s", metrics)


if __name__ == "__main__":
    main()
