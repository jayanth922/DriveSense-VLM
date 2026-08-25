#!/usr/bin/env python3
"""Upload DriveSense-VLM model artifacts to a HuggingFace Hub repo.

Pushes the quantized model weights, processor/tokenizer files, example
dashcam images, and the model card README to a model repo on HF Hub.

Usage:
    # With HF_TOKEN env var
    export HF_TOKEN=hf_xxx
    python scripts/upload_to_hf.py \\
        --model-dir outputs/quantized_model \\
        --processor-dir outputs/merged_model \\
        --examples-dir huggingface_space/examples \\
        --repo-id jayanth922/DriveSense-VLM

    # Or pass the token directly
    python scripts/upload_to_hf.py --token hf_xxx --model-dir outputs/quantized_model

Requirements:
    pip install huggingface_hub>=0.24
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("upload_to_hf")

_REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_REPO_ID = "jayanth922/DriveSense-VLM"
DEFAULT_MODEL_CARD = _REPO_ROOT / "hf_model_card" / "README.md"

MODEL_FILE_PATTERNS = (
    "*.safetensors",
    "*.json",
    "*.txt",
    "*.model",
    "*.bin",
    "tokenizer*",
    "special_tokens_map.json",
    "preprocessor_config.json",
    "chat_template*",
)
EXAMPLE_FILE_PATTERNS = ("*.jpg", "*.jpeg", "*.png")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        description="Upload DriveSense-VLM artifacts to HuggingFace Hub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to quantized model directory (contains *.safetensors + config.json).",
    )
    p.add_argument(
        "--processor-dir",
        type=Path,
        default=None,
        help="Optional path to processor/tokenizer dir (e.g. merged_model). "
             "If omitted, processor files are taken from --model-dir.",
    )
    p.add_argument(
        "--examples-dir",
        type=Path,
        default=_REPO_ROOT / "demo" / "examples",
        help="Path to example dashcam images. Uploaded under examples/ in the repo.",
    )
    p.add_argument(
        "--model-card",
        type=Path,
        default=DEFAULT_MODEL_CARD,
        help="Path to model card README.md to upload as repo README.",
    )
    p.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"HF Hub repo id (default: {DEFAULT_REPO_ID}).",
    )
    p.add_argument(
        "--token",
        default=None,
        help="HF token. Falls back to HF_TOKEN env var.",
    )
    p.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private (default: public).",
    )
    p.add_argument(
        "--commit-message",
        default="Upload DriveSense-VLM artifacts",
        help="Commit message for the upload.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List the files that would be uploaded without pushing.",
    )
    return p.parse_args()


def _resolve_token(cli_token: str | None) -> str:
    """Return the HF token from CLI flag or environment, or exit."""
    token = cli_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        logger.error(
            "No HuggingFace token. Set HF_TOKEN env var or pass --token."
        )
        sys.exit(2)
    return token


def _collect_model_files(model_dir: Path) -> list[Path]:
    """Return all files in model_dir matching MODEL_FILE_PATTERNS (deduped)."""
    if not model_dir.is_dir():
        logger.error("--model-dir not found: %s", model_dir)
        sys.exit(2)
    seen: set[Path] = set()
    files: list[Path] = []
    for pattern in MODEL_FILE_PATTERNS:
        for path in sorted(model_dir.glob(pattern)):
            if path.is_file() and path not in seen:
                seen.add(path)
                files.append(path)
    return files


def _collect_example_files(examples_dir: Path) -> list[Path]:
    """Return image files in examples_dir matching EXAMPLE_FILE_PATTERNS."""
    if not examples_dir.is_dir():
        logger.warning("--examples-dir not found: %s — skipping examples", examples_dir)
        return []
    files: list[Path] = []
    for pattern in EXAMPLE_FILE_PATTERNS:
        files.extend(sorted(examples_dir.glob(pattern)))
    return [p for p in files if p.is_file()]


def _print_upload_plan(
    repo_id: str,
    model_files: list[Path],
    processor_files: list[Path],
    example_files: list[Path],
    model_card: Path | None,
) -> None:
    """Log a human-readable summary of the upload plan."""
    logger.info("Upload plan for repo: %s", repo_id)
    logger.info("  Model files     (%d):", len(model_files))
    for f in model_files:
        logger.info("    - %s", f.name)
    if processor_files:
        logger.info("  Processor files (%d):", len(processor_files))
        for f in processor_files:
            logger.info("    - %s", f.name)
    logger.info("  Examples        (%d):", len(example_files))
    for f in example_files:
        logger.info("    - examples/%s", f.name)
    if model_card and model_card.exists():
        logger.info("  Model card     : %s -> README.md", model_card)
    elif model_card:
        logger.warning("  Model card     : %s NOT FOUND — skipping", model_card)


def main() -> None:
    """Entry point."""
    args = parse_args()

    model_files = _collect_model_files(args.model_dir)
    processor_files = (
        _collect_model_files(args.processor_dir) if args.processor_dir else []
    )
    example_files = _collect_example_files(args.examples_dir)

    if not model_files:
        logger.error("No model files found in %s", args.model_dir)
        sys.exit(2)

    _print_upload_plan(
        args.repo_id, model_files, processor_files, example_files, args.model_card
    )

    if args.dry_run:
        logger.info("--dry-run set — no upload performed")
        return

    token = _resolve_token(args.token)

    try:
        from huggingface_hub import HfApi, create_repo  # type: ignore[import]
    except ImportError:
        logger.error(
            "huggingface_hub not installed. Run: pip install huggingface_hub>=0.24"
        )
        sys.exit(2)

    api = HfApi(token=token)

    logger.info("Creating / verifying repo %s …", args.repo_id)
    create_repo(
        args.repo_id,
        token=token,
        private=args.private,
        repo_type="model",
        exist_ok=True,
    )

    for path in model_files:
        logger.info("Uploading model file: %s", path.name)
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=args.commit_message,
        )

    for path in processor_files:
        if path.name in {p.name for p in model_files}:
            continue
        logger.info("Uploading processor file: %s", path.name)
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=args.commit_message,
        )

    for path in example_files:
        logger.info("Uploading example: examples/%s", path.name)
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=f"examples/{path.name}",
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=args.commit_message,
        )

    if args.model_card and args.model_card.exists():
        logger.info("Uploading model card -> README.md")
        api.upload_file(
            path_or_fileobj=str(args.model_card),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=args.commit_message,
        )

    logger.info("Done. View at: https://huggingface.co/%s", args.repo_id)


if __name__ == "__main__":
    main()
