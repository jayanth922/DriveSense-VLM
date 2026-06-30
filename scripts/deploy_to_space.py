#!/usr/bin/env python3
"""Deploy the DriveSense-VLM Gradio demo to a HuggingFace Space.

Uploads everything under ``huggingface_space/`` (app.py, requirements.txt,
README.md, optional examples/) to a HuggingFace Space repo, creating the Space
if it does not yet exist.

Usage:
    python scripts/deploy_to_space.py --token hf_xxx
    python scripts/deploy_to_space.py --token hf_xxx --space-id user/my-demo
    HF_TOKEN=hf_xxx python scripts/deploy_to_space.py

The token needs write access to the target namespace. Get one at
https://huggingface.co/settings/tokens (role: write).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_SPACE_DIR = _REPO_ROOT / "huggingface_space"
_DEFAULT_SPACE_ID = "jayanth7111/DriveSense-VLM-demo"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Deploy huggingface_space/ to a HuggingFace Space.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace write token (defaults to the HF_TOKEN env var).",
    )
    p.add_argument(
        "--space-id",
        default=_DEFAULT_SPACE_ID,
        help=f"Target Space repo id, e.g. user/name (default: {_DEFAULT_SPACE_ID}).",
    )
    p.add_argument(
        "--space-dir",
        default=str(_DEFAULT_SPACE_DIR),
        help="Local directory to upload (default: huggingface_space/).",
    )
    p.add_argument(
        "--private",
        action="store_true",
        help="Create the Space as private (default: public).",
    )
    p.add_argument(
        "--factory-reboot",
        action="store_true",
        help="Trigger a factory reboot after upload (wipes the build cache).",
    )
    return p.parse_args()


def deploy(
    token: str,
    space_id: str,
    space_dir: Path,
    private: bool,
    factory_reboot: bool = False,
) -> str:
    """Create (if needed), upload to, and optionally factory-reboot the Space.

    Args:
        token:          HuggingFace write token.
        space_id:       Target Space repo id (``user/name``).
        space_dir:      Local directory whose contents are uploaded.
        private:        Whether to create the Space as private.
        factory_reboot: If True, request a factory reboot after the upload so
            the build runs from a clean cache (fixes stuck/broken builds).

    Returns:
        The URL of the deployed Space.
    """
    from huggingface_hub import HfApi  # noqa: PLC0415

    api = HfApi(token=token)
    api.create_repo(
        repo_id=space_id,
        repo_type="space",
        space_sdk="gradio",
        private=private,
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=str(space_dir),
        repo_id=space_id,
        repo_type="space",
        commit_message="Deploy DriveSense-VLM demo",
    )
    if factory_reboot:
        print("Requesting factory reboot …")
        api.restart_space(repo_id=space_id, factory_reboot=True)
    return f"https://huggingface.co/spaces/{space_id}"


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    if not args.token:
        print(
            "ERROR: no token provided. Pass --token or set HF_TOKEN.",
            file=sys.stderr,
        )
        sys.exit(1)

    space_dir = Path(args.space_dir)
    if not (space_dir / "app.py").exists():
        print(
            f"ERROR: {space_dir}/app.py not found — nothing to deploy.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Deploying {space_dir} → space '{args.space_id}' …")
    url = deploy(
        args.token, args.space_id, space_dir, args.private, args.factory_reboot
    )
    print(f"✓ Deployed: {url}")


if __name__ == "__main__":
    main()
