#!/usr/bin/env python3
"""Ship DriveSense-VLM to both remotes in one go: GitHub + the HF Space.

GitHub and the HuggingFace Space are separate remotes — pushing to one does
not update the other. This wraps both into a single command.

Steps (each can be skipped):
  1. (optional) Stage all changes and commit with ``-m``.
  2. ``git push`` the current branch to GitHub.
  3. Upload ``huggingface_space/`` to the HF Space (optionally factory-reboot).

Usage:
    python scripts/ship.py                      # push branch + deploy Space
    python scripts/ship.py -m "fix demo"        # commit everything first, then both
    python scripts/ship.py --factory-reboot     # + clean-cache rebuild of the Space
    python scripts/ship.py --skip-hf            # GitHub only
    python scripts/ship.py --skip-github        # HF Space only

The HF step uses --token, then $HF_TOKEN, then the locally saved token
(~/.cache/huggingface/token from `huggingface-cli login`).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
_DEFAULT_SPACE_DIR = _REPO_ROOT / "huggingface_space"
_DEFAULT_SPACE_ID = "jayanth7111/DriveSense-VLM-demo"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Push to GitHub and deploy to the HF Space in one command.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "-m", "--message",
        default=None,
        help="If set, stage all changes and commit with this message before pushing.",
    )
    p.add_argument("--remote", default="origin", help="Git remote (default: origin).")
    p.add_argument(
        "--branch",
        default=None,
        help="Branch to push (default: current branch).",
    )
    p.add_argument("--skip-github", action="store_true", help="Do not push to GitHub.")
    p.add_argument("--skip-hf", action="store_true", help="Do not deploy to the HF Space.")
    # HF Space options (mirror deploy_to_space.py)
    p.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="HF write token (defaults to $HF_TOKEN, then the saved login token).",
    )
    p.add_argument(
        "--space-id", default=_DEFAULT_SPACE_ID,
        help=f"Target Space repo id (default: {_DEFAULT_SPACE_ID}).",
    )
    p.add_argument(
        "--space-dir", default=str(_DEFAULT_SPACE_DIR),
        help="Local directory to upload (default: huggingface_space/).",
    )
    p.add_argument(
        "--factory-reboot", action="store_true",
        help="Factory-reboot the Space after upload (wipes the build cache).",
    )
    p.add_argument("--private", action="store_true", help="Create the Space as private.")
    return p.parse_args()


def _git(*args: str) -> None:
    """Run a git command in the repo root, raising on failure."""
    subprocess.run(["git", "-C", str(_REPO_ROOT), *args], check=True)


def _current_branch() -> str:
    """Return the current git branch name."""
    out = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "rev-parse", "--abbrev-ref", "HEAD"],
        check=True, capture_output=True, text=True,
    )
    return out.stdout.strip()


def push_github(remote: str, branch: str, message: str | None) -> None:
    """Optionally commit, then push the branch to GitHub.

    Args:
        remote:  Git remote name.
        branch:  Branch to push.
        message: If set, ``git add -A`` + commit before pushing.
    """
    if message:
        _git("add", "-A")
        _git("commit", "-m", message)
    print(f"→ git push {remote} {branch}")
    _git("push", remote, branch)
    print("✓ GitHub: pushed")


def deploy_hf(args: argparse.Namespace) -> None:
    """Deploy huggingface_space/ to the HF Space via deploy_to_space.deploy()."""
    if str(_SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS_DIR))
    import deploy_to_space  # noqa: PLC0415

    print(f"→ deploy {args.space_dir} → space '{args.space_id}'")
    url = deploy_to_space.deploy(
        token=args.token,
        space_id=args.space_id,
        space_dir=Path(args.space_dir),
        private=args.private,
        factory_reboot=args.factory_reboot,
    )
    print(f"✓ HF Space: {url}")


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    if args.skip_github and args.skip_hf:
        print("Nothing to do: both --skip-github and --skip-hf set.", file=sys.stderr)
        sys.exit(1)

    if not args.skip_github:
        branch = args.branch or _current_branch()
        push_github(args.remote, branch, args.message)

    if not args.skip_hf:
        space_dir = Path(args.space_dir)
        if not (space_dir / "app.py").exists():
            print(f"ERROR: {space_dir}/app.py not found — nothing to deploy.", file=sys.stderr)
            sys.exit(1)
        deploy_hf(args)

    print("Done.")


if __name__ == "__main__":
    main()
