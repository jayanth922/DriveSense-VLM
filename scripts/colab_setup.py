"""
Colab environment setup utility.

Handles Google Drive mounting, repo cloning, dependency installation,
symlink creation, and GPU verification.

Usage in Colab notebooks::

    !pip install pyyaml -q  # minimal bootstrap dep
    %cd /content
    !git clone https://github.com/jayanth922/DriveSense-VLM.git \
        /content/DriveSense-VLM 2>/dev/null || true
    %cd /content/DriveSense-VLM
    from scripts.colab_setup import setup_colab
    env = setup_colab(gpu_required="A100")
    # env contains: project_root, drive_root, gpu_name, gpu_vram_gb, drive_mounted
"""

from __future__ import annotations

import glob  # noqa: I001
import os
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def setup_colab(
    github_repo: str = "jayanth922/DriveSense-VLM",
    drive_root: str = "/content/drive/MyDrive/DriveSense-VLM",
    gpu_required: str | None = "A100",
    install_groups: list[str] | None = None,
) -> dict:
    """One-call Colab environment setup.

    Steps:

    1. Mount Google Drive
    2. Create Drive project directory structure
    3. Clone repo to /content/ (ephemeral, fast I/O for code)
    4. Create symlinks: data/ and outputs/ → Drive (persistent)
    5. Install pip dependencies if *install_groups* specified
    6. Verify GPU if *gpu_required* is not None
    7. Print environment summary

    Args:
        github_repo: ``"owner/repo"`` — used to clone the repo. Replace
            ``jayanth922`` with your GitHub handle.
        drive_root: Absolute path on mounted Google Drive where project data
            and outputs are persisted.
        gpu_required: ``"A100"``, ``"T4"``, or ``None`` (no GPU check).
            Raises ``RuntimeError`` if the mounted GPU does not match.
        install_groups: pip extras to install, e.g. ``["training", "data"]``.
            Runs ``pip install -e ".[group1,group2]" -q``.

    Returns:
        dict with keys ``project_root``, ``drive_root``, ``gpu_name``,
        ``gpu_vram_gb``, ``drive_mounted``.

    Raises:
        RuntimeError: If a required GPU is not available.
    """
    print("=" * 60)
    print("DriveSense-VLM — Colab Setup")
    print("=" * 60)

    drive_mounted = _mount_drive()
    _create_drive_dirs(drive_root)

    project_root = _clone_repo(github_repo)
    _create_symlinks(project_root, drive_root)

    if install_groups:
        _install_deps(project_root, install_groups)

    gpu_name, gpu_vram_gb = _verify_gpu(gpu_required)

    env = {
        "project_root": project_root,
        "drive_root": drive_root,
        "gpu_name": gpu_name,
        "gpu_vram_gb": gpu_vram_gb,
        "drive_mounted": drive_mounted,
    }
    _print_summary(env)
    return env


def check_and_resume_checkpoint(output_dir: str) -> str | None:
    """Check for existing training checkpoints on Drive.

    Args:
        output_dir: Path to training outputs directory (on Google Drive).

    Returns:
        Path to the latest checkpoint directory, or ``None`` if none found.
    """
    pattern = str(Path(output_dir) / "checkpoint-*")
    checkpoints = sorted(glob.glob(pattern))
    if not checkpoints:
        print("No checkpoints found — starting fresh training.")
        return None

    latest = checkpoints[-1]
    print(f"Found {len(checkpoints)} checkpoint(s). Latest: {latest}")
    print("Training will RESUME automatically from the latest checkpoint.")
    return latest


def display_compute_budget() -> None:
    """Print estimated compute unit usage for a full DriveSense-VLM run.

    Helps users track their Colab Pro budget (~200 CU total).
    """
    budget = [
        ("Data pipeline (T4, ~3 h)", 5),
        ("SFT training (A100, ~6 h)", 72),
        ("Optimization (A100, ~1.5 h)", 18),
        ("Benchmarks (A100, ~1 h)", 12),
        ("Evaluation (A100, ~1 h)", 12),
    ]
    total_est = sum(cu for _, cu in budget)
    total_budget = 200

    print("=" * 48)
    print(f"  Compute Budget — Colab Pro (~{total_budget} CU total)")
    print("=" * 48)
    for stage, cu in budget:
        bar = "#" * cu
        print(f"  {stage:<38} {cu:>3} CU  {bar}")
    print("-" * 48)
    print(f"  Estimated total:                         {total_est:>3} CU")
    print(f"  Buffer for reruns/debugging:             {total_budget - total_est:>3} CU")
    print("=" * 48)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _mount_drive() -> bool:
    """Mount Google Drive. No-op outside Colab."""
    try:
        from google.colab import drive  # type: ignore[import]

        drive.mount("/content/drive")
        print("[setup] Google Drive mounted at /content/drive")
        return True
    except ImportError:
        print("[setup] Not running in Colab — skipping Drive mount.")
        return False


def _create_drive_dirs(drive_root: str) -> None:
    """Create persistent directory structure on Google Drive."""
    dirs = [
        "",
        "data/nuscenes",
        "data/dada2000",
        "outputs/data",
        "outputs/training",
        "outputs/merged_model",
        "outputs/quantized_model",
        "outputs/tensorrt",
        "outputs/benchmarks",
        "outputs/eval",
        "outputs/predictions",
    ]
    root = Path(drive_root)
    for d in dirs:
        (root / d).mkdir(parents=True, exist_ok=True)
    print(f"[setup] Drive directory structure created at {drive_root}")


def _clone_repo(github_repo: str) -> str:
    """Clone repo to /content/DriveSense-VLM (ephemeral fast SSD)."""
    project_root = "/content/DriveSense-VLM"
    if Path(project_root).exists():
        print(f"[setup] Repo already cloned at {project_root} — skipping.")
    else:
        url = f"https://github.com/{github_repo}.git"
        print(f"[setup] Cloning {url} …")
        subprocess.run(["git", "clone", url, project_root], check=True)
        print(f"[setup] Repo cloned to {project_root}")

    # Always make sure we're on the latest commit
    subprocess.run(["git", "-C", project_root, "pull", "--quiet"], check=False)

    # Add project root to sys.path so `from scripts.colab_setup import …` works
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    os.chdir(project_root)
    return project_root


def _create_symlinks(project_root: str, drive_root: str) -> None:
    """Symlink data/ and outputs/ into the repo → Google Drive."""
    repo = Path(project_root)
    drive = Path(drive_root)

    for name in ("data", "outputs"):
        link = repo / name
        target = drive / name
        target.mkdir(parents=True, exist_ok=True)
        if link.is_symlink():
            link.unlink()
        elif link.exists():
            # Directory already exists from a previous clone — skip
            print(f"[setup] {link} already exists as a real directory — skipping symlink.")
            continue
        link.symlink_to(target)
        print(f"[setup] Symlinked {link} → {target}")


def _install_deps(project_root: str, groups: list[str]) -> None:
    """Install pip extras from the project's pyproject.toml."""
    extras = ",".join(groups)
    cmd = [sys.executable, "-m", "pip", "install", "-e", f".[{extras}]", "-q"]
    print(f"[setup] Installing deps: pip install -e '[{extras}]' …")
    t0 = time.time()
    subprocess.run(cmd, cwd=project_root, check=True)
    print(f"[setup] Dependencies installed in {time.time() - t0:.0f}s")


def _verify_gpu(gpu_required: str | None) -> tuple[str, float]:
    """Verify GPU availability and optionally assert the required type.

    Returns:
        ``(gpu_name, vram_gb)`` — empty string and 0.0 when no GPU.
    """
    try:
        import torch  # type: ignore[import]
    except ImportError:
        print("[setup] torch not installed — skipping GPU check.")
        return "", 0.0

    if not torch.cuda.is_available():
        if gpu_required:
            raise RuntimeError(
                "No GPU detected! "
                f"Fix: Runtime → Change runtime type → {gpu_required} GPU"
            )
        print("[setup] No GPU available (not required for this notebook).")
        return "", 0.0

    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9

    print(f"[setup] GPU: {name}")
    print(f"[setup] VRAM: {vram:.1f} GB")

    if gpu_required and gpu_required.upper() not in name.upper():
        print(
            f"[setup] WARNING: Expected {gpu_required} but got {name}. "
            "Training will work but may be slower or OOM on smaller GPUs."
        )

    return name, vram


def _print_summary(env: dict) -> None:
    """Print a human-readable environment summary."""
    print()
    print("=" * 60)
    print("  Environment Summary")
    print("=" * 60)
    print(f"  Project root (ephemeral) : {env['project_root']}")
    print(f"  Drive root (persistent)  : {env['drive_root']}")
    print(f"  Drive mounted            : {env['drive_mounted']}")
    if env["gpu_name"]:
        print(f"  GPU                      : {env['gpu_name']}")
        print(f"  VRAM                     : {env['gpu_vram_gb']:.1f} GB")
    else:
        print("  GPU                      : None (CPU only)")
    print("=" * 60)
    print()
    print("RECOVERY: If Colab disconnects, rerun setup cells — training")
    print("auto-resumes from the latest checkpoint on Google Drive.")
    print()
