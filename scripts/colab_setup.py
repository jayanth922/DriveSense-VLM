"""
Colab environment setup utility.

Handles Google Drive mounting, repo cloning, dependency installation,
symlink creation, and GPU verification.

Usage in Colab notebooks::

    import sys, os
    REPO_ROOT = "/content/DriveSense-VLM"
    if not os.path.exists(REPO_ROOT):
        os.system(f"git clone https://github.com/jayanth922/DriveSense-VLM.git {REPO_ROOT}")
    sys.path.insert(0, f"{REPO_ROOT}/src")
    sys.path.insert(0, REPO_ROOT)
    from scripts.colab_setup import setup_colab
    env = setup_colab(gpu_required="A100")
    # env: project_root, drive_root, gpu_name, gpu_vram_gb, drive_mounted
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
    packages: list[str] | None = None,
) -> dict:
    """One-call Colab environment setup.

    Steps:

    1. Mount Google Drive
    2. Create Drive project directory structure
    3. Clone/update repo to /content/ (ephemeral, fast I/O for code)
    4. Symlink data/ and outputs/ → Drive (persistent)
    5. Add /content/DriveSense-VLM/src to sys.path
    6. Install pip packages directly if *packages* specified
    7. Verify GPU if *gpu_required* is not None
    8. Print environment summary

    .. important::

        **Never** pass ``numpy``, ``pandas``, or ``nuscenes-devkit`` (without
        ``--no-deps``) in *packages*.  Colab ships numpy 2.x / pandas 2.x
        pre-installed; reinstalling them causes binary ABI mismatches.
        Use :func:`install_data_deps`, :func:`install_training_deps`, etc.
        for the correct per-notebook package sets.

    Args:
        github_repo: ``"owner/repo"`` string used to clone the repo.
        drive_root: Absolute path on mounted Google Drive for persistent data.
        gpu_required: ``"A100"``, ``"T4"``, or ``None`` (skip GPU check).
        packages: List of pip package specs to install directly (not extras).

    Returns:
        dict with keys ``project_root``, ``drive_root``, ``gpu_name``,
        ``gpu_vram_gb``, ``drive_mounted``.
    """
    print("=" * 60)
    print("DriveSense-VLM — Colab Setup")
    print("=" * 60)

    drive_mounted = _mount_drive()
    _create_drive_dirs(drive_root)

    project_root = _clone_repo(github_repo)
    _create_symlinks(project_root, drive_root)
    _add_to_path(project_root)

    if packages:
        _install_packages(packages)

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


def install_data_deps() -> None:
    """Install data-pipeline dependencies for Notebook 00.

    Does NOT install numpy or pandas (uses Colab pre-installed versions).
    Installs nuscenes-devkit with ``--no-deps`` to avoid its strict numpy version pin.
    """
    _upgrade_pip()
    _run_pip(["nuscenes-devkit", "--no-deps"])
    _run_pip(["pyquaternion", "matplotlib", "tqdm", "Pillow",
               "pyyaml", "requests", "openpyxl"])
    _run_pip(["scikit-learn", "scipy", "pyspark"])


def install_training_deps() -> None:
    """Install training dependencies for Notebook 01."""
    _upgrade_pip()
    _run_pip(["pyyaml", "tqdm", "Pillow", "requests"])
    _run_pip(["transformers", "peft", "accelerate", "datasets",
               "bitsandbytes", "wandb"])


def install_optimization_deps() -> None:
    """Install optimization dependencies for Notebook 02."""
    _upgrade_pip()
    _run_pip(["pyyaml", "tqdm", "Pillow", "requests"])
    _run_pip(["transformers", "peft", "accelerate"])
    _run_pip(["autoawq"])
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "tensorrt", "-q"],
        check=False,
    )


def install_benchmark_deps() -> None:
    """Install benchmark dependencies for Notebook 03."""
    _upgrade_pip()
    _run_pip(["pyyaml", "tqdm", "Pillow", "requests"])
    _run_pip(["transformers", "peft", "accelerate"])
    _run_pip(["vllm"])


def install_eval_deps() -> None:
    """Install evaluation dependencies for Notebook 04."""
    _upgrade_pip()
    _run_pip(["pyyaml", "tqdm", "Pillow", "requests"])
    _run_pip(["transformers", "peft", "accelerate"])
    _run_pip(["scikit-learn", "scipy"])


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
    """Print estimated compute unit usage for a full DriveSense-VLM run."""
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
        print(f"  {stage:<38} {cu:>3} CU  {'#' * cu}")
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
        "outputs/data/sft_ready",
    ]
    root = Path(drive_root)
    for d in dirs:
        (root / d).mkdir(parents=True, exist_ok=True)
    print(f"[setup] Drive directory structure created at {drive_root}")


def _clone_repo(github_repo: str) -> str:
    """Clone or update repo to /content/DriveSense-VLM."""
    project_root = "/content/DriveSense-VLM"
    if Path(project_root).exists():
        print(f"[setup] Repo exists at {project_root} — pulling latest.")
        subprocess.run(
            ["git", "-C", project_root, "pull", "--quiet"], check=False
        )
    else:
        url = f"https://github.com/{github_repo}.git"
        print(f"[setup] Cloning {url} ...")
        subprocess.run(["git", "clone", url, project_root], check=True)
        print(f"[setup] Repo cloned to {project_root}")

    os.chdir(project_root)
    return project_root


def _add_to_path(project_root: str) -> None:
    """Add project src to sys.path (replaces broken editable install)."""
    src_path = str(Path(project_root) / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    print(f"[setup] sys.path: added {src_path}")


def _create_symlinks(project_root: str, drive_root: str) -> None:
    """Symlink data/ and outputs/ inside the repo → Google Drive."""
    repo = Path(project_root)
    drive = Path(drive_root)

    for name in ("data", "outputs"):
        link = repo / name
        target = drive / name
        target.mkdir(parents=True, exist_ok=True)
        if link.is_symlink():
            link.unlink()
        elif link.exists() and not link.is_symlink():
            print(f"[setup] {link} is a real directory — skipping symlink.")
            continue
        link.symlink_to(target)
        print(f"[setup] Symlinked {link} → {target}")


def _upgrade_pip() -> None:
    """Upgrade pip, setuptools, wheel."""
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade",
         "pip", "setuptools", "wheel", "-q"],
        check=True,
    )


def _run_pip(packages: list[str]) -> None:
    """Install a list of packages directly (no project extras)."""
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", *packages],
        check=True,
    )


def _install_packages(packages: list[str]) -> None:
    """Upgrade pip then install provided packages."""
    t0 = time.time()
    _upgrade_pip()
    print(f"[setup] Installing: {' '.join(packages)}")
    _run_pip(packages)
    print(f"[setup] Packages installed in {time.time() - t0:.0f}s")


def _verify_gpu(gpu_required: str | None) -> tuple[str, float]:
    """Verify GPU availability and optionally assert the required type."""
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
        print("[setup] No GPU (not required for this notebook).")
        return "", 0.0

    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[setup] GPU: {name}  ({vram:.1f} GB)")

    if gpu_required and gpu_required.upper() not in name.upper():
        print(
            f"[setup] WARNING: Expected {gpu_required} but got {name}. "
            "May OOM on GPUs < 40 GB."
        )
    return name, vram


def _print_summary(env: dict) -> None:
    """Print a human-readable environment summary."""
    print()
    print("=" * 60)
    print("  Environment Ready")
    print("=" * 60)
    print(f"  Repo  (ephemeral) : {env['project_root']}")
    print(f"  Drive (persistent): {env['drive_root']}")
    print(f"  Drive mounted     : {env['drive_mounted']}")
    if env["gpu_name"]:
        print(f"  GPU               : {env['gpu_name']}")
        print(f"  VRAM              : {env['gpu_vram_gb']:.1f} GB")
    else:
        print("  GPU               : None (CPU only)")
    print("=" * 60)
    print("RECOVERY: If Colab disconnects, rerun setup cells —")
    print("training auto-resumes from the latest Drive checkpoint.")
    print()
