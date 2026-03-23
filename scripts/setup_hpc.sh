#!/usr/bin/env bash
# =============================================================================
# setup_hpc.sh — One-time HPC environment setup for DriveSense-VLM
# =============================================================================
# Usage:
#   bash scripts/setup_hpc.sh
#
# Run this script ONCE on the SJSU CoE HPC login/submit node after cloning
# the repository. It creates the conda environment "drivesense", installs all
# training dependencies, and verifies the installation.
#
# Requirements:
#   - SLURM login node with internet access
#   - Anaconda3 module available (module load anaconda3)
#   - CUDA 12.1 module available (module load cuda/12.1)
#   - ~20 GB free disk space in $HOME or $SCRATCH
# =============================================================================

set -euo pipefail

ENV_NAME="drivesense"
PYTHON_VERSION="3.10"
CUDA_VERSION="12.1"

echo "============================================================"
echo " DriveSense-VLM HPC Environment Setup"
echo "============================================================"
echo " Conda env:   ${ENV_NAME}"
echo " Python:      ${PYTHON_VERSION}"
echo " CUDA:        ${CUDA_VERSION}"
echo " Working dir: $(pwd)"
echo "============================================================"
echo ""

# ── Load required modules ──────────────────────────────────────────────────────
echo "[1/5] Loading HPC modules..."
module load cuda/${CUDA_VERSION} || {
    echo "WARNING: cuda/${CUDA_VERSION} module not found. Trying cuda..." >&2
    module load cuda || echo "WARNING: No CUDA module loaded — verify manually." >&2
}
module load anaconda3 || module load anaconda || {
    echo "ERROR: No anaconda module found. Contact HPC support." >&2
    exit 1
}

# ── Create conda environment ───────────────────────────────────────────────────
echo ""
echo "[2/5] Creating conda environment '${ENV_NAME}'..."

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "  Conda env '${ENV_NAME}' already exists — skipping creation."
    echo "  To recreate: conda env remove -n ${ENV_NAME} && bash scripts/setup_hpc.sh"
else
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
    echo "  Created conda env: ${ENV_NAME}"
fi

# ── Install PyTorch with CUDA ──────────────────────────────────────────────────
echo ""
echo "[3/5] Installing PyTorch 2.1+ with CUDA ${CUDA_VERSION}..."

conda run -n "${ENV_NAME}" conda install -y \
    pytorch torchvision torchaudio pytorch-cuda=12.1 \
    -c pytorch -c nvidia

echo "  PyTorch installed."

# ── Install DriveSense package in editable mode ────────────────────────────────
echo ""
echo "[4/5] Installing DriveSense-VLM package (training + data + eval extras)..."

# Editable install with HPC-relevant dependency groups
conda run -n "${ENV_NAME}" pip install -e ".[training,data,eval]" \
    --no-cache-dir \
    --find-links https://download.pytorch.org/whl/cu121

# Install flash-attn separately (requires torch to be installed first)
echo "  Installing flash-attn (requires CUDA, may take several minutes)..."
conda run -n "${ENV_NAME}" pip install flash-attn --no-build-isolation || {
    echo "  WARNING: flash-attn installation failed." >&2
    echo "  The model will fall back to 'sdpa' attention (configs/model.yaml)." >&2
}

echo "  DriveSense package installed."

# ── Verify installation ────────────────────────────────────────────────────────
echo ""
echo "[5/5] Verifying installation..."
echo ""

# Check PyTorch + CUDA
CUDA_CHECK=$(conda run -n "${ENV_NAME}" python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
" 2>&1)
echo "  ${CUDA_CHECK}"
echo ""

# Check DriveSense package
DS_CHECK=$(conda run -n "${ENV_NAME}" python -c "
import drivesense
print(f'DriveSense v{drivesense.__version__}')
" 2>&1)
echo "  ${DS_CHECK}"
echo ""

# Run sanity check
echo "  Running sanity check..."
conda run -n "${ENV_NAME}" python scripts/run_sanity_check.py || {
    echo "  WARNING: Sanity check reported issues — review above output." >&2
}

# ── Print activation instructions ─────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Setup Complete!"
echo "============================================================"
echo ""
echo " Activate the environment:"
echo "   conda activate ${ENV_NAME}"
echo ""
echo " Submit training job:"
echo "   sbatch slurm/train.sbatch"
echo ""
echo " Download datasets (if not yet done):"
echo "   bash scripts/download_nuscenes.sh mini     # ~4 GB dev"
echo "   bash scripts/download_nuscenes.sh trainval # ~60 GB full"
echo "   bash scripts/download_dada2000.sh"
echo ""
echo " Monitor training:"
echo "   squeue -u \$USER"
echo "   tail -f logs/slurm-<job_id>.out"
echo "============================================================"
