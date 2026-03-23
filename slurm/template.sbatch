#!/bin/bash
# =============================================================================
# template.sbatch — Base SLURM job template for DriveSense-VLM
# =============================================================================
# Do NOT submit this file directly. Use train.sbatch, eval.sbatch, or
# benchmark.sbatch which inherit these settings and add job-specific flags.
#
# To submit a job:
#   sbatch slurm/train.sbatch
#   sbatch slurm/eval.sbatch
#   sbatch slurm/benchmark.sbatch
# =============================================================================

# ── Resource Allocation (base — override in specific scripts) ──────────────────
#SBATCH --partition=gpu          # Adjust to your HPC partition name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# ── Environment Setup ──────────────────────────────────────────────────────────
module load cuda/12.1
module load anaconda3
conda activate drivesense

# ── Ensure log directory exists ────────────────────────────────────────────────
mkdir -p logs

# ── W&B Configuration ──────────────────────────────────────────────────────────
# Uncomment below if the HPC compute nodes have no outbound internet access.
# Offline mode saves runs locally; sync after job: wandb sync wandb/offline-run-*/
# export WANDB_MODE=offline

# ── Change to project directory ────────────────────────────────────────────────
cd $HOME/DriveSense-VLM

# ── Print job info ─────────────────────────────────────────────────────────────
echo "============================================================"
echo " Job ID:      $SLURM_JOB_ID"
echo " Job Name:    $SLURM_JOB_NAME"
echo " Node:        $SLURM_NODELIST"
echo " Partition:   $SLURM_JOB_PARTITION"
echo " Start time:  $(date '+%Y-%m-%d %H:%M:%S')"
echo " Working dir: $(pwd)"
echo "============================================================"

# GPU info
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true
