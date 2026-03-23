#!/bin/bash
# =============================================================================
# train.sbatch — SLURM job for DriveSense-VLM LoRA SFT training (Phase 2a)
# =============================================================================
# Usage:
#   sbatch slurm/train.sbatch
#
# Monitor:
#   squeue -u $USER
#   tail -f logs/slurm-<JOB_ID>.out
# =============================================================================

# ── Resource Allocation ────────────────────────────────────────────────────────
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1             # Request 1 GPU (A100 80GB or H100)
#SBATCH --mem=48G                # 48 GB system RAM (model + data loading)
#SBATCH --time=48:00:00          # 48-hour walltime — well under 7-day HPC limit
#SBATCH --job-name=drivesense-train
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --mail-user=your_email@sjsu.edu  # Uncomment and set your email

# ── Environment Setup ──────────────────────────────────────────────────────────
module load cuda/12.1
module load anaconda3
conda activate drivesense

mkdir -p logs

# ── W&B ───────────────────────────────────────────────────────────────────────
# export WANDB_MODE=offline  # Uncomment if HPC nodes have no internet

# ── CUDA Memory Optimization ───────────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TOKENIZERS_PARALLELISM=false

cd $HOME/DriveSense-VLM

# ── Job Info ───────────────────────────────────────────────────────────────────
echo "============================================================"
echo " DriveSense-VLM — SFT Training"
echo " Job ID:      $SLURM_JOB_ID"
echo " Node:        $SLURM_NODELIST"
echo " Start time:  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── Training ───────────────────────────────────────────────────────────────────
python -m drivesense.training.sft_trainer --config configs/training.yaml

echo "============================================================"
echo " Training completed: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
