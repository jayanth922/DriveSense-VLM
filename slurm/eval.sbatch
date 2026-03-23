#!/bin/bash
# =============================================================================
# eval.sbatch — SLURM job for DriveSense-VLM 4-level evaluation (Phase 4b)
# =============================================================================
# Usage:
#   sbatch slurm/eval.sbatch
#
# Depends on: completed training checkpoint in outputs/training/
# =============================================================================

# ── Resource Allocation ────────────────────────────────────────────────────────
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00           # Evaluation typically completes in 1–3 hours
#SBATCH --job-name=drivesense-eval
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --mail-user=your_email@sjsu.edu

# ── Environment Setup ──────────────────────────────────────────────────────────
module load cuda/12.1
module load anaconda3
conda activate drivesense

mkdir -p logs

# ── W&B ───────────────────────────────────────────────────────────────────────
# export WANDB_MODE=offline

export TOKENIZERS_PARALLELISM=false

cd $HOME/DriveSense-VLM

# ── Job Info ───────────────────────────────────────────────────────────────────
echo "============================================================"
echo " DriveSense-VLM — 4-Level Evaluation"
echo " Job ID:      $SLURM_JOB_ID"
echo " Node:        $SLURM_NODELIST"
echo " Start time:  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── Evaluation ─────────────────────────────────────────────────────────────────
python -m drivesense.eval --config configs/eval.yaml

echo "============================================================"
echo " Evaluation completed: $(date '+%Y-%m-%d %H:%M:%S')"
echo " Results: outputs/eval/"
echo "============================================================"
