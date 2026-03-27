#!/bin/bash
# =============================================================================
# eval.sbatch — SLURM job for DriveSense-VLM Phase 2b evaluation
# =============================================================================
# Usage:
#   sbatch slurm/eval.sbatch
#
# Depends on: completed training checkpoint in outputs/training/
# Monitor:
#   squeue -u $USER
#   tail -f logs/slurm-<JOB_ID>.out
# =============================================================================

# ── Resource Allocation ────────────────────────────────────────────────────────
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --job-name=drivesense-eval
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --mail-user=your_email@sjsu.edu

# ── Environment Setup ──────────────────────────────────────────────────────────
module load cuda/12.1
module load anaconda3
conda activate drivesense

mkdir -p logs outputs/predictions

# ── W&B ───────────────────────────────────────────────────────────────────────
export WANDB_PROJECT="drivesense-vlm"
# export WANDB_MODE=offline  # Uncomment if compute nodes have no internet

export TOKENIZERS_PARALLELISM=false

cd $HOME/DriveSense-VLM

# ── Job Info ───────────────────────────────────────────────────────────────────
echo "============================================================"
echo " DriveSense-VLM — Phase 2b Evaluation"
echo " Job ID:      $SLURM_JOB_ID"
echo " Node:        $SLURM_NODELIST"
echo " Start time:  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── Step 1: Generate predictions on test split ─────────────────────────────────
echo ""
echo "--- Step 1: Generating predictions ---"
python scripts/run_generate_predictions.py --split test \
    2>&1 | tee logs/generate-${SLURM_JOB_ID}.log

# ── Step 2: Level 1 — Grounding accuracy ──────────────────────────────────────
echo ""
echo "--- Step 2: Level 1 (Grounding) evaluation ---"
python scripts/run_evaluation.py --level 1 \
    2>&1 | tee logs/eval-level1-${SLURM_JOB_ID}.log

# ── Step 3: Level 2 — Reasoning quality (mock judge, no API key needed) ───────
echo ""
echo "--- Step 3: Level 2 (Reasoning) evaluation ---"
# Set ANTHROPIC_API_KEY for real LLM judge, or use --mock-judge for testing:
# export ANTHROPIC_API_KEY="your-key-here"
python scripts/run_evaluation.py --level 2 --mock-judge \
    2>&1 | tee logs/eval-level2-${SLURM_JOB_ID}.log

echo ""
echo "============================================================"
echo " Evaluation completed: $(date '+%Y-%m-%d %H:%M:%S')"
echo " Results: outputs/eval/"
echo "============================================================"
