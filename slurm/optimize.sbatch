#!/bin/bash
# =============================================================================
# optimize.sbatch — SLURM job for DriveSense-VLM Phase 3a+3b optimization
# =============================================================================
# Usage:
#   sbatch slurm/optimize.sbatch
#
# Depends on: completed LoRA adapter in outputs/training/checkpoint-best
# Monitor:
#   squeue -u $USER
#   tail -f logs/optimize-<JOB_ID>.out
# =============================================================================

# ── Resource Allocation ────────────────────────────────────────────────────────
#SBATCH --job-name=drivesense-optimize
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=logs/optimize-%j.out
#SBATCH --error=logs/optimize-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --mail-user=your_email@sjsu.edu

# ── Environment Setup ──────────────────────────────────────────────────────────
module load cuda/12.1
module load anaconda3
conda activate drivesense

mkdir -p logs outputs/merged_model outputs/quantized_model outputs/tensorrt

# ── W&B ───────────────────────────────────────────────────────────────────────
export WANDB_PROJECT="drivesense-vlm"
# export WANDB_MODE=offline  # Uncomment if compute nodes have no internet

export TOKENIZERS_PARALLELISM=false

cd $HOME/DriveSense-VLM

# ── Job Info ───────────────────────────────────────────────────────────────────
echo "============================================================"
echo " DriveSense-VLM — Phase 3a+3b Optimization Pipeline"
echo " Job ID:      $SLURM_JOB_ID"
echo " Node:        $SLURM_NODELIST"
echo " Start time:  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── Full Optimization Pipeline ────────────────────────────────────────────────
echo ""
echo "--- Running full optimization pipeline ---"
python scripts/run_optimize_model.py --all \
    --adapter-path outputs/training/checkpoint-best \
    2>&1 | tee logs/optimize-${SLURM_JOB_ID}.log

echo ""
echo "============================================================"
echo " Optimization complete: $(date '+%Y-%m-%d %H:%M:%S')"
echo " Merged model:     outputs/merged_model/"
echo " Quantized model:  outputs/quantized_model/"
echo " TensorRT engine:  outputs/tensorrt/vit.engine"
echo " Benchmark report: outputs/tensorrt/optimization_report.txt"
echo "============================================================"
