#!/bin/bash
# =============================================================================
# benchmark.sbatch — SLURM job for inference latency benchmarking (Phase 3d/4b)
# =============================================================================
# Measures end-to-end latency, throughput, and VRAM for Level 3 evaluation.
# Requires quantized model in outputs/quantized_model/ (Phase 3b complete).
#
# Usage:
#   sbatch slurm/benchmark.sbatch
# =============================================================================

# ── Resource Allocation ────────────────────────────────────────────────────────
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2:00:00           # Benchmark runs in 30–60 minutes
#SBATCH --job-name=drivesense-bench
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --mail-user=your_email@sjsu.edu

# ── Environment Setup ──────────────────────────────────────────────────────────
module load cuda/12.1
module load anaconda3
conda activate drivesense

mkdir -p logs

export TOKENIZERS_PARALLELISM=false

cd $HOME/DriveSense-VLM

# ── Job Info ───────────────────────────────────────────────────────────────────
echo "============================================================"
echo " DriveSense-VLM — Inference Benchmark"
echo " Job ID:      $SLURM_JOB_ID"
echo " Node:        $SLURM_NODELIST"
echo " Start time:  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# ── Benchmark ──────────────────────────────────────────────────────────────────
mkdir -p outputs/benchmarks

# Local transformers backend (always available once model is merged)
python scripts/run_benchmark.py --local --output outputs/benchmarks/local_bench.json

# vLLM backend (requires quantized model)
python scripts/run_benchmark.py --vllm --output outputs/benchmarks/vllm_bench.json

# ViT-only (requires TensorRT engine)
python scripts/run_benchmark.py --vit-only --output outputs/benchmarks/vit_bench.json

echo "============================================================"
echo " Benchmark completed: $(date '+%Y-%m-%d %H:%M:%S')"
echo " Results: outputs/benchmarks/"
echo "============================================================"
