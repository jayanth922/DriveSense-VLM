# SLURM Scripts (Alternative to Colab)

These SLURM batch scripts are provided for users with access to HPC clusters
(e.g., SJSU CoE A100/H100 nodes). They are kept for reference and remain
fully functional for HPC-based workflows.

**The primary execution method for this project is Google Colab Pro.**
See [`notebooks/`](../notebooks/) for the Colab-based workflow.

## Available Scripts

| Script | Phase | Description |
|--------|-------|-------------|
| `train.sbatch` | 2a | SFT training (A100, ~8 h) |
| `optimize.sbatch` | 3a+3b | LoRA merge + AWQ + TensorRT (~4 h) |
| `benchmark.sbatch` | 3c | Inference benchmark (~2 h) |
| `eval.sbatch` | 2b+4b | Levels 1–2 evaluation |

## Usage (HPC)

```bash
# Submit jobs in order
sbatch slurm/train.sbatch
sbatch slurm/optimize.sbatch
sbatch slurm/benchmark.sbatch
sbatch slurm/eval.sbatch
```

## Colab Equivalents

| SLURM script | Colab notebook |
|--------------|----------------|
| `train.sbatch` | `notebooks/01_training.ipynb` |
| `optimize.sbatch` | `notebooks/02_optimization.ipynb` |
| `benchmark.sbatch` | `notebooks/03_benchmark.ipynb` |
| `eval.sbatch` | `notebooks/04_evaluation.ipynb` |
