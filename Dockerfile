# DriveSense-VLM — reproducible pipeline image (data + eval + MLOps).
# CPU image: runs mining, auto-labeling (API), dataset build, L1/L4 eval, compare, and the
# regression gate. For GPU training/inference use the --build-arg below to swap the base.
#
#   docker build -t drivesense .
#   docker run --rm -e ANTHROPIC_API_KEY=$KEY drivesense \
#       python scripts/run_evaluation.py --level 1 \
#          --predictions results/v4/test_pred_full.jsonl \
#          --ground-truth data/sft_test_enriched.jsonl
#
# GPU training image (needs an NVIDIA runtime):
#   docker build --build-arg BASE=pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime -t drivesense-gpu .

ARG BASE=python:3.11-slim
FROM ${BASE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    TRANSFORMERS_VERBOSITY=error

WORKDIR /app

# System deps kept minimal; add build-essential only if a wheel needs compiling.
RUN apt-get update -qq && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Core runtime deps (data + eval + MLOps). Training/inference extras are opt-in to keep the
# CPU image small; install them in the GPU variant:
#   pip install "transformers<5" peft accelerate bitsandbytes qwen-vl-utils torch
COPY requirements-pipeline.txt* ./
RUN pip install --no-cache-dir \
      pyyaml pillow tqdm numpy requests scipy ijson anthropic

COPY . .

# Default: print the eval help so `docker run drivesense` is self-documenting.
CMD ["python", "scripts/run_evaluation.py", "--help"]
