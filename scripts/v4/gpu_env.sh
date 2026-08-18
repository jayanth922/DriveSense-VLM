#!/usr/bin/env bash
# Restore the v4 GPU training env after a pod restart.  Usage:  source /workspace/v4/gpu_env.sh
cd /workspace/DriveSense-VLM
export PYTHONPATH=/workspace/DriveSense-VLM/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/workspace/hf_cache          # cache the base model on the volume -> no re-download on restart
pip install -q "transformers<5" peft accelerate bitsandbytes pillow pyyaml qwen-vl-utils numpy scipy tqdm
python3 -c "import torch,transformers;print('torch',torch.__version__,'| transformers',transformers.__version__,'| cuda',torch.cuda.is_available())"
