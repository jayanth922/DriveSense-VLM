"""Inference optimization and serving modules for DriveSense-VLM.

Submodules:
    merge_lora:    Merge LoRA adapters into base model weights (Phase 3a).
    quantize:      AWQ 4-bit quantization of the LLM decoder (Phase 3b).
    tensorrt_vit:  TensorRT compilation of the Vision Transformer (Phase 3c).
    serve:         vLLM production serving wrapper (Phase 3d).
"""

from __future__ import annotations
