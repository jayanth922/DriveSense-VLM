"""TensorRT compilation of the Qwen3-VL Vision Transformer encoder.

Implements Phase 3c: exports the ViT backbone to ONNX, optimises it with
TensorRT in fp16 precision, and saves the compiled engine. Fixed batch size
(no dynamic axes) ensures deterministic latency on HPC inference hardware.

Pipeline:
    merged model (fp16 ViT) → ONNX export → trtexec / TRT Python API → .engine

Usage (HPC only — requires CUDA + TensorRT):
    python -m drivesense.inference.tensorrt_vit --config configs/inference.yaml

Implementation target: Phase 3c
"""

from __future__ import annotations

from pathlib import Path

# TensorRT — HPC only (NVIDIA SDK install)
try:
    import tensorrt as trt  # type: ignore[import]
except ImportError:
    trt = None  # type: ignore[assignment]

# PyTorch — HPC only
try:
    import torch  # type: ignore[import]
except ImportError:
    torch = None  # type: ignore[assignment]


def export_vit_to_onnx(
    model_dir: Path,
    onnx_path: Path,
    input_shape: tuple[int, int, int, int],
) -> Path:
    """Export the ViT encoder from the merged model to ONNX format.

    Args:
        model_dir: Path to merged (Phase 3a) or quantized (Phase 3b) model.
        onnx_path: Output path for the ONNX file.
        input_shape: Input tensor shape as (batch, channels, height, width).

    Returns:
        Path to the saved ONNX file.
    """
    raise NotImplementedError("Phase 3c: extract ViT submodule and call torch.onnx.export()")


def compile_onnx_to_engine(
    onnx_path: Path,
    engine_path: Path,
    config: dict,
) -> Path:
    """Compile ONNX ViT model to TensorRT engine in fp16 precision.

    Args:
        onnx_path: Path to the ONNX file from export_vit_to_onnx().
        engine_path: Output path for the TensorRT .engine file.
        config: Inference config dict from configs/inference.yaml ['tensorrt'].

    Returns:
        Path to the saved TensorRT engine file.
    """
    raise NotImplementedError("Phase 3c: build TRT network from ONNX and serialize engine")


def benchmark_engine(engine_path: Path, input_shape: tuple[int, int, int, int]) -> dict:
    """Measure TensorRT ViT engine latency statistics.

    Args:
        engine_path: Path to the compiled TensorRT engine.
        input_shape: Input shape as (batch, channels, height, width).

    Returns:
        Dict of latency stats:
        ``{"p50_ms": float, "p95_ms": float, "p99_ms": float, "mean_ms": float}``.
    """
    raise NotImplementedError("Phase 3c: run warmup + timed inference loops on TRT engine")
