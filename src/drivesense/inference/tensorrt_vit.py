"""Stage 3: TensorRT compilation of Qwen3-VL vision encoder.

Compiles the ViT to a TensorRT engine with fixed input resolution (672x448)
for deterministic, low-latency visual feature extraction.

Pipeline: PyTorch ViT → ONNX export → TensorRT engine

Fallback path: If ONNX export fails (custom ops) or TensorRT is unavailable,
falls back to torch.compile(mode="reduce-overhead") and documents the choice
in fallback_info.json.

Implemented in Phase 3b.
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HPC-only imports — guarded for local macOS dev
# ---------------------------------------------------------------------------

try:
    import torch as _torch
    import torch.nn as _nn
    _TORCH_AVAILABLE = True
except ImportError:
    _torch = None  # type: ignore[assignment]
    _nn = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

try:
    import tensorrt as _trt  # type: ignore[import]
    _TRT_AVAILABLE = True
except ImportError:
    _trt = None  # type: ignore[assignment]
    _TRT_AVAILABLE = False

try:
    import onnx as _onnx  # type: ignore[import]
    _ONNX_AVAILABLE = True
except ImportError:
    _onnx = None  # type: ignore[assignment]
    _ONNX_AVAILABLE = False

try:
    import onnxsim as _onnxsim  # type: ignore[import]
    _ONNXSIM_AVAILABLE = True
except ImportError:
    _onnxsim = None  # type: ignore[assignment]
    _ONNXSIM_AVAILABLE = False

try:
    from transformers import AutoProcessor as _AutoProcessor  # type: ignore[import]
    from transformers import Qwen2_5_VLForConditionalGeneration as _VLMClass  # type: ignore[import]
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    try:
        from transformers import AutoProcessor as _AutoProcessor  # type: ignore[import]  # noqa: I001
        from transformers import Qwen2VLForConditionalGeneration as _VLMClass  # type: ignore[import]  # noqa: I001
        _TRANSFORMERS_AVAILABLE = True
    except ImportError:
        _AutoProcessor = None  # type: ignore[assignment]
        _VLMClass = None  # type: ignore[assignment]
        _TRANSFORMERS_AVAILABLE = False

# Default fixed input shape: [batch=1, channels=3, height=448, width=672]
_DEFAULT_INPUT_SHAPE: tuple[int, int, int, int] = (1, 3, 448, 672)


# ---------------------------------------------------------------------------
# ViT wrapper module for ONNX export
# ---------------------------------------------------------------------------


class _ViTWrapper(_nn.Module if _nn is not None else object):  # type: ignore[misc]
    """Wrap Qwen3-VL visual encoder for ONNX export at fixed resolution.

    Accepts a standard [B, C, H, W] image tensor. Internally converts to
    the patch-sequence format expected by Qwen3-VL's ViT and returns the
    visual embedding sequence.

    Args:
        vit:      The visual encoder module from Qwen3-VL.
        grid_thw: Fixed grid [1, n_height_patches, n_width_patches].
    """

    def __init__(self, vit: Any, grid_thw: Any) -> None:
        super().__init__()
        self.vit = vit
        if _torch is not None:
            self.register_buffer("grid_thw", grid_thw)
        else:
            self.grid_thw = grid_thw

    def forward(self, pixel_values: Any) -> Any:  # type: ignore[override]
        """Forward pass: pixel_values [B, C, H, W] → visual embeddings.

        Args:
            pixel_values: Standard image tensor [B, C, H, W].

        Returns:
            Visual embedding tensor from the ViT.
        """
        return self.vit(pixel_values, grid_thw=self.grid_thw)


# ---------------------------------------------------------------------------
# ViTExtractor
# ---------------------------------------------------------------------------


class ViTExtractor:
    """Extract, compile, and benchmark the Qwen3-VL vision encoder.

    Full pipeline:
    1. Extract ViT submodule from full model
    2. Export to ONNX (opset 17, fixed shape, with fallback)
    3. Compile to TensorRT fp16 engine (with torch.compile fallback)
    4. Benchmark all available optimization levels
    5. Save artifacts + reports

    Args:
        config: Merged config dict (inference section required).
    """

    def __init__(self, config: dict) -> None:
        trt_cfg = config.get("tensorrt", {})
        self._output_dir = Path(trt_cfg.get("output_dir", "outputs/tensorrt"))
        self._onnx_path = Path(trt_cfg.get("onnx_path", "outputs/tensorrt/vit.onnx"))
        self._engine_path = Path(
            trt_cfg.get("engine_path", "outputs/tensorrt/vit.engine")
        )
        raw_shape = trt_cfg.get("input_shape", list(_DEFAULT_INPUT_SHAPE))
        self._input_shape: tuple[int, int, int, int] = tuple(raw_shape)  # type: ignore[assignment]
        self._precision: str = trt_cfg.get("precision", "fp16")
        self._max_workspace_gb: int = int(trt_cfg.get("max_workspace_size_gb", 4))
        self._dynamic_batch: bool = bool(trt_cfg.get("dynamic_batch", False))
        self._cfg = config

    # ── extraction ──────────────────────────────────────────────────────────

    def extract_vit(self, model_dir: Path) -> tuple[Any, Any]:
        """Extract the vision encoder submodule from Qwen3-VL.

        Args:
            model_dir: Path to merged model directory.

        Returns:
            ``(vit_module, processor)`` tuple.
        """
        _require_torch_transformers()
        model = _VLMClass.from_pretrained(  # type: ignore[union-attr]
            str(model_dir),
            torch_dtype=_torch.float16,  # type: ignore[union-attr]
            device_map="auto",
        )
        processor = _AutoProcessor.from_pretrained(str(model_dir))  # type: ignore[union-attr]
        vit = _get_vision_encoder(model)
        logger.info("Extracted ViT: %s", type(vit).__name__)
        return vit, processor

    # ── ONNX export ─────────────────────────────────────────────────────────

    def export_to_onnx(
        self,
        model_dir: Path,
        onnx_path: Path | None = None,
        input_shape: tuple[int, int, int, int] | None = None,
    ) -> Path:
        """Export the ViT to ONNX with fixed resolution.

        Tries direct export; if custom operators cause failure, falls back to
        torch.jit.trace → ONNX.

        Args:
            model_dir:   Path to merged model directory.
            onnx_path:   Output path (default from config).
            input_shape: [batch, channels, height, width]. Default: [1,3,448,672].

        Returns:
            Path to the saved ONNX file.
        """
        _require_torch_transformers()
        out_path = Path(onnx_path) if onnx_path else self._onnx_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shape = input_shape if input_shape is not None else self._input_shape

        vit, _ = self.extract_vit(model_dir)
        grid_thw = _compute_grid_thw(shape)
        wrapper = _ViTWrapper(vit, grid_thw)
        wrapper.eval()

        dummy = _torch.zeros(  # type: ignore[union-attr]
            shape,
            dtype=_torch.float16,  # type: ignore[union-attr]
            device=next(vit.parameters()).device,
        )

        fallback_info: dict[str, Any] = {"onnx_method": "direct", "trt_method": None}

        # Attempt 1: direct torch.onnx.export
        try:
            _export_onnx_direct(wrapper, dummy, out_path)
            logger.info("ONNX export succeeded (direct): %s", out_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Direct ONNX export failed: %s — trying torch.jit.trace fallback", exc
            )
            fallback_info["onnx_method"] = "jit_trace"
            fallback_info["onnx_direct_error"] = str(exc)
            try:
                _export_onnx_via_trace(wrapper, dummy, out_path)
                logger.info("ONNX export succeeded (jit.trace): %s", out_path)
            except Exception as exc2:  # noqa: BLE001
                logger.error("Both ONNX export paths failed: %s", exc2)
                fallback_info["onnx_method"] = "failed"
                fallback_info["onnx_trace_error"] = str(exc2)
                _save_fallback_info(self._output_dir, fallback_info)
                raise RuntimeError(
                    f"ONNX export failed. See fallback_info.json. Last error: {exc2}"
                ) from exc2

        # Validate and optionally simplify
        if _ONNX_AVAILABLE and _onnx is not None and out_path.exists():
            _validate_and_simplify_onnx(out_path)

        _save_fallback_info(self._output_dir, fallback_info)
        return out_path

    # ── TensorRT compilation ─────────────────────────────────────────────────

    def compile_tensorrt(
        self,
        onnx_path: Path,
        engine_path: Path | None = None,
        precision: str | None = None,
        max_workspace_gb: int | None = None,
    ) -> Path:
        """Compile ONNX ViT model to a TensorRT fp16 engine.

        Falls back to torch.compile if TensorRT is unavailable.

        Args:
            onnx_path:       Path to ONNX model.
            engine_path:     Output path for .engine file (default from config).
            precision:       "fp16" or "fp32" (default from config).
            max_workspace_gb: TRT workspace size (default from config).

        Returns:
            Path to TensorRT engine file (or torch.compile sentinel path).
        """
        out_path = Path(engine_path) if engine_path else self._engine_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        prec = precision or self._precision
        ws_gb = max_workspace_gb if max_workspace_gb is not None else self._max_workspace_gb

        fallback_path = self._output_dir / "fallback_info.json"
        fallback_info = json.loads(fallback_path.read_text()) if fallback_path.exists() else {}

        if _TRT_AVAILABLE and _trt is not None:
            try:
                _build_trt_engine(onnx_path, out_path, prec, ws_gb)
                fallback_info["trt_method"] = "tensorrt"
                _save_fallback_info(self._output_dir, fallback_info)
                return out_path
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "TensorRT compilation failed: %s — falling back to torch.compile", exc
                )
                fallback_info["trt_method"] = "torch_compile"
                fallback_info["trt_error"] = str(exc)
        else:
            logger.info("TensorRT not available — using torch.compile fallback")
            fallback_info["trt_method"] = "torch_compile"
            fallback_info["trt_note"] = "tensorrt package not installed"

        # torch.compile fallback — save sentinel file
        compile_sentinel = out_path.with_suffix(".torch_compile")
        compile_sentinel.write_text(
            json.dumps({
                "method": "torch_compile",
                "mode": "reduce-overhead",
                "note": (
                    "TensorRT unavailable or failed. "
                    "Use torch.compile(vit_module, mode='reduce-overhead') at runtime."
                ),
            }),
            encoding="utf-8",
        )
        fallback_info["torch_compile_sentinel"] = str(compile_sentinel)
        _save_fallback_info(self._output_dir, fallback_info)
        logger.info("torch.compile fallback documented at %s", compile_sentinel)
        return compile_sentinel

    # ── benchmarking ─────────────────────────────────────────────────────────

    def benchmark_vit(
        self,
        model_dir: Path,
        engine_path: Path | None = None,
        num_iterations: int = 100,
        warmup: int = 10,
    ) -> dict:
        """Benchmark ViT latency across all available optimization levels.

        Measures: PyTorch eager, torch.compile, TensorRT (if available).

        Args:
            model_dir:      Path to merged model directory.
            engine_path:    Path to TensorRT engine (or None to skip TRT).
            num_iterations: Timed iterations per backend.
            warmup:         Warmup iterations before timing.

        Returns:
            Full benchmark result dict with per-backend latency statistics.
        """
        _require_torch_transformers()
        shape = self._input_shape
        eng_path = Path(engine_path) if engine_path else self._engine_path

        vit, _ = self.extract_vit(model_dir)
        grid_thw = _compute_grid_thw(shape)
        wrapper = _ViTWrapper(vit, grid_thw)
        wrapper.eval()
        device = next(vit.parameters()).device
        dummy = _torch.zeros(shape, dtype=_torch.float16, device=device)  # type: ignore[union-attr]

        results: dict[str, Any] = {
            "input_shape": list(shape),
            "num_iterations": num_iterations,
            "pytorch_eager": None,
            "torch_compile": None,
            "tensorrt": None,
            "speedup_compile_vs_eager": None,
            "speedup_tensorrt_vs_eager": None,
        }

        # 1. PyTorch eager baseline
        eager_stats = _time_inference(wrapper, dummy, warmup, num_iterations)
        results["pytorch_eager"] = eager_stats
        logger.info(
            "Eager: mean=%.1f ms  p95=%.1f ms",
            eager_stats["mean_ms"], eager_stats["p95_ms"],
        )

        # 2. torch.compile
        try:
            compiled = _torch.compile(wrapper, mode="reduce-overhead")  # type: ignore[union-attr]
            compile_stats = _time_inference(compiled, dummy, warmup, num_iterations)
            results["torch_compile"] = compile_stats
            eager_mean = eager_stats["mean_ms"]
            if eager_mean > 0:
                results["speedup_compile_vs_eager"] = round(
                    eager_mean / compile_stats["mean_ms"], 2
                )
            logger.info(
                "torch.compile: mean=%.1f ms  speedup=%.2fx",
                compile_stats["mean_ms"],
                results["speedup_compile_vs_eager"] or 0,
            )
        except Exception as exc:  # noqa: BLE001
            logger.info("torch.compile benchmark skipped: %s", exc)

        # 3. TensorRT engine
        if eng_path.exists() and _TRT_AVAILABLE and _trt is not None:
            try:
                trt_stats = _benchmark_trt_engine(eng_path, shape, warmup, num_iterations)
                results["tensorrt"] = trt_stats
                eager_mean = eager_stats["mean_ms"]
                if eager_mean > 0:
                    results["speedup_tensorrt_vs_eager"] = round(
                        eager_mean / trt_stats["mean_ms"], 2
                    )
                logger.info(
                    "TensorRT: mean=%.1f ms  speedup=%.2fx",
                    trt_stats["mean_ms"],
                    results["speedup_tensorrt_vs_eager"] or 0,
                )
            except Exception as exc:  # noqa: BLE001
                logger.info("TensorRT benchmark skipped: %s", exc)

        return results

    # ── full pipeline ─────────────────────────────────────────────────────────

    def full_pipeline(self, model_dir: Path, output_dir: Path) -> dict:
        """Run the complete ViT optimization pipeline.

        Stages:
        1. Export to ONNX (with fallback)
        2. Compile to TensorRT (with torch.compile fallback)
        3. Benchmark all backends
        4. Save artifacts and reports

        Creates::

            output_dir/
            ├── vit.onnx
            ├── vit.engine  (or vit.torch_compile if TRT failed)
            ├── vit_benchmark.json
            ├── optimization_report.txt
            └── fallback_info.json

        Args:
            model_dir:  Path to merged model directory.
            output_dir: Where to write all artifacts.

        Returns:
            Benchmark results dict.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        onnx_path = out / "vit.onnx"
        engine_path = out / "vit.engine"

        # Step 1: ONNX export
        onnx_path = self.export_to_onnx(model_dir, onnx_path=onnx_path)

        # Step 2: TensorRT (or torch.compile fallback)
        engine_path = self.compile_tensorrt(onnx_path, engine_path=engine_path)

        # Step 3: Benchmark
        benchmark = self.benchmark_vit(
            model_dir, engine_path=engine_path if engine_path.suffix == ".engine" else None
        )
        (out / "vit_benchmark.json").write_text(
            json.dumps(benchmark, indent=2), encoding="utf-8"
        )

        # Step 4: Human-readable report
        fallback_path = out / "fallback_info.json"
        fallback_info = json.loads(fallback_path.read_text()) if fallback_path.exists() else {}
        report_path = out / "optimization_report.txt"
        report_path.write_text(
            _format_optimization_report(benchmark, fallback_info), encoding="utf-8"
        )

        logger.info("ViT optimization pipeline complete — see %s", out)
        return benchmark


# ---------------------------------------------------------------------------
# Module-level convenience functions (kept from original stub)
# ---------------------------------------------------------------------------


def export_vit_to_onnx(
    model_dir: Path,
    onnx_path: Path,
    input_shape: tuple[int, int, int, int],
) -> Path:
    """Export the ViT encoder to ONNX format.

    Args:
        model_dir:   Path to merged model.
        onnx_path:   Output path for ONNX file.
        input_shape: [batch, channels, height, width].

    Returns:
        Path to saved ONNX file.
    """
    extractor = ViTExtractor({"tensorrt": {"input_shape": list(input_shape)}})
    return extractor.export_to_onnx(model_dir, onnx_path, input_shape)


def compile_onnx_to_engine(onnx_path: Path, engine_path: Path, config: dict) -> Path:
    """Compile ONNX ViT to TensorRT engine.

    Args:
        onnx_path:   Path to ONNX file.
        engine_path: Output path for engine file.
        config:      Inference config dict.

    Returns:
        Path to compiled engine.
    """
    extractor = ViTExtractor(config)
    return extractor.compile_tensorrt(onnx_path, engine_path)


def benchmark_engine(
    engine_path: Path,
    input_shape: tuple[int, int, int, int],
) -> dict:
    """Measure TensorRT ViT engine latency statistics.

    Args:
        engine_path: Path to compiled TensorRT engine.
        input_shape: [batch, channels, height, width].

    Returns:
        Latency stats dict with p50_ms, p95_ms, p99_ms, mean_ms.
    """
    if not _TRT_AVAILABLE or _trt is None:
        raise ImportError("TensorRT not available")
    return _benchmark_trt_engine(engine_path, input_shape, warmup=5, n_iter=50)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_torch_transformers() -> None:
    """Raise ImportError if torch or transformers are missing."""
    missing = []
    if not _TORCH_AVAILABLE:
        missing.append("torch")
    if not _TRANSFORMERS_AVAILABLE:
        missing.append("transformers")
    if missing:
        raise ImportError(
            f"HPC dependencies not available: {', '.join(missing)}. "
            "Install on HPC: pip install -e '.[training]'"
        )


def _get_vision_encoder(model: Any) -> Any:
    """Return the ViT submodule from Qwen3-VL.

    Qwen3-VL stores the vision encoder at model.visual.

    Args:
        model: Loaded Qwen3-VL model.

    Returns:
        Vision encoder nn.Module.
    """
    for attr in ("visual", "vision_model", "encoder"):
        if hasattr(model, attr):
            return getattr(model, attr)
    # Fallback: search named children
    for name, module in model.named_children():
        if any(k in name.lower() for k in ("visual", "vision", "vit")):
            return module
    raise AttributeError(
        "Cannot locate vision encoder in model. "
        "Inspect model.named_modules() to find the correct attribute."
    )


def _compute_grid_thw(input_shape: tuple[int, int, int, int]) -> Any:
    """Compute Qwen3-VL grid_thw tensor for a fixed input resolution.

    Qwen3-VL uses 28×28 patches. For a 672×448 image:
    height_patches = 448 / 28 = 16, width_patches = 672 / 28 = 24.

    Args:
        input_shape: [batch, channels, height, width].

    Returns:
        LongTensor of shape [1, 3] = [[1, h_patches, w_patches]].
    """
    _, _, h, w = input_shape
    patch_size = 28
    h_patches = h // patch_size
    w_patches = w // patch_size
    return _torch.tensor([[1, h_patches, w_patches]], dtype=_torch.long)  # type: ignore[union-attr]


def _export_onnx_direct(wrapper: Any, dummy: Any, out_path: Path) -> None:
    """Export wrapper to ONNX using torch.onnx.export directly."""
    _torch.onnx.export(  # type: ignore[union-attr]
        wrapper,
        dummy,
        str(out_path),
        opset_version=17,
        input_names=["pixel_values"],
        output_names=["visual_embeddings"],
        dynamic_axes=None,
        do_constant_folding=True,
    )


def _export_onnx_via_trace(wrapper: Any, dummy: Any, out_path: Path) -> None:
    """Export wrapper via torch.jit.trace → ONNX as fallback."""
    traced = _torch.jit.trace(wrapper, dummy)  # type: ignore[union-attr]
    _torch.onnx.export(  # type: ignore[union-attr]
        traced,
        dummy,
        str(out_path),
        opset_version=17,
        input_names=["pixel_values"],
        output_names=["visual_embeddings"],
        dynamic_axes=None,
    )


def _validate_and_simplify_onnx(onnx_path: Path) -> None:
    """Validate ONNX model and optionally simplify with onnxsim."""
    try:
        model_proto = _onnx.load(str(onnx_path))  # type: ignore[union-attr]
        _onnx.checker.check_model(model_proto)  # type: ignore[union-attr]
        logger.info("ONNX validation passed")
    except Exception as exc:  # noqa: BLE001
        logger.warning("ONNX validation warning: %s", exc)
        return

    if _ONNXSIM_AVAILABLE and _onnxsim is not None:
        try:
            simplified, ok = _onnxsim.simplify(model_proto)  # type: ignore[union-attr]
            if ok:
                _onnx.save(simplified, str(onnx_path))  # type: ignore[union-attr]
                logger.info("ONNX model simplified with onnxsim")
        except Exception as exc:  # noqa: BLE001
            logger.debug("onnxsim simplification skipped: %s", exc)


def _build_trt_engine(
    onnx_path: Path,
    engine_path: Path,
    precision: str,
    max_workspace_gb: int,
) -> None:
    """Build and serialize a TensorRT engine from ONNX.

    Args:
        onnx_path:       Path to ONNX model.
        engine_path:     Output path for serialized engine.
        precision:       "fp16" or "fp32".
        max_workspace_gb: TRT workspace limit in GB.
    """
    trt_logger = _trt.Logger(_trt.Logger.WARNING)  # type: ignore[union-attr]
    builder = _trt.Builder(trt_logger)  # type: ignore[union-attr]
    network_flags = 1 << int(_trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)  # type: ignore[union-attr]
    network = builder.create_network(network_flags)
    parser = _trt.OnnxParser(network, trt_logger)  # type: ignore[union-attr]

    with onnx_path.open("rb") as fh:
        ok = parser.parse(fh.read())
    if not ok:
        errors = [parser.get_error(i) for i in range(parser.num_errors)]
        raise RuntimeError(f"TensorRT ONNX parse failed: {errors}")

    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(
        _trt.MemoryPoolType.WORKSPACE,  # type: ignore[union-attr]
        max_workspace_gb * (1 << 30),
    )
    if precision == "fp16":
        cfg.set_flag(_trt.BuilderFlag.FP16)  # type: ignore[union-attr]

    logger.info("Building TensorRT engine (this may take several minutes)…")
    engine = builder.build_serialized_network(network, cfg)
    if engine is None:
        raise RuntimeError("TensorRT engine build returned None")

    with engine_path.open("wb") as fh:
        fh.write(engine)
    logger.info("TensorRT engine saved: %s (%.1f MB)", engine_path,
                engine_path.stat().st_size / (1024 ** 2))

    # Verify engine loads
    runtime = _trt.Runtime(trt_logger)  # type: ignore[union-attr]
    with engine_path.open("rb") as fh:
        _ = runtime.deserialize_cuda_engine(fh.read())
    logger.info("TensorRT engine verified (loads successfully)")


def _time_inference(
    model: Any,
    dummy: Any,
    warmup: int,
    n_iter: int,
) -> dict[str, float]:
    """Time n_iter forward passes; return latency statistics in ms.

    Args:
        model:  PyTorch model to benchmark.
        dummy:  Input tensor.
        warmup: Number of warmup iterations.
        n_iter: Number of timed iterations.

    Returns:
        Dict with mean_ms, p50_ms, p95_ms, p99_ms, throughput_fps.
    """
    import numpy as _np  # local import to keep module top clean

    with _torch.no_grad():  # type: ignore[union-attr]
        for _ in range(warmup):
            _ = model(dummy)
        if hasattr(_torch, "cuda") and _torch.cuda.is_available():  # type: ignore[union-attr]
            _torch.cuda.synchronize()  # type: ignore[union-attr]

        latencies: list[float] = []
        for _ in range(n_iter):
            t0 = time.perf_counter()
            _ = model(dummy)
            if hasattr(_torch, "cuda") and _torch.cuda.is_available():  # type: ignore[union-attr]
                _torch.cuda.synchronize()  # type: ignore[union-attr]
            latencies.append((time.perf_counter() - t0) * 1000)

    arr = _np.array(latencies)
    return {
        "mean_ms": round(float(arr.mean()), 2),
        "p50_ms": round(float(_np.percentile(arr, 50)), 2),
        "p95_ms": round(float(_np.percentile(arr, 95)), 2),
        "p99_ms": round(float(_np.percentile(arr, 99)), 2),
        "throughput_fps": round(1000.0 / float(arr.mean()), 1) if arr.mean() > 0 else 0.0,
    }


def _benchmark_trt_engine(
    engine_path: Path,
    input_shape: tuple[int, int, int, int],
    warmup: int,
    n_iter: int,
) -> dict[str, float]:
    """Benchmark a serialized TensorRT engine.

    Args:
        engine_path: Path to .engine file.
        input_shape: Input tensor shape.
        warmup:      Warmup iterations.
        n_iter:      Timed iterations.

    Returns:
        Latency stats dict (same format as _time_inference).
    """
    import numpy as _np

    trt_logger = _trt.Logger(_trt.Logger.WARNING)  # type: ignore[union-attr]
    runtime = _trt.Runtime(trt_logger)  # type: ignore[union-attr]
    with engine_path.open("rb") as fh:
        engine = runtime.deserialize_cuda_engine(fh.read())
    context = engine.create_execution_context()

    dummy = _torch.randn(input_shape, dtype=_torch.float16).cuda()  # type: ignore[union-attr]

    bindings = [dummy.data_ptr()]
    out_shape = context.get_binding_shape(1)
    output = _torch.zeros(tuple(out_shape), dtype=_torch.float16).cuda()  # type: ignore[union-attr]
    bindings.append(output.data_ptr())

    latencies: list[float] = []
    for i in range(warmup + n_iter):
        _torch.cuda.synchronize()  # type: ignore[union-attr]
        t0 = time.perf_counter()
        context.execute_v2(bindings)
        _torch.cuda.synchronize()  # type: ignore[union-attr]
        if i >= warmup:
            latencies.append((time.perf_counter() - t0) * 1000)

    arr = _np.array(latencies)
    return {
        "mean_ms": round(float(arr.mean()), 2),
        "p50_ms": round(float(_np.percentile(arr, 50)), 2),
        "p95_ms": round(float(_np.percentile(arr, 95)), 2),
        "p99_ms": round(float(_np.percentile(arr, 99)), 2),
        "throughput_fps": round(1000.0 / float(arr.mean()), 1) if arr.mean() > 0 else 0.0,
    }


def _save_fallback_info(output_dir: Path, info: dict) -> None:
    """Write or update fallback_info.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "fallback_info.json"
    existing = {}
    if path.exists():
        with contextlib.suppress(json.JSONDecodeError, OSError):
            existing = json.loads(path.read_text(encoding="utf-8"))
    existing.update(info)
    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def _format_optimization_report(benchmark: dict, fallback_info: dict) -> str:
    """Format a human-readable ViT optimization report."""
    onnx_method = fallback_info.get("onnx_method", "unknown")
    trt_method = fallback_info.get("trt_method", "unknown")

    lines = [
        "=" * 62,
        "  DriveSense-VLM — ViT Optimization Report",
        "=" * 62,
        "",
        f"  Input shape       : {benchmark.get('input_shape', 'unknown')}",
        f"  Iterations        : {benchmark.get('num_iterations', 0)}",
        f"  ONNX export path  : {onnx_method}",
        f"  Acceleration path : {trt_method}",
        "",
        "  Latency Results",
        "  " + "-" * 40,
    ]

    for backend_key, label in [
        ("pytorch_eager", "PyTorch Eager (baseline)"),
        ("torch_compile", "torch.compile"),
        ("tensorrt", "TensorRT fp16"),
    ]:
        s = benchmark.get(backend_key)
        if s:
            lines.append(
                f"  {label:<30} mean={s['mean_ms']:.1f} ms  "
                f"p95={s['p95_ms']:.1f} ms  fps={s['throughput_fps']:.0f}"
            )
        else:
            lines.append(f"  {label:<30} not available")

    for spd_key, label in [
        ("speedup_compile_vs_eager", "torch.compile speedup"),
        ("speedup_tensorrt_vs_eager", "TensorRT speedup"),
    ]:
        v = benchmark.get(spd_key)
        if v is not None:
            lines.append(f"  {label:<30} {v:.2f}x")

    if trt_method == "torch_compile":
        lines += [
            "",
            "  NOTE: TensorRT compilation fell back to torch.compile.",
            "  This is expected when custom attention operators cannot",
            "  be exported to ONNX. torch.compile typically gives",
            "  1.5–2.5x speedup vs eager on A100/H100.",
        ]
    lines += ["", "=" * 62]
    return "\n".join(lines)
