"""Production serving for DriveSense-VLM.

Two backends:
- ``DriveSenseVLLMServer`` — vLLM batch inference for HPC throughput benchmarking.
- ``DriveSenseLocalInference`` — transformers generate() for HF Spaces / CPU dev.

Both expose ``predict(image)`` → structured hazard dict and
``predict_batch(images)`` → list of dicts.

``draw_hazard_boxes(image, annotation)`` overlays coloured bounding boxes on a
PIL Image using severity-coded RGBA fills.

Implemented in Phase 3c.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

DRIVESENSE_SYSTEM_PROMPT = (
    "You are DriveSense-VLM, a specialized vision-language model for autonomous vehicle "
    "safety. Analyse the dashcam image and detect rare driving hazards. "
    'Output ONLY valid JSON with this exact schema:\n'
    '{"hazards": [{"label": str, "bbox_2d": [x1,y1,x2,y2], "severity": str, '
    '"reasoning": str, "action": str}], "scene_summary": str, '
    '"ego_context": {"weather": str, "time_of_day": str, "road_type": str}}\n'
    "bbox_2d coordinates are integers in [0, 1000]. "
    "severity is one of: low, medium, high, critical."
)

SEVERITY_COLORS: dict[str, tuple[int, int, int]] = {
    "critical": (255, 0, 0),
    "high": (255, 140, 0),
    "medium": (255, 215, 0),
    "low": (50, 205, 50),
    "no_hazard": (65, 105, 225),
}

# ---------------------------------------------------------------------------
# HPC-only imports — guarded for local macOS dev
# ---------------------------------------------------------------------------

try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except ImportError:
    _torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

try:
    from vllm import LLM as _LLM  # type: ignore[import]
    from vllm import SamplingParams as _SamplingParams  # type: ignore[import]  # noqa: I001
    _VLLM_AVAILABLE = True
except ImportError:
    _LLM = None  # type: ignore[assignment]
    _SamplingParams = None  # type: ignore[assignment]
    _VLLM_AVAILABLE = False

try:
    from transformers import AutoProcessor as _AutoProcessor  # type: ignore[import]
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _AutoProcessor = None  # type: ignore[assignment]
    _TRANSFORMERS_AVAILABLE = False

try:
    from transformers import Qwen2_5_VLForConditionalGeneration as _VLMClass  # type: ignore[import]
    _VLM_CLASS_AVAILABLE = True
except ImportError:
    try:
        from transformers import Qwen2VLForConditionalGeneration as _VLMClass  # type: ignore[import]  # noqa: I001
        _VLM_CLASS_AVAILABLE = True
    except ImportError:
        _VLMClass = None  # type: ignore[assignment]
        _VLM_CLASS_AVAILABLE = False

try:
    from PIL import Image as _PILImage  # type: ignore[import]
    from PIL import ImageDraw as _ImageDraw  # type: ignore[import]  # noqa: I001
    from PIL import ImageFont as _ImageFont  # type: ignore[import]
    _PIL_AVAILABLE = True
except ImportError:
    _PILImage = None  # type: ignore[assignment]
    _ImageDraw = None  # type: ignore[assignment]
    _ImageFont = None  # type: ignore[assignment]
    _PIL_AVAILABLE = False


# ---------------------------------------------------------------------------
# DriveSenseVLLMServer
# ---------------------------------------------------------------------------


class DriveSenseVLLMServer:
    """vLLM-backed server for HPC throughput benchmarking.

    Loads the AWQ-quantized model into vLLM and exposes synchronous
    ``predict()`` / ``predict_batch()`` / ``benchmark()`` APIs.

    Args:
        config: Merged config dict; reads ``config["vllm"]`` section.
    """

    def __init__(self, config: dict) -> None:
        _require_vllm()
        vllm_cfg = config.get("vllm", {})
        self._model_path: str = vllm_cfg.get("model_path", "outputs/quantized_model")
        self._tensor_parallel_size: int = int(vllm_cfg.get("tensor_parallel_size", 1))
        self._gpu_memory_utilization: float = float(
            vllm_cfg.get("gpu_memory_utilization", 0.85)
        )
        self._max_model_len: int = int(vllm_cfg.get("max_model_len", 2048))
        self._engine: object = None
        self._cold_start_ms: float = 0.0
        self._load_engine()

    # ── public API ──────────────────────────────────────────────────────────

    def predict(self, image: PILImage) -> dict:
        """Run hazard detection on a single dashcam frame.

        Args:
            image: PIL Image — resized to 672×448 before encoding.

        Returns:
            Parsed hazard annotation dict.
        """
        results = self.predict_batch([image])
        return results[0]

    def predict_batch(self, images: list[PILImage]) -> list[dict]:
        """Run hazard detection on a batch of dashcam frames.

        Args:
            images: List of PIL Images.

        Returns:
            List of parsed hazard annotation dicts (same order as input).
        """
        _require_vllm()
        prompts = [_format_vllm_prompt(img) for img in images]
        sampling = _SamplingParams(  # type: ignore[call-arg]
            temperature=0,
            max_tokens=512,
            stop=["<|im_end|>"],
        )
        outputs = self._engine.generate(prompts, sampling)  # type: ignore[union-attr]
        return [_parse_json_output(o.outputs[0].text) for o in outputs]

    def benchmark(
        self,
        images: list[PILImage],
        concurrency: int = 8,
        num_iterations: int = 100,
    ) -> dict:
        """Measure throughput and latency under concurrent load.

        Args:
            images:        Pool of images to sample from.
            concurrency:   Number of parallel workers.
            num_iterations: Total requests to issue.

        Returns:
            Benchmark result dict with ``mean_ms``, ``p50_ms``, ``p95_ms``,
            ``p99_ms``, ``throughput_rps``, and GPU memory stats.
        """
        latencies = _benchmark_concurrent(
            fn=self.predict,
            images=images,
            concurrency=concurrency,
            num_iterations=num_iterations,
        )
        stats = _latency_stats(latencies)
        stats["throughput_rps"] = round(
            num_iterations / (sum(latencies) / 1000), 2
        )
        stats["gpu_memory"] = _get_gpu_memory_stats()
        stats["model_info"] = _get_model_info(self._model_path)
        stats["cold_start_ms"] = round(self._cold_start_ms, 1)
        return stats

    def shutdown(self) -> None:
        """Release GPU resources held by the vLLM engine."""
        with contextlib.suppress(Exception):
            del self._engine
            self._engine = None
            if _TORCH_AVAILABLE and _torch is not None:
                _torch.cuda.empty_cache()
        logger.info("vLLM engine shut down")

    # ── private ─────────────────────────────────────────────────────────────

    def _load_engine(self) -> None:
        t0 = time.perf_counter()
        logger.info("Loading vLLM engine from: %s", self._model_path)
        self._engine = _LLM(  # type: ignore[call-arg]
            model=self._model_path,
            quantization="awq",
            tensor_parallel_size=self._tensor_parallel_size,
            gpu_memory_utilization=self._gpu_memory_utilization,
            max_model_len=self._max_model_len,
            trust_remote_code=True,
        )
        self._cold_start_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "vLLM engine loaded in %.1f ms — model %s",
            self._cold_start_ms,
            self._model_path,
        )


# ---------------------------------------------------------------------------
# DriveSenseLocalInference
# ---------------------------------------------------------------------------


class DriveSenseLocalInference:
    """Transformers-backed local inference for HF Spaces / CPU dev.

    Lazy-loads the model on first predict call. Supports both AWQ-quantized
    and full-precision models.

    Args:
        config: Merged config dict; reads ``config["demo"]`` section.
    """

    def __init__(self, config: dict) -> None:
        demo_cfg = config.get("demo", {})
        self._model_path: str = demo_cfg.get("model_path", "outputs/quantized_model")
        self._device: str = demo_cfg.get("device", "auto")
        max_size = demo_cfg.get("max_image_size", [672, 448])
        self._max_image_size: tuple[int, int] = (int(max_size[0]), int(max_size[1]))
        self._model: object = None
        self._processor: object = None

    # ── public API ──────────────────────────────────────────────────────────

    def predict(self, image: PILImage) -> dict:
        """Run hazard detection on a single dashcam frame.

        Args:
            image: PIL Image — resized before inference.

        Returns:
            Parsed hazard annotation dict.
        """
        self._load()
        img = _resize_image(image, self._max_image_size)
        messages = _format_local_messages(img)
        return self._run_inference(messages, img)

    def predict_with_visualization(
        self, image: PILImage
    ) -> tuple[PILImage, dict]:
        """Predict and draw hazard boxes on the image.

        Args:
            image: Input PIL Image.

        Returns:
            Tuple of (annotated PIL Image, annotation dict).
        """
        annotation = self.predict(image)
        annotated = draw_hazard_boxes(image, annotation)
        return annotated, annotation

    # ── private ─────────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Lazy-load model and processor on first call."""
        if self._model is not None:
            return
        if not _TRANSFORMERS_AVAILABLE or not _VLM_CLASS_AVAILABLE:
            raise ImportError(
                "transformers not available. "
                "Install: pip install transformers accelerate"
            )
        dtype = _resolve_dtype(self._device)
        logger.info("Loading local model from: %s", self._model_path)
        self._model = _VLMClass.from_pretrained(  # type: ignore[union-attr]
            self._model_path,
            torch_dtype=dtype,
            device_map=self._device,
        )
        self._processor = _AutoProcessor.from_pretrained(self._model_path)  # type: ignore[union-attr]
        logger.info("Local model loaded — %s", self._model_path)

    def _run_inference(self, messages: list[dict], image: PILImage) -> dict:
        """Run the model forward pass and parse JSON output."""
        text = self._processor.apply_chat_template(  # type: ignore[union-attr]
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._processor(  # type: ignore[union-attr]
            text=[text],
            images=[image],
            return_tensors="pt",
        )
        device = next(self._model.parameters()).device  # type: ignore[union-attr]
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with _torch.no_grad():  # type: ignore[union-attr]
            out_ids = self._model.generate(  # type: ignore[union-attr]
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )
        input_len = inputs["input_ids"].shape[1]
        raw = self._processor.batch_decode(  # type: ignore[union-attr]
            out_ids[:, input_len:],
            skip_special_tokens=True,
        )[0]
        return _parse_json_output(raw)


# ---------------------------------------------------------------------------
# draw_hazard_boxes
# ---------------------------------------------------------------------------


def draw_hazard_boxes(
    image: PILImage,
    annotation: dict,
    image_size: tuple[int, int] | None = None,
) -> PILImage:
    """Overlay severity-coded bounding boxes on a PIL Image.

    Coordinates in ``annotation`` use [0, 1000] normalised space.
    Each hazard gets a semi-transparent RGBA fill and a solid outline with
    a label showing ``"{label} ({severity})"``.

    Args:
        image:      Input PIL Image (RGB or RGBA).
        annotation: Parsed annotation dict with ``hazards`` list.
        image_size: Override (width, height); defaults to ``image.size``.

    Returns:
        New PIL Image with bounding boxes drawn.
    """
    if not _PIL_AVAILABLE:
        logger.warning("PIL not available — returning original image")
        return image

    w, h = image_size if image_size else image.size
    hazards = annotation.get("hazards", [])

    base = image.convert("RGBA")
    overlay = _PILImage.new("RGBA", base.size, (0, 0, 0, 0))  # type: ignore[union-attr]
    draw = _ImageDraw.Draw(overlay)  # type: ignore[union-attr]

    for hazard in hazards:
        bbox = hazard.get("bbox_2d", [])
        if len(bbox) != 4:
            continue
        severity = str(hazard.get("severity", "no_hazard")).lower()
        label = str(hazard.get("label", "hazard"))
        color = SEVERITY_COLORS.get(severity, SEVERITY_COLORS["no_hazard"])

        x1 = int(bbox[0] * w / 1000)
        y1 = int(bbox[1] * h / 1000)
        x2 = int(bbox[2] * w / 1000)
        y2 = int(bbox[3] * h / 1000)

        # Semi-transparent fill (alpha ~20%)
        fill_color = (*color, 50)
        draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        # Solid outline
        outline_color = (*color, 255)
        draw.rectangle([x1, y1, x2, y2], outline=outline_color, width=2)

        # Label above box
        text = f"{label} ({severity})"
        font = _get_font()
        text_y = max(0, y1 - 18)
        draw.text((x1 + 2, text_y), text, fill=outline_color, font=font)

    composited = _PILImage.alpha_composite(base, overlay)  # type: ignore[union-attr]
    return composited.convert("RGB")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_vllm() -> None:
    """Raise ImportError if vLLM is not installed."""
    if not _VLLM_AVAILABLE:
        raise ImportError(
            "vLLM not available. Install on HPC: pip install vllm"
        )


def _resize_image(image: PILImage, max_size: tuple[int, int]) -> PILImage:
    """Resize image to max_size (width, height) preserving aspect ratio."""
    if not _PIL_AVAILABLE:
        return image
    target_w, target_h = max_size
    img_w, img_h = image.size
    if img_w == target_w and img_h == target_h:
        return image
    ratio = min(target_w / img_w, target_h / img_h)
    new_w = int(img_w * ratio)
    new_h = int(img_h * ratio)
    return image.resize((new_w, new_h), _PILImage.LANCZOS)  # type: ignore[attr-defined]


def _format_vllm_prompt(image: PILImage) -> dict:
    """Build a vLLM prompt dict for a single image."""
    return {
        "prompt": (
            f"<|im_start|>system\n{DRIVESENSE_SYSTEM_PROMPT}<|im_end|>\n"
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            "Analyse this dashcam frame for hazards.<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        "multi_modal_data": {"image": image},
    }


def _format_local_messages(image: PILImage) -> list[dict]:
    """Build transformers chat messages list for a single image."""
    return [
        {"role": "system", "content": DRIVESENSE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Analyse this dashcam frame for hazards."},
            ],
        },
    ]


def _parse_json_output(raw_text: str) -> dict:
    """Extract and parse the JSON annotation from raw model output.

    Tries direct parse first, then extracts from markdown code fences,
    then falls back to regex extraction of the outermost ``{...}`` block.

    Args:
        raw_text: Raw model generation string.

    Returns:
        Parsed annotation dict. On failure, returns ``{"parse_failure": raw_text}``.
    """
    text = raw_text.strip()

    # Direct parse
    with contextlib.suppress(json.JSONDecodeError):
        return json.loads(text)

    # Strip markdown fences
    fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if fence_match:
        with contextlib.suppress(json.JSONDecodeError):
            return json.loads(fence_match.group(1))

    # Extract outermost {...}
    brace_match = re.search(r"\{[\s\S]*\}", text)
    if brace_match:
        with contextlib.suppress(json.JSONDecodeError):
            return json.loads(brace_match.group(0))

    logger.warning("Failed to parse model output as JSON: %s", text[:200])
    return {"parse_failure": raw_text, "hazards": []}


def _resolve_dtype(device: str) -> object:
    """Return appropriate torch dtype for the target device."""
    if not _TORCH_AVAILABLE or _torch is None:
        return None
    if device == "cpu":
        return _torch.float32
    return _torch.bfloat16


def _latency_stats(latencies_ms: list[float]) -> dict:
    """Compute mean/p50/p95/p99 from a list of millisecond latencies."""
    if not latencies_ms:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
    try:
        import numpy as np  # type: ignore[import]
        arr = np.array(latencies_ms, dtype=float)
        return {
            "mean_ms": round(float(arr.mean()), 2),
            "p50_ms": round(float(np.percentile(arr, 50)), 2),
            "p95_ms": round(float(np.percentile(arr, 95)), 2),
            "p99_ms": round(float(np.percentile(arr, 99)), 2),
        }
    except ImportError:
        sorted_lat = sorted(latencies_ms)
        n = len(sorted_lat)
        return {
            "mean_ms": round(sum(sorted_lat) / n, 2),
            "p50_ms": round(sorted_lat[int(n * 0.50)], 2),
            "p95_ms": round(sorted_lat[int(n * 0.95)], 2),
            "p99_ms": round(sorted_lat[min(int(n * 0.99), n - 1)], 2),
        }


def _benchmark_concurrent(
    fn: object,
    images: list[PILImage],
    concurrency: int,
    num_iterations: int,
) -> list[float]:
    """Issue ``num_iterations`` calls to ``fn`` with ``concurrency`` threads.

    Returns a list of per-call latencies in milliseconds.
    """
    import itertools

    latencies: list[float] = []
    image_cycle = itertools.cycle(images)

    def _single_call(img: PILImage) -> float:
        t0 = time.perf_counter()
        fn(img)  # type: ignore[operator]
        return (time.perf_counter() - t0) * 1000

    call_images = [next(image_cycle) for _ in range(num_iterations)]
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_single_call, img) for img in call_images]
        for fut in as_completed(futures):
            with contextlib.suppress(Exception):
                latencies.append(fut.result())
    return latencies


def _get_gpu_memory_stats() -> dict:
    """Return GPU memory usage dict (free/used/total in GB)."""
    if not _TORCH_AVAILABLE or _torch is None:
        return {}
    with contextlib.suppress(Exception):
        if _torch.cuda.is_available():
            free, total = _torch.cuda.mem_get_info()
            used = total - free
            return {
                "total_gb": round(total / 1e9, 2),
                "used_gb": round(used / 1e9, 2),
                "free_gb": round(free / 1e9, 2),
            }
    return {}


def _get_gpu_name() -> str:
    """Return the GPU device name, or 'CPU' if CUDA unavailable."""
    if not _TORCH_AVAILABLE or _torch is None:
        return "CPU"
    with contextlib.suppress(Exception):
        if _torch.cuda.is_available():
            return _torch.cuda.get_device_name(0)
    return "CPU"


def _get_model_info(model_path: str) -> dict:
    """Return basic model info from config.json (no model load)."""
    cfg_path = Path(model_path) / "config.json"
    if not cfg_path.exists():
        return {"model_path": model_path}
    with contextlib.suppress(Exception):
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        return {
            "model_path": model_path,
            "model_type": cfg.get("model_type", "unknown"),
            "quantization": cfg.get("quantization_config", {}).get("quant_type", "none"),
        }
    return {"model_path": model_path}


def _get_font() -> object:
    """Return a PIL ImageFont (default if truetype unavailable)."""
    if not _PIL_AVAILABLE or _ImageFont is None:
        return None
    with contextlib.suppress(Exception):
        return _ImageFont.load_default()
    return None
