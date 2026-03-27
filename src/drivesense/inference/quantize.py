"""Stage 2: AWQ 4-bit quantization of the LLM decoder.

Quantizes the language model portion of Qwen3-VL while keeping
the vision encoder in full precision (fp16/bf16).

AWQ (Activation-aware Weight Quantization) preserves "salient" weight
channels for better accuracy than naive quantization.

Implemented in Phase 3a.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HPC-only imports — guarded for local macOS dev
# ---------------------------------------------------------------------------

try:
    from awq import AutoAWQForCausalLM as _AutoAWQ  # type: ignore[import]
    _AWQ_AVAILABLE = True
except ImportError:
    _AutoAWQ = None  # type: ignore[assignment]
    _AWQ_AVAILABLE = False

try:
    from transformers import AutoProcessor as _AutoProcessor  # type: ignore[import]
    from transformers import AutoTokenizer as _AutoTokenizer  # type: ignore[import]
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _AutoProcessor = None  # type: ignore[assignment]
    _AutoTokenizer = None  # type: ignore[assignment]
    _TRANSFORMERS_AVAILABLE = False

try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except ImportError:
    _torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

# Vision-encoder module name prefixes to exclude from quantization
_VIT_MODULE_PREFIXES: tuple[str, ...] = (
    "visual",
    "vision_model",
    "vit",
    "image_encoder",
    "patch_embed",
)


# ---------------------------------------------------------------------------
# AWQQuantizer
# ---------------------------------------------------------------------------


class AWQQuantizer:
    """AWQ 4-bit quantization pipeline for Qwen3-VL.

    Key design: Quantizes ONLY the LLM decoder, preserving the vision
    encoder in full precision. This is critical because quantizing the
    ViT degrades image understanding disproportionately.

    Args:
        config: Merged config dict (inference + model sections required).
    """

    def __init__(self, config: dict) -> None:
        quant_cfg = config.get("quantization", {})
        self._output_dir = Path(
            quant_cfg.get("output_dir", "outputs/quantized_model")
        )
        self._bits: int = int(quant_cfg.get("bits", 4))
        self._group_size: int = int(quant_cfg.get("group_size", 128))
        self._zero_point: bool = bool(quant_cfg.get("zero_point", True))
        self._calibration_samples: int = int(
            quant_cfg.get("calibration_samples", 128)
        )
        self._cfg = config

    # ── public API ──────────────────────────────────────────────────────────

    def prepare_calibration_data(
        self,
        dataset_path: Path,
        num_samples: int | None = None,
    ) -> list[str]:
        """Load calibration text strings from the SFT training JSONL.

        AWQ needs representative inputs to identify salient weight channels.
        Extracts text content (system + user messages) from SFT records.

        Args:
            dataset_path: Path to SFT JSONL file (sft_train.jsonl).
            num_samples:  Number of samples to load (default from config).

        Returns:
            List of formatted calibration text strings.
        """
        n = num_samples if num_samples is not None else self._calibration_samples
        path = Path(dataset_path)
        if not path.exists():
            logger.warning("Calibration dataset not found: %s", path)
            return _fallback_calibration_texts(n)

        texts: list[str] = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    text = _extract_text_from_record(rec)
                    if text:
                        texts.append(text)
                except (json.JSONDecodeError, KeyError):
                    continue
                if len(texts) >= n:
                    break

        # Pad with fallback if not enough records
        if len(texts) < n:
            texts.extend(_fallback_calibration_texts(n - len(texts)))

        logger.info("Prepared %d calibration samples from %s", len(texts), path)
        return texts[:n]

    def quantize(
        self,
        merged_model_dir: Path,
        output_dir: Path | None = None,
        calibration_data: list[str] | None = None,
    ) -> Path:
        """Quantize the merged model with AWQ (LLM decoder only).

        Args:
            merged_model_dir: Path to merged full-precision model (Phase 3a).
            output_dir:       Where to save quantized model (default from config).
            calibration_data: Pre-loaded calibration texts (loads from SFT if None).

        Returns:
            Path to quantized model directory.
        """
        _require_awq()
        out = Path(output_dir) if output_dir else self._output_dir
        out.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        logger.info("Loading model for AWQ quantization: %s", merged_model_dir)

        model = _AutoAWQ.from_pretrained(  # type: ignore[union-attr]
            str(merged_model_dir),
            safetensors=True,
            device_map="auto",
        )
        tokenizer = _AutoTokenizer.from_pretrained(str(merged_model_dir))  # type: ignore[union-attr]

        if calibration_data is None:
            sft_path = Path(
                self._cfg.get("annotation", {}).get(
                    "sft_output_dir", "outputs/data/sft_ready"
                )
            ) / "sft_train.jsonl"
            calibration_data = self.prepare_calibration_data(sft_path)

        modules_to_skip = _discover_vision_modules(model)
        logger.info("Excluding %d ViT modules from quantization", len(modules_to_skip))

        quant_config = {
            "zero_point": self._zero_point,
            "q_group_size": self._group_size,
            "w_bit": self._bits,
            "version": "GEMM",
            "modules_to_not_convert": modules_to_skip,
        }
        logger.info(
            "AWQ config: %d-bit, group_size=%d, zero_point=%s",
            self._bits,
            self._group_size,
            self._zero_point,
        )

        model.quantize(  # type: ignore[union-attr]
            tokenizer=tokenizer,
            quant_config=quant_config,
            calib_data=calibration_data,
        )

        logger.info("Saving quantized model to: %s", out)
        model.save_quantized(str(out))  # type: ignore[union-attr]
        tokenizer.save_pretrained(str(out))

        # Copy processor if present
        _copy_processor_files(merged_model_dir, out)

        elapsed = time.perf_counter() - t0
        stats = self.get_quantization_stats(out)
        logger.info(
            "Quantization complete in %.1fs — %.2f GB (%.1fx compression)",
            elapsed,
            stats["quantized_size_gb"],
            stats["compression_ratio"],
        )
        return out

    def benchmark_quality(
        self,
        merged_model_dir: Path,
        quantized_model_dir: Path,
        test_samples: list[dict],
    ) -> dict:
        """Compare quantized vs full-precision model on test samples.

        Args:
            merged_model_dir:    Path to full-precision merged model.
            quantized_model_dir: Path to AWQ quantized model.
            test_samples:        List of GT annotation dicts to test on.

        Returns:
            Dict with text_similarity, bbox_mae, label_agreement,
            size_reduction, original_size_gb, quantized_size_gb.
        """
        _require_torch()
        orig_stats = self.get_quantization_stats(merged_model_dir)
        quant_stats = self.get_quantization_stats(quantized_model_dir)

        orig_gb = orig_stats["quantized_size_gb"]  # reuses same method
        quant_gb = quant_stats["quantized_size_gb"]
        size_reduction = orig_gb / quant_gb if quant_gb > 0 else 0.0

        if not test_samples:
            return _empty_quality_metrics(orig_gb, quant_gb, size_reduction)

        try:
            orig_outputs = _run_text_inference(merged_model_dir, test_samples)
            quant_outputs = _run_text_inference(str(quantized_model_dir), test_samples)
            text_sim = _compute_text_similarity(orig_outputs, quant_outputs)
            bbox_mae = _compute_bbox_mae(orig_outputs, quant_outputs)
            label_agr = _compute_label_agreement(orig_outputs, quant_outputs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Quality benchmark inference failed: %s", exc)
            return _empty_quality_metrics(orig_gb, quant_gb, size_reduction)

        return {
            "text_similarity": round(text_sim, 4),
            "bbox_mae": round(bbox_mae, 2),
            "label_agreement": round(label_agr, 4),
            "size_reduction": round(size_reduction, 2),
            "original_size_gb": round(orig_gb, 3),
            "quantized_size_gb": round(quant_gb, 3),
        }

    def get_quantization_stats(self, quantized_dir: Path) -> dict:
        """Return model size, compression ratio, and layer counts.

        Args:
            quantized_dir: Path to (quantized or full) model directory.

        Returns:
            Dict with model_size_bytes, quantized_size_gb, compression_ratio,
            quantized_layers, total_weight_files.
        """
        out = Path(quantized_dir)
        st_files = list(out.glob("*.safetensors"))
        bin_files = list(out.glob("*.bin"))
        weight_files = st_files or bin_files
        total_bytes = sum(f.stat().st_size for f in weight_files)
        size_gb = total_bytes / (1024 ** 3)

        # Detect quantized layers from quant_config.json (AutoAWQ saves this)
        quant_layers = 0
        qcfg_path = out / "quant_config.json"
        if qcfg_path.exists():
            with qcfg_path.open(encoding="utf-8") as fh:
                qcfg = json.load(fh)
            quant_layers = qcfg.get("num_quantized_layers", 0)

        # Rough compression estimate vs bfloat16
        fp16_size = size_gb * (16 / self._bits)  # approximate original size
        compression = fp16_size / size_gb if size_gb > 0 else 1.0

        return {
            "model_size_bytes": total_bytes,
            "quantized_size_gb": round(size_gb, 3),
            "compression_ratio": round(compression, 2),
            "quantized_layers": quant_layers,
            "total_weight_files": len(weight_files),
        }


# ---------------------------------------------------------------------------
# Module-level convenience functions (kept from original stub)
# ---------------------------------------------------------------------------


def quantize_model(
    merged_model_dir: Path,
    output_dir: Path,
    config: dict,
    calibration_data: list[str] | None = None,
) -> Path:
    """Convenience wrapper for AWQ quantization.

    Args:
        merged_model_dir: Path to merged model (Phase 3a output).
        output_dir:       Destination for quantized model.
        config:           Inference config dict.
        calibration_data: Optional list of calibration text strings.

    Returns:
        Path to quantized model directory.
    """
    quantizer = AWQQuantizer(config)
    return quantizer.quantize(merged_model_dir, output_dir, calibration_data)


def load_calibration_data(data_dir: Path, n_samples: int = 128) -> list[str]:
    """Load calibration samples from training dataset for AWQ.

    Args:
        data_dir:  Path to DriveSense SFT data directory.
        n_samples: Number of samples for AWQ weight search.

    Returns:
        List of formatted prompt strings.
    """
    sft_path = Path(data_dir) / "sft_train.jsonl"
    quantizer = AWQQuantizer({"quantization": {"calibration_samples": n_samples}})
    return quantizer.prepare_calibration_data(sft_path, n_samples)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_awq() -> None:
    """Raise ImportError if AWQ/Transformers dependencies are missing."""
    missing = []
    if not _AWQ_AVAILABLE:
        missing.append("autoawq")
    if not _TRANSFORMERS_AVAILABLE:
        missing.append("transformers")
    if missing:
        raise ImportError(
            f"HPC dependencies not available: {', '.join(missing)}. "
            "Install on HPC: pip install autoawq transformers"
        )


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError("torch not available. Install on HPC.")


def _discover_vision_modules(model: object) -> list[str]:
    """Inspect model.named_modules() to find ViT modules to exclude.

    Args:
        model: Loaded AutoAWQ model.

    Returns:
        List of top-level module name strings to skip during quantization.
    """
    excluded: list[str] = []
    try:
        for name, _ in model.named_modules():  # type: ignore[union-attr]
            parts = name.split(".")
            top = parts[0] if parts else name
            if any(pfx in top.lower() for pfx in _VIT_MODULE_PREFIXES) and top not in excluded:
                excluded.append(top)
                logger.debug("Excluding ViT module from quantization: %s", top)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Module discovery failed: %s — using defaults", exc)
        excluded = ["visual"]
    if not excluded:
        excluded = ["visual"]  # Qwen3-VL default
    return excluded


def _extract_text_from_record(rec: dict) -> str:
    """Extract calibration text from an SFT JSONL record.

    Args:
        rec: SFT JSONL record with ``messages`` key.

    Returns:
        Concatenated text from system + user messages.
    """
    parts: list[str] = []
    for msg in rec.get("messages", []):
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "assistant":
            continue
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
    return " ".join(p for p in parts if p).strip()


def _fallback_calibration_texts(n: int) -> list[str]:
    """Generate minimal fallback calibration strings when dataset unavailable."""
    return [
        "Analyse this dashcam image for road hazards and describe what you see."
    ] * n


def _copy_processor_files(src: Path, dst: Path) -> None:
    """Copy processor/tokenizer files from src → dst if available."""
    if not _TRANSFORMERS_AVAILABLE or _AutoProcessor is None:
        return
    try:
        proc = _AutoProcessor.from_pretrained(str(src))  # type: ignore[union-attr]
        proc.save_pretrained(str(dst))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Processor copy failed: %s", exc)


def _run_text_inference(model_dir: str | Path, samples: list[dict]) -> list[str]:
    """Run text generation on samples with a loaded model.

    Args:
        model_dir: Path to model directory.
        samples:   List of annotation dicts to generate predictions for.

    Returns:
        List of raw output strings.
    """
    from drivesense.inference.merge_lora import (  # noqa: PLC0415, I001
        _TORCH_AVAILABLE as _T,
        _VLMClass,
    )
    if not _T or _VLMClass is None:
        return [""] * len(samples)

    model = _VLMClass.from_pretrained(  # type: ignore[union-attr]
        str(model_dir), device_map="auto", torch_dtype=_torch.float16  # type: ignore[union-attr]
    )
    proc = _AutoProcessor.from_pretrained(str(model_dir))  # type: ignore[union-attr]
    device = next(model.parameters()).device
    outputs: list[str] = []

    for sample in samples:
        text = sample.get("scene_summary", "Analyse this dashcam frame.")
        inputs = proc(text=[text], return_tensors="pt")
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with _torch.no_grad():  # type: ignore[union-attr]
            ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        outputs.append(
            proc.batch_decode(ids[:, inputs["input_ids"].shape[1]:],
                              skip_special_tokens=True)[0]
        )
    return outputs


def _compute_text_similarity(a: list[str], b: list[str]) -> float:
    """Compute mean character-level overlap between two output lists."""
    if not a or not b:
        return 0.0
    scores: list[float] = []
    for x, y in zip(a, b):  # noqa: B905
        if not x and not y:
            scores.append(1.0)
            continue
        longer = max(len(x), len(y))
        if longer == 0:
            scores.append(1.0)
            continue
        overlap = sum(cx == cy for cx, cy in zip(x, y))  # noqa: B905
        scores.append(overlap / longer)
    return sum(scores) / len(scores) if scores else 0.0


def _compute_bbox_mae(a: list[str], b: list[str]) -> float:
    """Compute mean absolute error on bbox coordinates from JSON outputs."""
    import re
    errors: list[float] = []
    pattern = re.compile(r'"bbox_2d"\s*:\s*\[([^\]]+)\]')
    for x, y in zip(a, b):  # noqa: B905
        bboxes_a = pattern.findall(x)
        bboxes_b = pattern.findall(y)
        for ba, bb in zip(bboxes_a, bboxes_b):  # noqa: B905
            try:
                coords_a = [float(c) for c in ba.split(",")]
                coords_b = [float(c) for c in bb.split(",")]
                errors.extend(abs(ca - cb) for ca, cb in zip(coords_a, coords_b))  # noqa: B905
            except ValueError:
                continue
    return sum(errors) / len(errors) if errors else 0.0


def _compute_label_agreement(a: list[str], b: list[str]) -> float:
    """Compute fraction of samples with identical hazard labels."""
    import re
    matches = 0
    pattern = re.compile(r'"label"\s*:\s*"([^"]+)"')
    for x, y in zip(a, b):  # noqa: B905
        labels_a = set(pattern.findall(x))
        labels_b = set(pattern.findall(y))
        if labels_a == labels_b:
            matches += 1
    return matches / len(a) if a else 0.0


def _empty_quality_metrics(orig_gb: float, quant_gb: float, size_reduction: float) -> dict:
    return {
        "text_similarity": 0.0,
        "bbox_mae": 0.0,
        "label_agreement": 0.0,
        "size_reduction": round(size_reduction, 2),
        "original_size_gb": round(orig_gb, 3),
        "quantized_size_gb": round(quant_gb, 3),
    }
