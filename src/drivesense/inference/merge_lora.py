"""Stage 1: Merge LoRA adapters into base Qwen3-VL model.

Produces a full-weight model ready for quantization or direct inference.
The merge must happen BEFORE quantization — merging after quantization
produces incorrect results.

Implemented in Phase 3a.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

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
    from peft import PeftModel as _PeftModel  # type: ignore[import]
    _PEFT_AVAILABLE = True
except ImportError:
    _PeftModel = None  # type: ignore[assignment]
    _PEFT_AVAILABLE = False

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


# ---------------------------------------------------------------------------
# LoRAMerger
# ---------------------------------------------------------------------------


class LoRAMerger:
    """Merges trained LoRA adapters into the base Qwen3-VL model.

    Pipeline:
    1. Load base model in bfloat16
    2. Load LoRA adapter weights via PEFT
    3. Merge with ``merge_and_unload()``
    4. Save full-weight model as .safetensors
    5. Save processor alongside
    6. Verify merge by comparing logits

    Args:
        config: Merged config dict (inference + model sections required).
    """

    def __init__(self, config: dict) -> None:
        merge_cfg = config.get("merge", {})
        model_cfg = config.get("model", {})
        self._output_dir = Path(merge_cfg.get("output_dir", "outputs/merged_model"))
        self._safe_serialization: bool = bool(merge_cfg.get("safe_serialization", True))
        self._model_name: str = model_cfg.get("name", "Qwen/Qwen3-VL-2B-Instruct")
        self._revision: str = model_cfg.get("revision", "main")
        self._torch_dtype: str = model_cfg.get("torch_dtype", "bfloat16")
        self._cfg = config

    # ── public API ──────────────────────────────────────────────────────────

    def merge(
        self,
        adapter_path: Path,
        output_dir: Path | None = None,
        verify: bool = True,
    ) -> Path:
        """Merge LoRA adapter into base model and save to disk.

        Args:
            adapter_path: Path to LoRA adapter directory (Phase 2a output).
            output_dir:   Where to save merged model (default from config).
            verify:       If True, compare logits pre/post merge.

        Returns:
            Path to merged model directory.
        """
        _require_hpc_deps()
        out = Path(output_dir) if output_dir else self._output_dir
        out.mkdir(parents=True, exist_ok=True)

        dtype = _parse_torch_dtype(self._torch_dtype)
        t0 = time.perf_counter()

        logger.info("Loading base model: %s", self._model_name)
        base = _VLMClass.from_pretrained(  # type: ignore[union-attr]
            self._model_name,
            torch_dtype=dtype,
            device_map="auto",
            revision=self._revision,
        )

        logger.info("Loading LoRA adapter from: %s", adapter_path)
        peft_model = _PeftModel.from_pretrained(base, str(adapter_path))  # type: ignore[union-attr]

        # Capture logits before merge for verification
        sample_logits = None
        if verify:
            sample_logits = _sample_logits(peft_model)

        logger.info("Merging adapter weights (merge_and_unload)…")
        merged = peft_model.merge_and_unload()

        logger.info("Saving merged model to: %s", out)
        merged.save_pretrained(  # type: ignore[union-attr]
            str(out), safe_serialization=self._safe_serialization
        )
        _copy_processor(adapter_path, out)

        elapsed = time.perf_counter() - t0
        stats = self.get_merge_stats(out)
        logger.info(
            "Merge complete in %.1fs — %.2f GB, %d parameters",
            elapsed,
            stats["model_size_gb"],
            stats["total_parameters"],
        )

        if verify and sample_logits is not None:
            self.verify_merge(out, adapter_path)

        return out

    def verify_merge(self, merged_dir: Path, adapter_path: Path) -> bool:
        """Load merged model and compare logits against the adapter model.

        Args:
            merged_dir:   Path to merged model directory.
            adapter_path: Path to original LoRA adapter.

        Returns:
            True if logits are equivalent within floating-point tolerance.
        """
        _require_hpc_deps()
        logger.info("Verifying merge: comparing adapter vs merged logits…")

        dtype = _parse_torch_dtype(self._torch_dtype)
        base = _VLMClass.from_pretrained(  # type: ignore[union-attr]
            self._model_name,
            torch_dtype=dtype,
            device_map="auto",
            revision=self._revision,
        )
        adapter_model = _PeftModel.from_pretrained(base, str(adapter_path))  # type: ignore[union-attr]
        adapter_logits = _sample_logits(adapter_model)

        merged_model = _VLMClass.from_pretrained(  # type: ignore[union-attr]
            str(merged_dir),
            torch_dtype=dtype,
            device_map="auto",
        )
        merged_logits = _sample_logits(merged_model)

        ok = bool(
            _torch.allclose(  # type: ignore[union-attr]
                adapter_logits.float(), merged_logits.float(), atol=1e-3
            )
        )
        if ok:
            logger.info("Verification PASSED — logits match within tolerance")
        else:
            logger.warning("Verification FAILED — logits differ; check merge")
        return ok

    def get_merge_stats(self, merged_dir: Path) -> dict:
        """Return statistics about the merged model directory.

        Args:
            merged_dir: Path to merged model directory.

        Returns:
            Dict with ``total_parameters``, ``model_size_gb``,
            ``safetensors_files``, ``config_hash``.
        """
        out = Path(merged_dir)
        st_files = sorted(p.name for p in out.glob("*.safetensors"))
        bin_files = sorted(p.name for p in out.glob("*.bin"))
        all_weight_files = st_files or bin_files

        total_bytes = sum(
            (out / f).stat().st_size for f in all_weight_files if (out / f).exists()
        )
        size_gb = total_bytes / (1024 ** 3)

        config_hash = ""
        cfg_path = out / "config.json"
        if cfg_path.exists():
            config_hash = hashlib.md5(  # noqa: S324
                cfg_path.read_bytes()
            ).hexdigest()

        # Count parameters from config if model not loaded
        total_params = _count_params_from_config(out)

        return {
            "total_parameters": total_params,
            "model_size_gb": round(size_gb, 3),
            "safetensors_files": all_weight_files,
            "config_hash": config_hash,
        }


# ---------------------------------------------------------------------------
# Module-level convenience functions (kept from original stub)
# ---------------------------------------------------------------------------


def merge_lora_checkpoint(
    base_model_name: str,
    lora_checkpoint_dir: Path,
    output_dir: Path,
    safe_serialization: bool = True,
) -> Path:
    """Convenience wrapper — merge and save with minimal config.

    Args:
        base_model_name:    HuggingFace model ID.
        lora_checkpoint_dir: Path to LoRA adapter directory.
        output_dir:         Destination for merged model.
        safe_serialization: Save as .safetensors if True.

    Returns:
        Path to merged model directory.
    """
    config = {
        "model": {"name": base_model_name, "revision": "main", "torch_dtype": "bfloat16"},
        "merge": {"output_dir": str(output_dir), "safe_serialization": safe_serialization},
    }
    merger = LoRAMerger(config)
    return merger.merge(lora_checkpoint_dir, output_dir=output_dir, verify=False)


def verify_merge(merged_model_dir: Path, test_prompt: str = "Describe this scene.") -> bool:
    """Sanity-check the merged model with a short text generation.

    Args:
        merged_model_dir: Path to merged model directory.
        test_prompt:      Short text prompt.

    Returns:
        True if model produces a non-empty response.
    """
    if not _TORCH_AVAILABLE or not _VLM_CLASS_AVAILABLE:
        logger.warning("torch/transformers not available — skipping verify_merge")
        return False
    try:
        model = _VLMClass.from_pretrained(  # type: ignore[union-attr]
            str(merged_model_dir),
            torch_dtype=_torch.bfloat16,  # type: ignore[union-attr]
            device_map="auto",
        )
        processor = _AutoProcessor.from_pretrained(str(merged_model_dir))  # type: ignore[union-attr]
        inputs = processor(text=[test_prompt], return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with _torch.no_grad():  # type: ignore[union-attr]
            out_ids = model.generate(**inputs, max_new_tokens=16, do_sample=False)
        result = processor.batch_decode(out_ids[:, inputs["input_ids"].shape[1]:],
                                        skip_special_tokens=True)[0]
        ok = len(result.strip()) > 0
        logger.info("verify_merge: generated %d chars — %s", len(result), "OK" if ok else "EMPTY")
        return ok
    except Exception as exc:  # noqa: BLE001
        logger.warning("verify_merge failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_hpc_deps() -> None:
    """Raise ImportError if HPC dependencies are missing."""
    missing = []
    if not _TORCH_AVAILABLE:
        missing.append("torch")
    if not _PEFT_AVAILABLE:
        missing.append("peft")
    if not _VLM_CLASS_AVAILABLE:
        missing.append("transformers (Qwen2_5_VLForConditionalGeneration)")
    if missing:
        raise ImportError(
            f"HPC dependencies not available: {', '.join(missing)}. "
            "Install the full stack on HPC: pip install -e '.[training]'"
        )


def _parse_torch_dtype(dtype_str: str) -> object:
    """Convert string dtype name to torch dtype."""
    _map = {"bfloat16": "bfloat16", "float16": "float16", "float32": "float32"}
    name = _map.get(dtype_str, "bfloat16")
    return getattr(_torch, name, _torch.bfloat16)  # type: ignore[union-attr]


def _sample_logits(model: object) -> object:
    """Run a tiny dummy forward pass and return last-token logits."""
    device = next(model.parameters()).device  # type: ignore[union-attr]
    dummy = _torch.randint(1, 100, (1, 8), device=device)  # type: ignore[union-attr]
    with _torch.no_grad():  # type: ignore[union-attr]
        out = model(input_ids=dummy)  # type: ignore[union-attr]
    return out.logits[:, -1, :]


def _copy_processor(src: Path, dst: Path) -> None:
    """Copy processor/tokenizer files from src to dst."""
    if not _TRANSFORMERS_AVAILABLE or _AutoProcessor is None:
        logger.warning("transformers not available — skipping processor copy")
        return
    with contextlib.suppress(Exception):
        proc = _AutoProcessor.from_pretrained(str(src))  # type: ignore[union-attr]
        proc.save_pretrained(str(dst))
        logger.info("Processor saved to %s", dst)


def _count_params_from_config(model_dir: Path) -> int:
    """Estimate parameter count from model config.json (no model load)."""
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        return 0
    with contextlib.suppress(Exception):
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        # Qwen3-VL reports num_parameters in config if saved by transformers
        if "num_parameters" in cfg:
            return int(cfg["num_parameters"])
        # Fallback: rough estimate from hidden_size + num_layers
        h = cfg.get("hidden_size", 0)
        n = cfg.get("num_hidden_layers", 0)
        vocab = cfg.get("vocab_size", 0)
        return int(h * h * 4 * n + vocab * h)  # very rough
    return 0
