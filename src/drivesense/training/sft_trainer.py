"""LoRA SFT training for DriveSense-VLM.

Fine-tunes Qwen3-VL-2B-Instruct with LoRA adapters on the annotated
DriveSense dataset for autonomous vehicle hazard detection.

Phase 2a implementation.

Usage (HPC only):
    python -m drivesense.training --config configs/training.yaml
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image as PILImage

from drivesense.utils.config import load_config, merge_configs

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# GPU-only imports — all guarded for macOS dev
# ---------------------------------------------------------------------------
try:
    import torch  # type: ignore[import]
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

try:
    from transformers import (  # type: ignore[import]
        AutoProcessor,
        Trainer,
        TrainingArguments,
    )
    try:
        from transformers import (  # type: ignore[import]
            Qwen2_5_VLForConditionalGeneration as _QwenVLClass,
        )
    except ImportError:
        from transformers import AutoModelForCausalLM as _QwenVLClass  # type: ignore[import]
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    AutoProcessor = None  # type: ignore[assignment]
    Trainer = None  # type: ignore[assignment]
    TrainingArguments = None  # type: ignore[assignment]
    _QwenVLClass = None  # type: ignore[assignment]
    _TRANSFORMERS_AVAILABLE = False

try:
    from peft import LoraConfig, get_peft_model  # type: ignore[import]
    _PEFT_AVAILABLE = True
except ImportError:
    LoraConfig = None  # type: ignore[assignment]
    get_peft_model = None  # type: ignore[assignment]
    _PEFT_AVAILABLE = False

try:
    import wandb  # type: ignore[import]
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None  # type: ignore[assignment]
    _WANDB_AVAILABLE = False

try:
    from qwen_vl_utils import process_vision_info  # type: ignore[import]
    _QWEN_VL_UTILS_AVAILABLE = True
except ImportError:
    process_vision_info = None  # type: ignore[assignment]
    _QWEN_VL_UTILS_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DriveSenseSFTDataset
# ---------------------------------------------------------------------------


class DriveSenseSFTDataset:
    """Training dataset for Qwen3-VL SFT on DriveSense JSONL data.

    Loads SFT JSONL produced by Phase 1c, applies the Qwen3-VL chat template,
    processes images, and creates label tensors with non-assistant tokens masked
    to -100 so the loss is computed only on the structured JSON response.

    Args:
        jsonl_path: Path to ``sft_train.jsonl`` or ``sft_val.jsonl``.
        processor: Qwen3-VL ``AutoProcessor`` instance.
        max_seq_length: Sequences longer than this are truncated.
    """

    def __init__(
        self,
        jsonl_path: Path,
        processor: Any,  # noqa: ANN401
        max_seq_length: int = 2048,
    ) -> None:
        self._processor = processor
        self._max_seq_len = max_seq_length
        self._examples: list[dict] = []

        jsonl_path = Path(jsonl_path)
        if jsonl_path.exists():
            with jsonl_path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        self._examples.append(json.loads(line))
        else:
            logger.warning("SFT JSONL not found: %s", jsonl_path)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> dict:
        """Process one training example into model-ready tensors.

        Applies the Qwen3-VL chat template, processes images, tokenizes, and
        masks non-assistant tokens in the labels tensor.

        Args:
            idx: Example index.

        Returns:
            Dict with ``input_ids``, ``attention_mask``, ``labels``, and
            optionally ``pixel_values`` + ``image_grid_thw``.
        """
        entry = self._examples[idx]
        messages = _normalize_image_paths(entry.get("messages", []))

        text_full = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        text_prefix = self._processor.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        image_inputs = _extract_images(messages)

        inputs = self._processor(
            text=[text_full],
            images=image_inputs or None,
            return_tensors="pt",
            padding=False,
        )
        prefix_inputs = self._processor(
            text=[text_prefix],
            images=image_inputs or None,
            return_tensors="pt",
            padding=False,
        )
        prefix_len = prefix_inputs["input_ids"].shape[1]

        labels = inputs["input_ids"].clone()
        labels[0, :prefix_len] = -100  # mask system + user + assistant header

        seq_len = min(inputs["input_ids"].shape[1], self._max_seq_len)
        result: dict = {
            "input_ids": inputs["input_ids"][0, :seq_len],
            "attention_mask": inputs["attention_mask"][0, :seq_len],
            "labels": labels[0, :seq_len],
        }
        if "pixel_values" in inputs:
            result["pixel_values"] = inputs["pixel_values"]
        if "image_grid_thw" in inputs:
            result["image_grid_thw"] = inputs["image_grid_thw"]
        return result

    @staticmethod
    def find_assistant_start(input_ids: list[int], tokenizer: Any) -> int:  # noqa: ANN401
        """Find the token index where assistant content begins.

        Searches for the ``<|im_start|>assistant\\n`` header sequence in
        ``input_ids`` and returns the position immediately after it.

        Args:
            input_ids: Flat list of token IDs.
            tokenizer: Tokenizer with an ``encode()`` method.

        Returns:
            Index of the first assistant content token, or
            ``len(input_ids)`` if the header is not found.
        """
        header_text = "<|im_start|>assistant\n"
        try:
            header_ids = list(tokenizer.encode(header_text, add_special_tokens=False))
        except Exception:  # noqa: BLE001
            return len(input_ids)
        n = len(header_ids)
        ids = list(input_ids)
        for i in range(len(ids) - n + 1):
            if ids[i : i + n] == header_ids:
                return i + n
        return len(input_ids)


# ---------------------------------------------------------------------------
# DriveSenseDataCollator
# ---------------------------------------------------------------------------


class DriveSenseDataCollator:
    """Collator for variable-length multimodal Qwen3-VL sequences.

    Pads ``input_ids``, ``attention_mask``, and ``labels`` to the maximum
    length in the batch.  Concatenates ``pixel_values`` along the patch
    dimension and stacks ``image_grid_thw`` — because Qwen3-VL dynamically
    tiles images into variable patch counts that cannot be naively stacked.

    Args:
        processor: Qwen3-VL processor (supplies ``pad_token_id``).
        max_seq_length: Hard upper bound; sequences longer than this are truncated.
    """

    def __init__(self, processor: Any, max_seq_length: int = 2048) -> None:  # noqa: ANN401
        self._processor = processor
        self._max_seq_len = max_seq_length
        try:
            self._pad_id: int = processor.tokenizer.pad_token_id or 0
        except AttributeError:
            self._pad_id = 0

    def __call__(self, features: list[dict]) -> dict:
        """Collate a list of feature dicts into a batched tensor dict.

        Args:
            features: List of dicts from :meth:`DriveSenseSFTDataset.__getitem__`.

        Returns:
            Batched dict with ``input_ids``, ``attention_mask``, ``labels``, and
            optionally ``pixel_values`` and ``image_grid_thw``.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("torch is required for data collation")

        max_len = min(
            max(f["input_ids"].shape[0] for f in features),
            self._max_seq_len,
        )
        bs = len(features)

        input_ids = torch.full((bs, max_len), self._pad_id, dtype=torch.long)
        attention_mask = torch.zeros(bs, max_len, dtype=torch.long)
        labels = torch.full((bs, max_len), -100, dtype=torch.long)

        for i, feat in enumerate(features):
            seq_len = min(feat["input_ids"].shape[0], max_len)
            input_ids[i, :seq_len] = feat["input_ids"][:seq_len]
            attention_mask[i, :seq_len] = feat["attention_mask"][:seq_len]
            labels[i, :seq_len] = feat["labels"][:seq_len]

        batch: dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        _collate_pixel_values(features, batch)
        return batch


def _collate_pixel_values(features: list[dict], batch: dict) -> None:
    """Concatenate pixel_values and image_grid_thw across the batch.

    Qwen3-VL produces a variable number of image patches per image; they
    cannot be stacked and must be concatenated along the patch dimension.

    Args:
        features: List of feature dicts from the dataset.
        batch: Partially built batch dict — modified in-place.
    """
    pv_list = [f["pixel_values"] for f in features if "pixel_values" in f]
    if pv_list:
        batch["pixel_values"] = torch.cat(pv_list, dim=0)
    grid_list = [f["image_grid_thw"] for f in features if "image_grid_thw" in f]
    if grid_list:
        batch["image_grid_thw"] = torch.cat(grid_list, dim=0)


# ---------------------------------------------------------------------------
# Internal image helpers
# ---------------------------------------------------------------------------


def _normalize_image_paths(messages: list[dict]) -> list[dict]:
    """Ensure image paths in messages use ``file://`` prefix for Qwen3-VL.

    Args:
        messages: List of chat message dicts.

    Returns:
        New list with image paths normalised (originals are not mutated).
    """
    result: list[dict] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            new_content = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    path = item.get("image", "")
                    if path and not path.startswith(("file://", "http://", "https://")):
                        item = {**item, "image": f"file://{path}"}
                new_content.append(item)
            msg = {**msg, "content": new_content}
        result.append(msg)
    return result


def _extract_images(messages: list[dict]) -> list[PILImage.Image]:
    """Extract PIL images from message content.

    Tries ``qwen_vl_utils.process_vision_info`` first; falls back to a simple
    path-based loader when the package is not installed.

    Args:
        messages: Normalised chat messages (image paths use ``file://`` prefix).

    Returns:
        List of RGB PIL Images.  Empty list if no images found or files missing.
    """
    if _QWEN_VL_UTILS_AVAILABLE:
        try:
            image_inputs, _ = process_vision_info(messages)
            return image_inputs or []
        except Exception:  # noqa: BLE001
            pass
    return _extract_images_fallback(messages)


def _extract_images_fallback(messages: list[dict]) -> list[PILImage.Image]:
    """PIL-based image extractor used when qwen_vl_utils is unavailable."""
    images: list[PILImage.Image] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "image":
                continue
            path = item.get("image", "")
            if path.startswith("file://"):
                path = path[7:]
            p = Path(path)
            if p.exists():
                try:
                    images.append(PILImage.open(p).convert("RGB"))
                except Exception:  # noqa: BLE001
                    logger.warning("Could not load image: %s", p)
    return images


# ---------------------------------------------------------------------------
# Model + processor setup
# ---------------------------------------------------------------------------


def setup_model_and_processor(config: dict) -> tuple:
    """Load Qwen3-VL model and processor and apply LoRA adapters.

    Selects ``flash_attention_2`` if ``flash_attn`` is installed, otherwise
    falls back to ``sdpa``.  Enables gradient checkpointing with
    ``use_reentrant=False`` for PEFT compatibility.

    Args:
        config: Merged config dict (model + data + training).

    Returns:
        ``(model, processor, lora_config)`` ready for training.

    Raises:
        RuntimeError: If ``torch``, ``transformers``, or ``peft`` are not installed.
    """
    _require_gpu_deps()
    model_cfg = config.get("model", {})
    lora_cfg = config.get("lora", {})
    vision_cfg = config.get("vision", {})
    training_cfg = config.get("training", {})

    torch_dtype = getattr(torch, model_cfg.get("torch_dtype", "bfloat16"))
    attn_key = model_cfg.get("attn_implementation", "flash_attention_2")
    attn_impl = _resolve_attn_implementation(attn_key)

    logger.info("Loading %s (dtype=%s, attn=%s)", model_cfg["name"], torch_dtype, attn_impl)
    model = _QwenVLClass.from_pretrained(
        model_cfg["name"],
        revision=model_cfg.get("revision", "main"),
        torch_dtype=torch_dtype,
        attn_implementation=attn_impl,
        device_map="auto",
    )

    min_px = vision_cfg.get("min_pixels", 256) * 28 * 28
    max_px = vision_cfg.get("max_pixels", 1280) * 28 * 28
    processor = AutoProcessor.from_pretrained(
        model_cfg["name"], min_pixels=min_px, max_pixels=max_px
    )

    lora_config = LoraConfig(
        r=lora_cfg.get("rank", 32),
        lora_alpha=lora_cfg.get("alpha", 64),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
    )
    model = get_peft_model(model, lora_config)

    if training_cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    model.print_trainable_parameters()
    return model, processor, lora_config


def _resolve_attn_implementation(preferred: str) -> str:
    """Return the best available attention implementation.

    Args:
        preferred: Requested implementation (``"flash_attention_2"`` or ``"sdpa"``).

    Returns:
        ``preferred`` if its package is available, else ``"sdpa"``.
    """
    if preferred == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
            return "flash_attention_2"
        except ImportError:
            logger.warning("flash_attn not installed; using sdpa fallback")
            return "sdpa"
    return preferred


def _require_gpu_deps() -> None:
    """Raise RuntimeError if any GPU training package is missing."""
    missing = []
    if not _TORCH_AVAILABLE:
        missing.append("torch")
    if not _TRANSFORMERS_AVAILABLE:
        missing.append("transformers")
    if not _PEFT_AVAILABLE:
        missing.append("peft")
    if missing:
        raise RuntimeError(
            f"GPU training packages not installed: {', '.join(missing)}. "
            "Install with: pip install -e '.[training]'"
        )


# ---------------------------------------------------------------------------
# TrainingArguments setup
# ---------------------------------------------------------------------------


def setup_training_args(config: dict, output_dir: Path) -> Any:  # noqa: ANN401
    """Build HuggingFace TrainingArguments from the merged config.

    Maps ``configs/training.yaml`` values to TrainingArguments fields.
    Always sets ``remove_unused_columns=False`` (required for multimodal inputs).

    Args:
        config: Merged config dict (training section is used).
        output_dir: Directory for checkpoints and adapter weights.

    Returns:
        ``TrainingArguments`` instance.
    """
    _require_gpu_deps()
    t = config.get("training", {})
    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=t.get("num_epochs", 5),
        per_device_train_batch_size=t.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=t.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 4),
        learning_rate=t.get("learning_rate", 2e-5),
        weight_decay=t.get("weight_decay", 0.01),
        warmup_ratio=t.get("warmup_ratio", 0.05),
        lr_scheduler_type=t.get("lr_scheduler_type", "cosine"),
        bf16=t.get("bf16", True),
        gradient_checkpointing=t.get("gradient_checkpointing", True),
        dataloader_num_workers=t.get("dataloader_num_workers", 4),
        dataloader_pin_memory=t.get("dataloader_pin_memory", True),
        save_strategy=t.get("save_strategy", "epoch"),
        save_total_limit=t.get("save_total_limit", 3),
        load_best_model_at_end=t.get("load_best_model_at_end", True),
        metric_for_best_model=t.get("metric_for_best_model", "eval_loss"),
        greater_is_better=False,
        logging_steps=t.get("logging_steps", 10),
        eval_strategy=t.get("eval_strategy", "epoch"),
        report_to=t.get("report_to", "wandb"),
        remove_unused_columns=False,  # CRITICAL: VLM needs all columns
        max_steps=t.get("max_steps", -1),
    )


# ---------------------------------------------------------------------------
# Trainer setup
# ---------------------------------------------------------------------------


def setup_trainer(
    model: Any,  # noqa: ANN401
    processor: Any,  # noqa: ANN401
    train_dataset: DriveSenseSFTDataset,
    val_dataset: DriveSenseSFTDataset,
    training_args: Any,  # noqa: ANN401
    callbacks: list | None = None,
) -> Any:  # noqa: ANN401
    """Create a HuggingFace Trainer with the custom DriveSense data collator.

    Args:
        model: PEFT-wrapped Qwen3-VL model.
        processor: Qwen3-VL processor (used for padding token).
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        training_args: ``TrainingArguments`` instance.
        callbacks: Optional list of ``TrainerCallback`` instances.

    Returns:
        ``Trainer`` instance ready to call ``.train()``.
    """
    _require_gpu_deps()
    collator = DriveSenseDataCollator(
        processor, max_seq_length=training_args.max_seq_length
        if hasattr(training_args, "max_seq_length") else 2048
    )
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        callbacks=callbacks or [],
    )


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------


def train(config_path: str | Path) -> dict:
    """Main SFT training entry point.

    Loads all configs, builds the Qwen3-VL model with LoRA, trains, evaluates,
    saves the LoRA adapter, and returns training metrics.

    Args:
        config_path: Path to ``configs/training.yaml``.

    Returns:
        Metrics dict with ``train_loss``, ``eval_loss``, ``epochs_trained``,
        ``best_checkpoint``, and ``lora_adapter_dir``.
    """
    _require_gpu_deps()
    config_path = Path(config_path)
    config = _load_all_configs(config_path)

    training_cfg = config.get("training", {})
    output_dir = Path(training_cfg.get("output_dir", "outputs/training"))
    output_dir.mkdir(parents=True, exist_ok=True)

    _init_wandb(config)

    model, processor, _ = setup_model_and_processor(config)
    train_ds, val_ds = _load_datasets(config, processor)

    training_args = setup_training_args(config, output_dir)
    training_args = _apply_debug_overrides(training_args, config)

    callbacks = _build_callbacks(config, processor, val_ds)
    trainer = setup_trainer(model, processor, train_ds, val_ds, training_args, callbacks)

    resume_from = _resolve_checkpoint(training_cfg, output_dir)
    try:
        train_result = trainer.train(resume_from_checkpoint=resume_from)
        eval_result = trainer.evaluate()
        return _finalize_training(model, processor, trainer, train_result, eval_result, output_dir)
    except Exception as exc:  # noqa: BLE001
        logger.error("Training failed: %s", exc, exc_info=True)
        _save_emergency_checkpoint(trainer, output_dir)
        if _WANDB_AVAILABLE and wandb is not None:
            wandb.finish(exit_code=1)
        raise


def _load_all_configs(config_path: Path) -> dict:
    """Load and merge model.yaml + data.yaml + training.yaml."""
    cfg_dir = config_path.parent
    model_cfg = load_config(cfg_dir / "model.yaml")
    data_cfg = load_config(cfg_dir / "data.yaml")
    training_cfg = load_config(config_path)
    return merge_configs(model_cfg, data_cfg, training_cfg)


def _init_wandb(config: dict) -> None:
    """Initialize W&B run if available and configured."""
    if not _WANDB_AVAILABLE or wandb is None:
        return
    wandb_cfg = config.get("wandb", {})
    if not wandb_cfg.get("project"):
        return
    try:
        wandb.init(
            project=wandb_cfg["project"],
            entity=wandb_cfg.get("entity"),
            tags=wandb_cfg.get("tags", []),
            notes=wandb_cfg.get("notes", ""),
            config=config,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("W&B init failed (offline mode?): %s", exc)


def _load_datasets(
    config: dict, processor: Any  # noqa: ANN401
) -> tuple[DriveSenseSFTDataset, DriveSenseSFTDataset]:
    """Locate and load train/val SFT JSONL datasets."""
    ann_cfg = config.get("annotation", {})
    sft_dir = Path(ann_cfg.get("sft_output_dir", "outputs/data/sft_ready"))
    max_seq = config.get("training", {}).get("max_seq_length", 2048)
    logger.info("Loading SFT datasets from %s", sft_dir)
    train_ds = DriveSenseSFTDataset(sft_dir / "sft_train.jsonl", processor, max_seq)
    val_ds = DriveSenseSFTDataset(sft_dir / "sft_val.jsonl", processor, max_seq)
    logger.info("Train: %d examples | Val: %d examples", len(train_ds), len(val_ds))
    return train_ds, val_ds


def _apply_debug_overrides(
    training_args: Any, config: dict  # noqa: ANN401
) -> Any:  # noqa: ANN401
    """Apply debug-mode config overrides to TrainingArguments (returns same object)."""
    debug_cfg = config.get("training", {}).get("debug", {})
    if not debug_cfg:
        return training_args
    for attr, key in [
        ("max_steps", "max_steps"),
        ("eval_steps", "eval_steps"),
        ("logging_steps", "logging_steps"),
        ("num_train_epochs", "num_epochs"),
        ("save_strategy", "save_strategy"),
    ]:
        if key in debug_cfg:
            setattr(training_args, attr, debug_cfg[key])
    return training_args


def _build_callbacks(
    config: dict, processor: Any, val_ds: DriveSenseSFTDataset  # noqa: ANN401
) -> list:
    """Build the list of training callbacks from config."""
    from drivesense.training.callbacks import (
        EarlyStoppingCallback,
        GPUMemoryCallback,
        SamplePredictionCallback,
        TrainingMetricsCallback,
    )
    es_cfg = config.get("early_stopping", {})
    return [
        GPUMemoryCallback(),
        TrainingMetricsCallback(),
        EarlyStoppingCallback(
            patience=int(es_cfg.get("patience", 2)),
            threshold=float(es_cfg.get("threshold", 0.001)),
        ),
        SamplePredictionCallback(processor, val_ds, num_samples=3),
    ]


def _resolve_checkpoint(training_cfg: dict, output_dir: Path) -> str | None:
    """Determine the checkpoint path to resume from, if any."""
    resume_from = training_cfg.get("resume_from_checkpoint")
    if resume_from == "latest":
        checkpoints = sorted(output_dir.glob("checkpoint-*"))
        return str(checkpoints[-1]) if checkpoints else None
    return str(resume_from) if resume_from else None


def _finalize_training(
    model: Any,  # noqa: ANN401
    processor: Any,  # noqa: ANN401
    trainer: Any,  # noqa: ANN401
    train_result: Any,  # noqa: ANN401
    eval_result: dict,
    output_dir: Path,
) -> dict:
    """Save adapter, write metrics JSON, log W&B artifact, return metrics."""
    lora_dir = output_dir / "lora_adapter"
    model.save_pretrained(str(lora_dir))
    processor.save_pretrained(str(lora_dir))
    logger.info("LoRA adapter saved to %s", lora_dir)

    metrics = {
        "train_loss": getattr(train_result, "training_loss", None),
        "eval_loss": eval_result.get("eval_loss"),
        "epochs_trained": train_result.metrics.get("epoch") if train_result.metrics else None,
        "best_checkpoint": str(output_dir),
        "lora_adapter_dir": str(lora_dir),
    }
    with (output_dir / "training_metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    if _WANDB_AVAILABLE and wandb is not None:
        try:
            wandb.log({"final/eval_loss": metrics["eval_loss"]})
            artifact = wandb.Artifact("lora-adapter", type="model")
            artifact.add_dir(str(lora_dir))
            wandb.log_artifact(artifact)
            wandb.finish()
        except Exception as exc:  # noqa: BLE001
            logger.warning("W&B finalization failed: %s", exc)

    return metrics


def _save_emergency_checkpoint(trainer: Any, output_dir: Path) -> None:  # noqa: ANN401
    """Save a best-effort emergency checkpoint on training failure."""
    emergency_dir = output_dir / "emergency_checkpoint"
    try:
        trainer.save_model(str(emergency_dir))
        logger.info("Emergency checkpoint saved to %s", emergency_dir)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to save emergency checkpoint: %s", exc)
