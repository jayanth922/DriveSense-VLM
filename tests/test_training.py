"""Tests for Phase 2a: LoRA SFT training pipeline.

All tests run on CPU-only macOS.  The actual Qwen3-VL model is never loaded;
model, processor, and GPU operations are fully mocked.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

torch = pytest.importorskip("torch", reason="torch not installed")

from drivesense.training.sft_trainer import (  # noqa: E402
    DriveSenseDataCollator,
    DriveSenseSFTDataset,
    _normalize_image_paths,
)
from drivesense.utils.config import load_config, merge_configs  # noqa: E402

# ---------------------------------------------------------------------------
# Mock processor
# ---------------------------------------------------------------------------


class MockProcessor:
    """Minimal Qwen3-VL processor mock for CPU tests."""

    class _Tok:
        pad_token_id = 0
        eos_token_id = 2

        def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
            return [hash(c) % 900 + 100 for c in text[:6]]

    tokenizer = _Tok()

    def apply_chat_template(
        self,
        messages: list,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", c.get("image", ""))
                    for c in content
                    if isinstance(c, dict)
                )
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    def __call__(
        self,
        text: object = None,
        images: object = None,
        return_tensors: str = "pt",
        padding: bool = False,
        **kwargs: object,
    ) -> dict:
        import torch

        # Vary seq_len with text length so prefix (<= full) produces fewer tokens.
        t = text[0] if isinstance(text, list) else (text or "")
        seq_len = max(4, min(20, len(str(t)) // 10))
        return {
            "input_ids": torch.randint(1, 100, (1, seq_len)),
            "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
            "pixel_values": torch.randn(4, 3, 28, 28),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        }

    def batch_decode(self, ids: object, skip_special_tokens: bool = True) -> list[str]:
        return ["mock decoded output"]

    def save_pretrained(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def processor() -> MockProcessor:
    return MockProcessor()


@pytest.fixture()
def sample_sft_jsonl(tmp_path: Path) -> Path:
    """Create a minimal SFT JSONL file with 3 examples."""
    examples = []
    for i in range(3):
        examples.append({
            "messages": [
                {"role": "system", "content": "You are DriveSense."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"/fake/frame_{i}.png"},
                        {"type": "text", "text": "Analyze this image."},
                    ],
                },
                {
                    "role": "assistant",
                    "content": json.dumps({
                        "hazards": [{
                            "bbox_2d": [100, 200, 400, 600],
                            "label": "high_density",
                            "severity": "medium",
                            "reasoning": "Multiple agents detected at high density.",
                            "action": "Reduce speed.",
                        }],
                        "scene_summary": "Urban street scene.",
                        "ego_context": {
                            "weather": "clear", "time_of_day": "day", "road_type": "urban"
                        },
                    }),
                },
            ],
            "images": [f"/fake/frame_{i}.png"],
            "frame_id": f"ns_{i:03d}",
            "source": "nuscenes",
        })
    p = tmp_path / "sft_train.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in examples), encoding="utf-8")
    return p


@pytest.fixture()
def dataset(sample_sft_jsonl: Path, processor: MockProcessor) -> DriveSenseSFTDataset:
    return DriveSenseSFTDataset(sample_sft_jsonl, processor, max_seq_length=64)


# ---------------------------------------------------------------------------
# DriveSenseSFTDataset tests
# ---------------------------------------------------------------------------


class TestSFTDataset:
    def test_loading(self, dataset: DriveSenseSFTDataset) -> None:
        """Dataset loads correct number of examples."""
        assert len(dataset) == 3

    def test_loading_missing_file(self, processor: MockProcessor, tmp_path: Path) -> None:
        """Missing JSONL file produces an empty dataset (no crash)."""
        ds = DriveSenseSFTDataset(tmp_path / "nonexistent.jsonl", processor)
        assert len(ds) == 0

    def test_getitem_returns_expected_keys(self, dataset: DriveSenseSFTDataset) -> None:
        """__getitem__ returns a dict with at least the required tensor keys."""
        item = dataset[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item

    def test_getitem_includes_image_tensors(self, dataset: DriveSenseSFTDataset) -> None:
        """pixel_values and image_grid_thw are included from processor output."""
        item = dataset[0]
        assert "pixel_values" in item
        assert "image_grid_thw" in item

    def test_label_masking_has_minus_100(self, dataset: DriveSenseSFTDataset) -> None:
        """Non-assistant tokens are masked to -100 in labels."""
        item = dataset[0]
        labels = item["labels"]
        # prefix tokens must all be -100
        assert (labels == -100).any(), "Expected some -100 labels (prefix masking)"

    def test_label_masking_assistant_tokens_kept(self, dataset: DriveSenseSFTDataset) -> None:
        """At least some labels are not -100 (assistant content preserved)."""
        item = dataset[0]
        labels = item["labels"]
        assert (labels != -100).any(), "Expected some non-(-100) labels (assistant content)"

    def test_truncation(self, sample_sft_jsonl: Path, processor: MockProcessor) -> None:
        """Sequences are truncated to max_seq_length."""
        ds = DriveSenseSFTDataset(sample_sft_jsonl, processor, max_seq_length=10)
        item = ds[0]
        assert item["input_ids"].shape[0] <= 10

    def test_find_assistant_start_finds_header(self, processor: MockProcessor) -> None:
        """find_assistant_start returns position after the assistant header."""
        tokenizer = processor.tokenizer
        header = "<|im_start|>assistant\n"
        header_ids = tokenizer.encode(header, add_special_tokens=False)
        dummy_prefix = [1, 2, 3]
        dummy_suffix = [99, 100]
        input_ids = dummy_prefix + header_ids + dummy_suffix
        pos = DriveSenseSFTDataset.find_assistant_start(input_ids, tokenizer)
        assert pos == len(dummy_prefix) + len(header_ids)

    def test_find_assistant_start_fallback(self, processor: MockProcessor) -> None:
        """find_assistant_start returns len(input_ids) when header is absent."""
        ids = [1, 2, 3, 4, 5]
        pos = DriveSenseSFTDataset.find_assistant_start(ids, processor.tokenizer)
        assert pos == len(ids)


# ---------------------------------------------------------------------------
# DriveSenseDataCollator tests
# ---------------------------------------------------------------------------


class TestDataCollator:
    @pytest.fixture()
    def collator(self, processor: MockProcessor) -> DriveSenseDataCollator:
        return DriveSenseDataCollator(processor, max_seq_length=64)

    def test_collator_pads_input_ids(
        self, collator: DriveSenseDataCollator, dataset: DriveSenseSFTDataset
    ) -> None:
        """input_ids in batch share the same padded length."""
        features = [dataset[0], dataset[1]]
        batch = collator(features)
        assert batch["input_ids"].shape[0] == 2
        assert batch["input_ids"].shape[1] == batch["attention_mask"].shape[1]

    def test_collator_pads_labels_with_minus_100(
        self, collator: DriveSenseDataCollator, dataset: DriveSenseSFTDataset
    ) -> None:
        """Padded label positions are -100."""
        features = [dataset[0], dataset[1]]
        batch = collator(features)
        seq_len = batch["labels"].shape[1]
        for i in range(batch["labels"].shape[0]):
            item_len = dataset[i]["labels"].shape[0]
            if item_len < seq_len:
                assert (batch["labels"][i, item_len:] == -100).all()

    def test_collator_concatenates_pixel_values(
        self, collator: DriveSenseDataCollator, dataset: DriveSenseSFTDataset
    ) -> None:
        """pixel_values are concatenated (not stacked) along patch dimension."""
        features = [dataset[0], dataset[1]]
        batch = collator(features)
        assert "pixel_values" in batch
        # each item has 4 patches (from MockProcessor), so batch should have 8
        assert batch["pixel_values"].shape[0] == 8

    def test_collator_stacks_image_grid_thw(
        self, collator: DriveSenseDataCollator, dataset: DriveSenseSFTDataset
    ) -> None:
        """image_grid_thw tensors are concatenated into a single tensor."""
        features = [dataset[0], dataset[1]]
        batch = collator(features)
        assert "image_grid_thw" in batch
        assert batch["image_grid_thw"].shape[0] == 2


# ---------------------------------------------------------------------------
# Image path normalisation
# ---------------------------------------------------------------------------


class TestNormalizeImagePaths:
    def test_plain_path_gets_file_prefix(self) -> None:
        """Bare file paths are prefixed with file://."""
        msgs = [{"role": "user", "content": [{"type": "image", "image": "/tmp/f.png"}]}]
        result = _normalize_image_paths(msgs)
        assert result[0]["content"][0]["image"] == "file:///tmp/f.png"

    def test_file_prefix_not_doubled(self) -> None:
        """Paths already prefixed with file:// are not modified."""
        msgs = [{"role": "user", "content": [{"type": "image", "image": "file:///tmp/f.png"}]}]
        result = _normalize_image_paths(msgs)
        assert result[0]["content"][0]["image"] == "file:///tmp/f.png"

    def test_http_url_not_modified(self) -> None:
        """HTTP URLs are left unchanged."""
        msgs = [{"role": "user", "content": [{"type": "image", "image": "http://x.com/f.png"}]}]
        result = _normalize_image_paths(msgs)
        assert result[0]["content"][0]["image"] == "http://x.com/f.png"

    def test_non_image_items_unchanged(self) -> None:
        """Non-image content items are passed through unmodified."""
        msgs = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
        result = _normalize_image_paths(msgs)
        assert result[0]["content"][0]["text"] == "hello"


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestConfigLoading:
    def test_all_three_configs_load(self) -> None:
        """model.yaml, data.yaml, and training.yaml all load without error."""
        cfg_dir = Path("configs")
        model_cfg = load_config(cfg_dir / "model.yaml")
        data_cfg = load_config(cfg_dir / "data.yaml")
        training_cfg = load_config(cfg_dir / "training.yaml")
        assert "model" in model_cfg
        assert "nuscenes" in data_cfg or "paths" in data_cfg
        assert "training" in training_cfg

    def test_merged_config_has_all_sections(self) -> None:
        """Merged config contains model, lora, training, wandb sections."""
        cfg_dir = Path("configs")
        merged = merge_configs(
            load_config(cfg_dir / "model.yaml"),
            load_config(cfg_dir / "data.yaml"),
            load_config(cfg_dir / "training.yaml"),
        )
        for key in ("model", "lora", "training", "wandb"):
            assert key in merged, f"Missing key: {key}"

    def test_lora_config_matches_yaml(self) -> None:
        """LoRA config values match model.yaml expectations."""
        cfg = load_config("configs/model.yaml")
        lora = cfg.get("lora", {})
        assert lora["rank"] == 32
        assert lora["alpha"] == 64
        assert "q_proj" in lora["target_modules"]
        assert "v_proj" in lora["target_modules"]

    def test_wandb_config_has_project(self) -> None:
        """W&B config includes project name and tags."""
        cfg = load_config("configs/training.yaml")
        wandb_cfg = cfg.get("wandb", {})
        assert wandb_cfg.get("project") == "drivesense-vlm"
        assert isinstance(wandb_cfg.get("tags", []), list)

    def test_debug_config_present(self) -> None:
        """training.debug section exists in training.yaml."""
        cfg = load_config("configs/training.yaml")
        debug = cfg.get("training", {}).get("debug", {})
        assert "max_steps" in debug
        assert "num_epochs" in debug


# ---------------------------------------------------------------------------
# TrainingArguments + LoRA config construction (skip if deps absent)
# ---------------------------------------------------------------------------


def _has_transformers_and_accelerate() -> bool:
    import importlib.util
    return bool(
        importlib.util.find_spec("transformers") and importlib.util.find_spec("accelerate")
    )


class TestTrainingArgs:
    @pytest.mark.skipif(
        not _has_transformers_and_accelerate(),
        reason="transformers + accelerate not installed",
    )
    def test_training_args_from_config(self) -> None:
        """setup_training_args maps config keys to TrainingArguments fields."""
        from drivesense.training.sft_trainer import setup_training_args

        cfg = {
            "training": {
                "num_epochs": 3,
                "per_device_train_batch_size": 2,
                "learning_rate": 1e-4,
                "gradient_accumulation_steps": 2,
                "save_strategy": "epoch",
                "eval_strategy": "epoch",
                "max_steps": -1,
                "bf16": False,
                "gradient_checkpointing": False,
                "dataloader_num_workers": 0,
                "dataloader_pin_memory": False,
                "logging_steps": 5,
                "report_to": "none",
                "weight_decay": 0.0,
                "warmup_ratio": 0.0,
                "lr_scheduler_type": "linear",
                "save_total_limit": 1,
                "load_best_model_at_end": False,
                "metric_for_best_model": "eval_loss",
                "per_device_eval_batch_size": 2,
            }
        }
        with patch("drivesense.training.sft_trainer._TORCH_AVAILABLE", True), \
             patch("drivesense.training.sft_trainer._TRANSFORMERS_AVAILABLE", True), \
             patch("drivesense.training.sft_trainer._PEFT_AVAILABLE", True):
            args = setup_training_args(cfg, Path("/tmp/ds_test_args"))

        assert args.num_train_epochs == 3
        assert args.learning_rate == pytest.approx(1e-4)
        assert args.remove_unused_columns is False


class TestLoraConfig:
    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("peft"),
        reason="peft not installed",
    )
    def test_lora_config_from_yaml(self) -> None:
        """LoRA config can be built from model.yaml values."""
        from peft import LoraConfig

        cfg = load_config("configs/model.yaml")
        lora_cfg = cfg.get("lora", {})
        lc = LoraConfig(
            r=lora_cfg["rank"],
            lora_alpha=lora_cfg["alpha"],
            target_modules=lora_cfg["target_modules"],
            lora_dropout=lora_cfg["dropout"],
            bias=lora_cfg["bias"],
            task_type=lora_cfg["task_type"],
        )
        assert lc.r == 32
        assert lc.lora_alpha == 64
        assert "q_proj" in lc.target_modules


# ---------------------------------------------------------------------------
# Dry-run mock test
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_dry_run_with_mock_model(self, tmp_path: Path) -> None:
        """run_dry_run with a mock model completes without error."""
        sys.path.insert(0, str(_SRC.parent / "scripts"))
        from run_training import _create_mock_model_processor

        model, mock_proc = _create_mock_model_processor()

        # Build a minimal SFT dataset
        example = {
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": [
                    {"type": "image", "image": "/fake/img.png"},
                    {"type": "text", "text": "analyze"},
                ]},
                {"role": "assistant", "content": '{"hazards":[]}'},
            ],
            "images": ["/fake/img.png"],
            "frame_id": "test_001",
            "source": "nuscenes",
        }
        sft_file = tmp_path / "sft_train.jsonl"
        sft_file.write_text(json.dumps(example), encoding="utf-8")

        ds = DriveSenseSFTDataset(sft_file, mock_proc, max_seq_length=16)
        collator = DriveSenseDataCollator(mock_proc, max_seq_length=16)
        batch = collator([ds[0]])

        import torch

        with torch.no_grad():
            out = model(**batch)
        assert hasattr(out, "loss")
        assert out.loss.item() >= 0.0
