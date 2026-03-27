#!/usr/bin/env python3
"""Generate model predictions on the test (or val) split.

Usage:
    python scripts/run_generate_predictions.py
    python scripts/run_generate_predictions.py --split test
    python scripts/run_generate_predictions.py --split val
    python scripts/run_generate_predictions.py --adapter-path outputs/training/checkpoint-best
    python scripts/run_generate_predictions.py --mock
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from drivesense.utils.config import load_config, merge_configs  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_generate_predictions")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="DriveSense-VLM: Generate predictions on a data split",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--split",
        default="test",
        choices=["test", "val"],
        help="Dataset split to run inference on",
    )
    p.add_argument(
        "--adapter-path",
        default=None,
        help="Path to LoRA adapter (overrides config default)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output JSONL path (default: outputs/predictions/<split>_predictions.jsonl)",
    )
    p.add_argument(
        "--mock",
        action="store_true",
        help="Use a tiny mock model (no model download required)",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override max_new_tokens from config",
    )
    p.add_argument(
        "--config",
        default="configs/eval.yaml",
        help="Path to eval.yaml",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main generation logic
# ---------------------------------------------------------------------------


def load_split_data(config: dict, split: str) -> list[dict]:
    """Load ground truth records for the given split.

    Args:
        config: Merged config dict.
        split:  ``"test"`` or ``"val"``.

    Returns:
        List of GT records with at minimum ``frame_id``.
    """
    sft_dir = Path(
        config.get("annotation", {}).get(
            "sft_output_dir", "outputs/data/sft_ready"
        )
    )
    split_file = sft_dir / f"sft_{split}.jsonl"
    if not split_file.exists():
        logger.error("Split file not found: %s", split_file)
        logger.info("Run the annotation pipeline first: python scripts/run_annotation_pipeline.py")
        sys.exit(1)

    records: list[dict] = []
    with split_file.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    logger.info("Loaded %d records from %s", len(records), split_file)
    return records


def setup_model(config: dict, adapter_path: str | None, mock: bool) -> tuple:
    """Load the model and processor for inference.

    Args:
        config:       Merged config dict.
        adapter_path: Optional LoRA adapter path override.
        mock:         Use tiny mock model instead.

    Returns:
        ``(model, processor)`` tuple.
    """
    if mock:
        return _create_mock_model_processor()

    try:
        from drivesense.training.sft_trainer import setup_model_and_processor
    except ImportError as exc:
        logger.error(
            "Training deps not installed: %s\n"
            "Install: pip install -e '.[training]'",
            exc,
        )
        sys.exit(1)

    if adapter_path:
        config.setdefault("model", {})["adapter_path"] = adapter_path

    model, processor, _ = setup_model_and_processor(config)
    return model, processor


def run_inference(
    model: object,
    processor: object,
    records: list[dict],
    output_path: Path,
    max_new_tokens: int = 512,
) -> dict:
    """Run inference over all records and write predictions JSONL.

    Args:
        model:          Loaded model.
        processor:      Loaded processor.
        records:        GT split records.
        output_path:    Destination JSONL path.
        max_new_tokens: Token budget for generation.

    Returns:
        Summary stats dict.
    """
    try:
        import torch as _torch_check  # noqa: F401
    except ImportError:
        logger.error("torch not installed")
        sys.exit(1)

    try:
        from tqdm import tqdm as _tqdm  # type: ignore[import]
        _has_tqdm = True
    except ImportError:
        _has_tqdm = False

    from drivesense.data.annotation import AnnotationValidator
    from drivesense.training.sft_trainer import _extract_images, _normalize_image_paths

    device = next(model.parameters()).device  # type: ignore[union-attr]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    iterator = _tqdm(records, desc=f"Generating ({output_path.stem})") if _has_tqdm else records
    total = parse_ok = parse_fail = 0
    total_ms = 0.0

    with output_path.open("w", encoding="utf-8") as fh:
        for rec in iterator:
            frame_id = rec.get("frame_id", f"frame_{total:06d}")
            messages = rec.get("messages", [])

            # Use only system + user messages
            prompt = [m for m in messages if m.get("role") != "assistant"]
            if not prompt:
                prompt = [
                    {"role": "system", "content": "You are DriveSense-VLM."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Analyse this dashcam image for hazards."},
                        ],
                    },
                ]
            prompt = _normalize_image_paths(prompt)

            t0 = time.perf_counter()
            raw_output = _generate_single(
                model, processor, prompt, device, max_new_tokens, _extract_images
            )
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            total_ms += elapsed_ms

            parsed = AnnotationValidator.parse_llm_response(raw_output)
            success = parsed is not None
            parse_ok += int(success)
            parse_fail += int(not success)
            total += 1

            fh.write(
                json.dumps({
                    "frame_id": frame_id,
                    "raw_output": raw_output,
                    "parsed_output": parsed,
                    "parse_success": success,
                    "generation_time_ms": elapsed_ms,
                }) + "\n"
            )

    avg_ms = total_ms / total if total > 0 else 0.0
    summary = {
        "total": total,
        "parse_success": parse_ok,
        "parse_failures": parse_fail,
        "parse_success_rate": round(parse_ok / total, 4) if total > 0 else 0.0,
        "avg_generation_ms": round(avg_ms, 1),
        "output_path": str(output_path),
    }
    logger.info(
        "Done: %d frames, %.0f%% parsed OK, avg %.0f ms/frame",
        total,
        (parse_ok / total * 100) if total > 0 else 0,
        avg_ms,
    )
    return summary


def _generate_single(
    model: object,
    processor: object,
    messages: list[dict],
    device: object,
    max_new_tokens: int,
    extract_images_fn: object,
) -> str:
    """Run generation for one example and return the decoded text.

    Args:
        model:             Loaded model.
        processor:         Loaded processor.
        messages:          Prompt messages (system + user).
        device:            Target device.
        max_new_tokens:    Token budget.
        extract_images_fn: Function to extract PIL images from messages.

    Returns:
        Decoded output string (empty string on error).
    """
    import torch

    try:
        text = processor.apply_chat_template(  # type: ignore[union-attr]
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs = extract_images_fn(messages)  # type: ignore[operator]
        inputs = processor(  # type: ignore[operator]
            text=[text],
            images=image_inputs or None,
            return_tensors="pt",
            padding=False,
        )
        inputs_gpu = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            out_ids = model.generate(  # type: ignore[union-attr]
                **inputs_gpu,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )
        prompt_len = inputs["input_ids"].shape[1]
        return processor.batch_decode(  # type: ignore[union-attr]
            out_ids[:, prompt_len:], skip_special_tokens=True
        )[0]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Generation error: %s", exc)
        return ""


def _create_mock_model_processor() -> tuple:
    """Return tiny CPU model + mock processor for --mock runs.

    Returns:
        ``(model, processor)`` tuple.
    """
    import torch.nn as nn

    class _MockModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(10, 10)

        def generate(self, input_ids: object = None, **kwargs: object) -> object:  # type: ignore[override]
            import torch as _t
            n = kwargs.get("max_new_tokens", 8)
            prompt_len = input_ids.shape[1] if hasattr(input_ids, "shape") else 1  # type: ignore[union-attr]
            return _t.zeros(1, prompt_len + n, dtype=_t.long)

        def parameters(self, recurse: bool = True):  # noqa: ANN201, ANN202
            return super().parameters(recurse)

    class _MockProcessor:
        class _Tok:
            pad_token_id = 0

        tokenizer = _Tok()

        def apply_chat_template(
            self,
            messages: list,
            tokenize: bool = False,
            add_generation_prompt: bool = False,
        ) -> str:
            return " ".join(str(m) for m in messages)

        def __call__(self, text: object = None, images: object = None,
                     return_tensors: str = "pt", **kw: object) -> dict:
            import torch as _t
            return {
                "input_ids": _t.randint(1, 100, (1, 8)),
                "attention_mask": _t.ones(1, 8, dtype=_t.long),
            }

        def batch_decode(self, ids: object, skip_special_tokens: bool = True) -> list[str]:
            return [
                '{"hazards": [], "scene_summary": "mock output", '
                '"ego_context": {"weather": "clear", "time_of_day": "day", "road_type": "urban"}}'
            ]

        def save_pretrained(self, path: str) -> None:
            Path(path).mkdir(parents=True, exist_ok=True)

    return _MockModel(), _MockProcessor()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    cfg_dir = Path(args.config).parent
    model_cfg = load_config(cfg_dir / "model.yaml")
    data_cfg = load_config(cfg_dir / "data.yaml")
    training_cfg = load_config(cfg_dir / "training.yaml")
    eval_cfg = load_config(args.config)
    config = merge_configs(model_cfg, data_cfg, training_cfg, eval_cfg)

    gen_cfg = config.get("generation", {})
    max_new_tokens = args.max_new_tokens or int(gen_cfg.get("max_new_tokens", 512))

    output_path = Path(
        args.output
        or f"outputs/predictions/{args.split}_predictions.jsonl"
    )

    logger.info("Split: %s  |  Output: %s  |  Mock: %s", args.split, output_path, args.mock)

    records = load_split_data(config, args.split)
    model, processor = setup_model(config, args.adapter_path, args.mock)

    summary = run_inference(
        model=model,
        processor=processor,
        records=records,
        output_path=output_path,
        max_new_tokens=max_new_tokens,
    )

    print("\n--- Prediction Generation Summary ---")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
