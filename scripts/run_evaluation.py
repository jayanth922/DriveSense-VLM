#!/usr/bin/env python3
"""Phase 2b: Run DriveSense-VLM evaluation.

Usage:
    python scripts/run_evaluation.py --level 1
    python scripts/run_evaluation.py --level 2           # needs ANTHROPIC_API_KEY
    python scripts/run_evaluation.py --level 1 2
    python scripts/run_evaluation.py --level 1 2 --mock-judge
    python scripts/run_evaluation.py --predictions outputs/predictions/test_predictions.jsonl
    python scripts/run_evaluation.py --generate-predictions
    python scripts/run_evaluation.py --generate-predictions --level 1 2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
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
logger = logging.getLogger("run_evaluation")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="DriveSense-VLM Phase 2b: Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--config", default="configs/eval.yaml", help="Path to eval.yaml"
    )
    p.add_argument(
        "--level",
        nargs="+",
        default=["1"],
        choices=["1", "2"],
        help="Evaluation levels to run (1 = grounding, 2 = reasoning)",
    )
    p.add_argument(
        "--mock-judge",
        action="store_true",
        help="Use MockLLMJudge for Level 2 (no API key required)",
    )
    p.add_argument(
        "--predictions",
        default=None,
        help="Path to predictions JSONL (overrides config)",
    )
    p.add_argument(
        "--ground-truth",
        default=None,
        help="Path to ground truth JSONL (overrides config)",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (overrides config)",
    )
    p.add_argument(
        "--generate-predictions",
        action="store_true",
        help="Run inference on test set before evaluating",
    )
    p.add_argument(
        "--mock",
        action="store_true",
        help="Use mock model for --generate-predictions (no download required)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Prediction generation
# ---------------------------------------------------------------------------


def generate_predictions(config: dict, mock: bool = False) -> Path:
    """Run inference on the test split and save predictions to JSONL.

    Args:
        config: Merged config dict.
        mock:   If True, use a mock model (no download).

    Returns:
        Path to the saved predictions JSONL file.
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        logger.error("torch not installed. Install with: pip install -e '.[training]'")
        sys.exit(1)

    from drivesense.eval.grounding import GroundingEvaluator

    eval_data = config.get("eval_data", {})
    test_path = Path(
        eval_data.get("ground_truth_path", "outputs/data/sft_ready/sft_test.jsonl")
    )
    gen_cfg = config.get("generation", {})
    output_path = Path(
        gen_cfg.get("predictions_path", "outputs/predictions/test_predictions.jsonl")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    max_new_tokens: int = int(gen_cfg.get("max_new_tokens", 512))

    if mock:
        model, processor = _create_mock_model_processor()
    else:
        from drivesense.training.sft_trainer import setup_model_and_processor
        model, processor, _ = setup_model_and_processor(config)

    gt_evaluator = GroundingEvaluator(config)
    gt_list = gt_evaluator.load_ground_truth(test_path)

    _run_inference_loop(
        model=model,
        processor=processor,
        gt_list=gt_list,
        output_path=output_path,
        max_new_tokens=max_new_tokens,
    )
    logger.info("Predictions saved to %s", output_path)
    return output_path


def _run_inference_loop(
    model: object,
    processor: object,
    gt_list: list[dict],
    output_path: Path,
    max_new_tokens: int = 512,
) -> None:
    """Run the model over gt_list and write predictions JSONL.

    Args:
        model:          Loaded model (may be mock).
        processor:      Loaded processor.
        gt_list:        GT records with ``frame_id`` and ``images``.
        output_path:    Destination JSONL path.
        max_new_tokens: Generation budget.
    """
    import time

    try:
        from tqdm import tqdm as _tqdm  # type: ignore[import]
    except ImportError:
        _tqdm = None  # type: ignore[assignment]

    from drivesense.data.annotation import AnnotationValidator
    from drivesense.training.sft_trainer import _normalize_image_paths

    device = next(model.parameters()).device  # type: ignore[union-attr]
    iterator = _tqdm(gt_list, desc="Generating") if _tqdm else gt_list

    with output_path.open("w", encoding="utf-8") as fh:
        for gt in iterator:
            frame_id = gt.get("frame_id", "")
            messages = gt.get("messages", [])
            if not messages:
                # Build messages from GT dict (fallback)
                messages = _build_messages_from_gt(gt)

            # Use only system + user (no assistant)
            prompt_messages = [m for m in messages if m.get("role") != "assistant"]
            prompt_messages = _normalize_image_paths(prompt_messages)

            t0 = time.perf_counter()
            raw_output = _generate_one(
                model, processor, prompt_messages, max_new_tokens, device
            )
            elapsed_ms = int((time.perf_counter() - t0) * 1000)

            parsed = AnnotationValidator.parse_llm_response(raw_output)
            record = {
                "frame_id": frame_id,
                "raw_output": raw_output,
                "parsed_output": parsed,
                "parse_success": parsed is not None,
                "generation_time_ms": elapsed_ms,
            }
            fh.write(json.dumps(record) + "\n")


def _generate_one(
    model: object,
    processor: object,
    messages: list[dict],
    max_new_tokens: int,
    device: object,
) -> str:
    """Run a single generation step and return decoded text.

    Args:
        model:          Loaded model.
        processor:      Loaded processor.
        messages:       Prompt messages (system + user only).
        max_new_tokens: Token budget.
        device:         Target device.

    Returns:
        Decoded output string.
    """
    import torch

    from drivesense.training.sft_trainer import _extract_images

    try:
        text = processor.apply_chat_template(  # type: ignore[union-attr]
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs = _extract_images(messages)
        inputs = processor(  # type: ignore[operator]
            text=[text],
            images=image_inputs or None,
            return_tensors="pt",
            padding=False,
        )
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            out_ids = model.generate(  # type: ignore[union-attr]
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                do_sample=False,
            )
        prompt_len = inputs["input_ids"].shape[1]
        return processor.batch_decode(  # type: ignore[union-attr]
            out_ids[:, prompt_len:], skip_special_tokens=True
        )[0]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Generation error for frame: %s", exc)
        return ""


def _build_messages_from_gt(gt: dict) -> list[dict]:
    """Build a minimal prompt from a GT dict (no messages key)."""
    return [
        {"role": "system", "content": "You are DriveSense-VLM."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyse this dashcam image for hazards."},
            ],
        },
    ]


def _create_mock_model_processor() -> tuple:
    """Return a tiny CPU mock model + processor for testing.

    Returns:
        ``(model, processor)`` tuple.
    """
    import torch
    import torch.nn as nn

    class _MockModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(10, 10)

        def forward(self, **kwargs: object) -> object:
            import types
            return types.SimpleNamespace(
                loss=torch.tensor(1.0),
                logits=torch.zeros(1, 5, 10),
            )

        def generate(self, input_ids: object = None, **kwargs: object) -> object:  # type: ignore[override]
            import torch as _t
            bs = input_ids.shape[0] if hasattr(input_ids, "shape") else 1  # type: ignore[union-attr]
            n = kwargs.get("max_new_tokens", 8)
            return _t.zeros(bs, (input_ids.shape[1] if hasattr(input_ids, "shape") else 1) + n,  # type: ignore[union-attr]
                            dtype=_t.long)

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
                '{"hazards": [], "scene_summary": "mock", '
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

    # Load and merge all configs
    cfg_dir = Path(args.config).parent
    model_cfg = load_config(cfg_dir / "model.yaml")
    data_cfg = load_config(cfg_dir / "data.yaml")
    training_cfg = load_config(cfg_dir / "training.yaml")
    eval_cfg = load_config(args.config)
    config = merge_configs(model_cfg, data_cfg, training_cfg, eval_cfg)

    if args.output_dir:
        config["output_dir"] = args.output_dir

    output_dir = Path(config.get("output_dir", "outputs/eval"))

    # Prediction generation
    predictions_path: Path
    if args.generate_predictions:
        predictions_path = generate_predictions(config, mock=args.mock)
    else:
        eval_data = config.get("eval_data", {})
        predictions_path = Path(
            args.predictions
            or eval_data.get(
                "predictions_path",
                "outputs/predictions/test_predictions.jsonl",
            )
        )

    ground_truth_path = Path(
        args.ground_truth
        or config.get("eval_data", {}).get(
            "ground_truth_path", "outputs/data/sft_ready/sft_test.jsonl"
        )
    )

    levels = set(args.level)
    all_metrics: dict[str, dict] = {}

    if "1" in levels:
        from drivesense.eval.grounding import GroundingEvaluator

        evaluator = GroundingEvaluator(config)
        metrics = evaluator.evaluate(predictions_path, ground_truth_path)
        evaluator.generate_report(metrics, output_dir / "level1")
        evaluator.log_to_wandb(metrics)
        all_metrics["level1"] = metrics
        logger.info(
            "Level 1 done — Recall=%.3f  Precision=%.3f  F1=%.3f",
            metrics.get("hazard_detection_rate", 0),
            metrics.get("precision", 0),
            metrics.get("f1_score", 0),
        )

    if "2" in levels:
        from drivesense.eval.reasoning import ReasoningEvaluator

        r_evaluator = ReasoningEvaluator(config, use_mock=args.mock_judge)
        r_metrics = r_evaluator.evaluate(predictions_path, ground_truth_path)
        r_evaluator.generate_report(r_metrics, output_dir / "level2")
        r_evaluator.log_to_wandb(r_metrics)
        all_metrics["level2"] = r_metrics
        logger.info(
            "Level 2 done — Overall=%.3f  PassRate=%.1f%%",
            r_metrics.get("overall_score", 0),
            r_metrics.get("pass_rate", 0) * 100,
        )

    # Combined summary
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        k: {kk: vv for kk, vv in v.items() if kk != "judge_results_raw"}
        for k, v in all_metrics.items()
    }
    summary_path = output_dir / "eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Summary written to %s", summary_path)

    print("\n--- Evaluation Complete ---")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
