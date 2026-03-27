"""Entry point for ``python -m drivesense.eval``.

Usage:
    python -m drivesense.eval --config configs/eval.yaml --level 1
    python -m drivesense.eval --config configs/eval.yaml --level 2
    python -m drivesense.eval --config configs/eval.yaml --level all
    python -m drivesense.eval --config configs/eval.yaml --level all --mock-judge
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("drivesense.eval")

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DriveSense-VLM Phase 2b: Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="configs/eval.yaml", help="Path to eval.yaml")
    p.add_argument(
        "--level",
        default="all",
        choices=["1", "2", "all"],
        help="Evaluation level(s) to run",
    )
    p.add_argument(
        "--mock-judge",
        action="store_true",
        help="Use MockLLMJudge for Level 2 (no API key required)",
    )
    p.add_argument(
        "--predictions",
        default=None,
        help="Override predictions JSONL path from config",
    )
    p.add_argument(
        "--ground-truth",
        default=None,
        help="Override ground truth path from config",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory from config",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Level runners
# ---------------------------------------------------------------------------


def _run_level1(
    config: dict,
    predictions_path: Path,
    ground_truth_path: Path,
    output_dir: Path,
) -> dict:
    """Run Level 1 (grounding) evaluation."""
    from drivesense.eval.grounding import GroundingEvaluator

    evaluator = GroundingEvaluator(config)
    metrics = evaluator.evaluate(predictions_path, ground_truth_path)
    evaluator.generate_report(metrics, output_dir)
    evaluator.log_to_wandb(metrics)

    logger.info(
        "Level 1 — Recall=%.3f  Precision=%.3f  F1=%.3f  MeanIoU=%.3f",
        metrics.get("hazard_detection_rate", 0),
        metrics.get("precision", 0),
        metrics.get("f1_score", 0),
        metrics.get("mean_iou", 0),
    )
    return metrics


def _run_level2(
    config: dict,
    predictions_path: Path,
    ground_truth_path: Path,
    output_dir: Path,
    mock_judge: bool = False,
) -> dict:
    """Run Level 2 (reasoning) evaluation."""
    from drivesense.eval.reasoning import ReasoningEvaluator

    evaluator = ReasoningEvaluator(config, use_mock=mock_judge)
    metrics = evaluator.evaluate(predictions_path, ground_truth_path)
    evaluator.generate_report(metrics, output_dir)
    evaluator.log_to_wandb(metrics)

    logger.info(
        "Level 2 — Overall=%.3f  PassRate=%.1f%%  Judged=%d",
        metrics.get("overall_score", 0),
        metrics.get("pass_rate", 0) * 100,
        metrics.get("total_judged", 0),
    )
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse args, load config, and run the requested evaluation levels."""
    args = _parse_args()

    try:
        from drivesense.utils.config import load_config
    except ImportError as exc:
        logger.error("Cannot import drivesense: %s", exc)
        sys.exit(1)

    config = load_config(args.config)

    # Apply CLI overrides
    if args.output_dir:
        config["output_dir"] = args.output_dir

    output_dir = Path(config.get("output_dir", "outputs/eval"))

    # Resolve data paths
    eval_data = config.get("eval_data", {})
    predictions_path = Path(
        args.predictions
        or eval_data.get("predictions_path", "outputs/predictions/test_predictions.jsonl")
    )
    ground_truth_path = Path(
        args.ground_truth
        or eval_data.get("ground_truth_path", "outputs/data/sft_ready/sft_test.jsonl")
    )

    run_l1 = args.level in ("1", "all")
    run_l2 = args.level in ("2", "all")

    all_metrics: dict[str, dict] = {}

    if run_l1:
        all_metrics["level1"] = _run_level1(
            config, predictions_path, ground_truth_path, output_dir / "level1"
        )

    if run_l2:
        all_metrics["level2"] = _run_level2(
            config, predictions_path, ground_truth_path,
            output_dir / "level2", mock_judge=args.mock_judge
        )

    # Write combined summary
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "eval_summary.json"
    summary_path.write_text(
        json.dumps(
            {k: {kk: vv for kk, vv in v.items() if kk != "judge_results_raw"}
             for k, v in all_metrics.items()},
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Evaluation summary written to %s", summary_path)


if __name__ == "__main__":
    main()
