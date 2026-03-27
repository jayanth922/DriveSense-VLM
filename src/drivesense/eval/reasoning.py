"""Level 2: Reasoning Quality Evaluation.

Uses LLM-as-judge (Claude) to evaluate the quality of hazard reasoning,
action recommendations, and scene understanding on a 1–5 Likert scale.

Phase 2b implementation.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependencies — guarded for macOS CPU dev
# ---------------------------------------------------------------------------

try:
    import anthropic as _anthropic_lib  # type: ignore[import]
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _anthropic_lib = None  # type: ignore[assignment]
    _ANTHROPIC_AVAILABLE = False

try:
    import wandb  # type: ignore[import]
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None  # type: ignore[assignment]
    _WANDB_AVAILABLE = False

from drivesense.data.annotation import AnnotationValidator  # noqa: E402, I001


# ---------------------------------------------------------------------------
# LLMJudge
# ---------------------------------------------------------------------------


class LLMJudge:
    """LLM-as-judge for evaluating reasoning quality.

    Uses Claude to score model outputs on correctness, completeness, and
    action relevance using a 1–5 scale.

    Args:
        config:  Eval config dict (from ``configs/eval.yaml``).
        api_key: Anthropic API key.  Falls back to ``ANTHROPIC_API_KEY`` env var.
    """

    JUDGE_SYSTEM_PROMPT = (
        "You are an expert evaluator for autonomous vehicle perception systems. "
        "You will be given the ground truth hazard annotation and a model's predicted "
        "annotation. Score the model's prediction on the specified dimension using a "
        "1-5 scale.\n\n"
        "Scoring criteria:\n"
        "1 = Completely wrong or irrelevant\n"
        "2 = Major errors, partially relevant\n"
        "3 = Acceptable, some errors or omissions\n"
        "4 = Good, minor issues only\n"
        "5 = Excellent, accurate and comprehensive\n\n"
        "Respond with ONLY a JSON object: "
        '{"score": <int>, "justification": "<brief explanation>"}'
    )

    JUDGE_DIMENSIONS: dict[str, str] = {
        "correctness": (
            "How accurately does the model's reasoning explain the actual hazard? "
            "Does it correctly identify why the situation is dangerous?"
        ),
        "completeness": (
            "Does the reasoning cover all relevant risk factors? "
            "Are there important safety considerations missing?"
        ),
        "action_relevance": (
            "Is the recommended driving action appropriate and specific to the "
            "identified hazard? Would following this action improve safety?"
        ),
    }

    def __init__(self, config: dict, api_key: str | None = None) -> None:
        if not _ANTHROPIC_AVAILABLE or _anthropic_lib is None:
            raise ImportError(
                "anthropic package required for LLMJudge. "
                "Install with: pip install anthropic"
            )
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. "
                "Export it or pass api_key= to LLMJudge."
            )
        judge_cfg = config.get("reasoning", {}).get("judge", {})
        self._model: str = judge_cfg.get("model", "claude-sonnet-4-6")
        self._temperature: float = float(judge_cfg.get("temperature", 0.1))
        self._client = _anthropic_lib.Anthropic(api_key=key)

    def judge_single(
        self,
        prediction: dict,
        ground_truth: dict,
        dimension: str,
        image_path: str | None = None,  # noqa: ARG002
    ) -> dict:
        """Score a single prediction on one dimension.

        Args:
            prediction:  Model's predicted annotation dict.
            ground_truth: GT annotation dict.
            dimension:   One of ``correctness``, ``completeness``,
                         ``action_relevance``.
            image_path:  Optional path to image (reserved for future use).

        Returns:
            Dict with ``score``, ``justification``, and ``dimension``.
        """
        dim_description = self.JUDGE_DIMENSIONS.get(
            dimension,
            f"Evaluate the model on: {dimension}",
        )
        user_prompt = _build_judge_prompt(prediction, ground_truth, dimension, dim_description)
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=256,
                temperature=self._temperature,
                system=self.JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw = response.content[0].text
            parsed = _parse_judge_response(raw)
            return {
                "score": parsed.get("score", 1),
                "justification": parsed.get("justification", ""),
                "dimension": dimension,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Judge call failed for dimension=%s: %s", dimension, exc)
            return {"score": 1, "justification": f"<error: {exc}>", "dimension": dimension}

    def judge_batch(
        self,
        predictions: list[dict],
        ground_truth: list[dict],
        dimensions: list[str] | None = None,
        max_concurrent: int = 5,
    ) -> list[dict]:
        """Score a batch of predictions across all dimensions.

        Uses a thread pool for concurrent API calls.

        Args:
            predictions:   List of prediction dicts.
            ground_truth:  List of GT dicts (parallel to predictions).
            dimensions:    Dimensions to judge.  Defaults to all three.
            max_concurrent: Max parallel API calls.

        Returns:
            List of result dicts, each with ``frame_id`` and ``scores`` map.
        """
        dims = dimensions or list(self.JUDGE_DIMENSIONS.keys())
        results: list[dict] = [
            {"frame_id": p.get("frame_id"), "scores": {}} for p in predictions
        ]

        def _score_one(idx: int, dim: str) -> tuple[int, str, dict]:
            score = self.judge_single(predictions[idx], ground_truth[idx], dim)
            return idx, dim, score

        with ThreadPoolExecutor(max_workers=max_concurrent) as exe:
            futures = {
                exe.submit(_score_one, i, d): (i, d)
                for i in range(len(predictions))
                for d in dims
            }
            for fut in as_completed(futures):
                with contextlib.suppress(Exception):
                    idx, dim, score = fut.result()
                    results[idx]["scores"][dim] = score

        return results


# ---------------------------------------------------------------------------
# MockLLMJudge
# ---------------------------------------------------------------------------


class MockLLMJudge:
    """Mock judge for testing.  Returns plausible random scores in [3, 5]."""

    JUDGE_DIMENSIONS = LLMJudge.JUDGE_DIMENSIONS

    def judge_single(
        self,
        prediction: dict,  # noqa: ARG002
        ground_truth: dict,  # noqa: ARG002
        dimension: str,
        image_path: str | None = None,  # noqa: ARG002
    ) -> dict:
        """Return a mock score for one dimension.

        Args:
            prediction:  Unused (mock).
            ground_truth: Unused (mock).
            dimension:   Dimension name.
            image_path:  Unused (mock).

        Returns:
            Dict with score in [3, 5], mock justification, and dimension.
        """
        return {
            "score": random.randint(3, 5),
            "justification": "Mock evaluation.",
            "dimension": dimension,
        }

    def judge_batch(
        self,
        predictions: list[dict],
        ground_truth: list[dict],
        dimensions: list[str] | None = None,
        max_concurrent: int = 5,  # noqa: ARG002
    ) -> list[dict]:
        """Return mock scores for all prediction–GT pairs.

        Args:
            predictions:   List of prediction dicts.
            ground_truth:  List of GT dicts.
            dimensions:    Dimensions to score.  Defaults to all three.
            max_concurrent: Ignored.

        Returns:
            List of result dicts with mock scores.
        """
        dims = dimensions or list(LLMJudge.JUDGE_DIMENSIONS.keys())
        results = []
        for pred, gt in zip(predictions, ground_truth):  # noqa: B905
            scores = {
                dim: self.judge_single(pred, gt, dim)
                for dim in dims
            }
            results.append({"frame_id": pred.get("frame_id"), "scores": scores})
        return results


# ---------------------------------------------------------------------------
# compute_reasoning_metrics
# ---------------------------------------------------------------------------


def compute_reasoning_metrics(judge_results: list[dict]) -> dict:
    """Aggregate LLM judge scores into summary metrics.

    Args:
        judge_results: List of result dicts returned by
                       :meth:`LLMJudge.judge_batch`.

    Returns:
        Full metrics dict with per-dimension stats, overall score,
        pass rate, and failure counts.
    """
    dims = ["correctness", "completeness", "action_relevance"]
    agg: dict[str, list[int]] = {d: [] for d in dims}
    judge_failures = 0

    for result in judge_results:
        scores = result.get("scores", {})
        if not scores:
            judge_failures += 1
            continue
        for dim in dims:
            dim_result = scores.get(dim, {})
            score = dim_result.get("score") if isinstance(dim_result, dict) else None
            if score is not None:
                with contextlib.suppress(TypeError, ValueError):
                    agg[dim].append(int(score))

    def _stats(values: list[int]) -> dict:
        if not values:
            return {
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "distribution": dict.fromkeys(range(1, 6), 0),
            }
        n = len(values)
        mean = sum(values) / n
        s_vals = sorted(values)
        median = float(
            s_vals[n // 2]
            if n % 2 == 1
            else (s_vals[n // 2 - 1] + s_vals[n // 2]) / 2
        )
        std = (sum((v - mean) ** 2 for v in values) / n) ** 0.5
        dist = {i: values.count(i) for i in range(1, 6)}
        return {
            "mean": round(mean, 4),
            "median": median,
            "std": round(std, 4),
            "distribution": dist,
        }

    dim_metrics = {dim: _stats(agg[dim]) for dim in dims}

    all_scores = [s for vals in agg.values() for s in vals]
    overall = sum(all_scores) / len(all_scores) if all_scores else 0.0

    # pass_rate: fraction with ALL dimensions >= 3.5
    total_judged = sum(1 for r in judge_results if r.get("scores"))
    pass_count = 0
    for result in judge_results:
        scores = result.get("scores", {})
        if not scores:
            continue
        dim_scores = [
            scores.get(dim, {}).get("score", 0) or 0
            for dim in dims
            if dim in scores
        ]
        if dim_scores and all(s >= 3.5 for s in dim_scores):
            pass_count += 1

    pass_rate = pass_count / total_judged if total_judged > 0 else 0.0

    return {
        "correctness": dim_metrics["correctness"],
        "completeness": dim_metrics["completeness"],
        "action_relevance": dim_metrics["action_relevance"],
        "overall_score": round(overall, 4),
        "pass_rate": round(pass_rate, 4),
        "total_judged": total_judged,
        "judge_failures": judge_failures,
    }


# ---------------------------------------------------------------------------
# ReasoningEvaluator
# ---------------------------------------------------------------------------


class ReasoningEvaluator:
    """Complete Level 2 evaluation pipeline.

    Args:
        config:   Merged eval config dict (from ``configs/eval.yaml``).
        use_mock: If ``True`` (or no API key found), use :class:`MockLLMJudge`.
    """

    def __init__(self, config: dict, use_mock: bool = False) -> None:
        self._cfg = config
        self._output_dir = Path(config.get("output_dir", "outputs/eval"))
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if use_mock or not api_key or not _ANTHROPIC_AVAILABLE:
            self._judge: LLMJudge | MockLLMJudge = MockLLMJudge()
            if not use_mock:
                logger.info(
                    "ANTHROPIC_API_KEY not set — using MockLLMJudge for Level 2 eval"
                )
        else:
            self._judge = LLMJudge(config, api_key=api_key)
        judge_cfg = config.get("reasoning", {}).get("judge", {})
        self._dimensions: list[str] = judge_cfg.get(
            "dimensions", ["correctness", "completeness", "action_relevance"]
        )
        self._max_concurrent: int = int(judge_cfg.get("max_concurrent", 5))

    def evaluate(
        self,
        predictions_path: Path,
        ground_truth_path: Path,
    ) -> dict:
        """Run complete Level 2 evaluation with LLM judge.

        Args:
            predictions_path:  Path to predictions JSONL.
            ground_truth_path: Path to GT manifest / SFT JSONL.

        Returns:
            Full reasoning metrics dict including raw judge results.
        """
        from drivesense.eval.grounding import GroundingEvaluator

        evaluator = GroundingEvaluator(self._cfg)
        preds = evaluator.load_predictions(predictions_path)
        gt_list = evaluator.load_ground_truth(ground_truth_path)

        gt_by_frame = {g["frame_id"]: g for g in gt_list if "frame_id" in g}
        matched_preds: list[dict] = []
        matched_gt: list[dict] = []
        for pred in preds:
            if pred.get("parse_failure", False):
                continue
            gt = gt_by_frame.get(pred.get("frame_id", ""))
            if gt:
                matched_preds.append(pred)
                matched_gt.append(gt)

        logger.info(
            "Level 2 eval: judging %d matched prediction–GT pairs", len(matched_preds)
        )

        judge_results = self._judge.judge_batch(
            matched_preds,
            matched_gt,
            dimensions=self._dimensions,
            max_concurrent=self._max_concurrent,
        )

        metrics = compute_reasoning_metrics(judge_results)
        metrics["judge_results_raw"] = judge_results
        return metrics

    def generate_report(self, metrics: dict, output_dir: Path) -> Path:
        """Write Level 2 artefacts to *output_dir*.

        Creates::

            output_dir/
            ├── reasoning_metrics.json
            ├── reasoning_report.txt
            ├── per_dimension_scores.json
            └── judge_results_raw.json

        Args:
            metrics:    Full metrics dict returned by :meth:`evaluate`.
            output_dir: Directory to write artefacts into.

        Returns:
            Path to the plain-text report file.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        scalar = {
            k: v
            for k, v in metrics.items()
            if k not in ("correctness", "completeness", "action_relevance",
                         "judge_results_raw")
        }
        (out / "reasoning_metrics.json").write_text(
            json.dumps(scalar, indent=2), encoding="utf-8"
        )

        per_dim = {
            dim: metrics.get(dim, {})
            for dim in ("correctness", "completeness", "action_relevance")
        }
        (out / "per_dimension_scores.json").write_text(
            json.dumps(per_dim, indent=2), encoding="utf-8"
        )

        raw = metrics.get("judge_results_raw", [])
        (out / "judge_results_raw.json").write_text(
            json.dumps(raw, indent=2), encoding="utf-8"
        )

        report_path = out / "reasoning_report.txt"
        report_path.write_text(_format_reasoning_report(metrics), encoding="utf-8")
        logger.info("Level 2 report written to %s", report_path)
        return report_path

    def log_to_wandb(self, metrics: dict) -> None:
        """Log reasoning metrics to W&B.

        Args:
            metrics: Full metrics dict returned by :meth:`evaluate`.
        """
        if not _WANDB_AVAILABLE or wandb is None:
            return

        dims = ("correctness", "completeness", "action_relevance")
        scalars: dict[str, Any] = {
            "eval/reasoning_overall": metrics.get("overall_score", 0),
            "eval/reasoning_pass_rate": metrics.get("pass_rate", 0),
        }
        for dim in dims:
            mean = metrics.get(dim, {}).get("mean", 0)
            scalars[f"eval/reasoning_{dim}"] = mean
        with contextlib.suppress(Exception):
            wandb.log(scalars)

        # Score distribution table
        for dim in dims:
            dist = metrics.get(dim, {}).get("distribution", {})
            if dist:
                rows = [[score, count] for score, count in dist.items()]
                with contextlib.suppress(Exception):
                    wandb.log({
                        f"eval/reasoning_{dim}_dist": wandb.Table(
                            columns=["Score", "Count"], data=rows
                        )
                    })


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_judge_prompt(
    prediction: dict,
    ground_truth: dict,
    dimension: str,
    dim_description: str,
) -> str:
    """Build the user-turn judge prompt."""
    gt_str = json.dumps(
        {k: v for k, v in ground_truth.items() if k != "frame_id"},
        indent=2,
    )
    pred_str = json.dumps(
        {k: v for k, v in prediction.items()
         if k not in ("frame_id", "parse_failure", "ego_context")},
        indent=2,
    )
    return (
        f"Evaluate dimension: **{dimension}**\n"
        f"Criteria: {dim_description}\n\n"
        f"--- GROUND TRUTH ---\n{gt_str}\n\n"
        f"--- MODEL PREDICTION ---\n{pred_str}\n\n"
        f"Score the prediction on '{dimension}' (1–5):"
    )


def _parse_judge_response(raw: str) -> dict:
    """Extract score and justification from a judge response string."""
    parsed = AnnotationValidator.parse_llm_response(raw)
    if parsed and "score" in parsed:
        return parsed
    # Fallback: scan for a digit
    import re
    m = re.search(r'"score"\s*:\s*([1-5])', raw)
    if m:
        return {"score": int(m.group(1)), "justification": raw}
    return {"score": 1, "justification": f"<parse error: {raw[:80]}>"}


def _format_reasoning_report(metrics: dict) -> str:
    """Format a human-readable Level 2 evaluation report."""
    dims = ("correctness", "completeness", "action_relevance")
    lines = [
        "=" * 62,
        "  DriveSense-VLM — Level 2 Reasoning Evaluation Report",
        "=" * 62,
        "",
        f"  Overall Score      : {metrics.get('overall_score', 0):.4f} / 5.0",
        f"  Pass Rate (≥3.5)   : {metrics.get('pass_rate', 0):.1%}",
        f"  Total Judged       : {metrics.get('total_judged', 0)}",
        f"  Judge Failures     : {metrics.get('judge_failures', 0)}",
        "",
        "  Per-Dimension Scores",
        "  " + "-" * 40,
    ]
    for dim in dims:
        d = metrics.get(dim, {})
        lines.append(
            f"  {dim:<22}  mean={d.get('mean', 0):.3f}  "
            f"median={d.get('median', 0):.1f}  std={d.get('std', 0):.3f}"
        )
        dist = d.get("distribution", {})
        if dist:
            bar = "  " + " ".join(
                f"{score}:{'█' * count}" for score, count in sorted(dist.items())
            )
            lines.append(bar)
    lines += ["", "=" * 62]
    return "\n".join(lines)
