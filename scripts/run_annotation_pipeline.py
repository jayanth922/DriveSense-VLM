#!/usr/bin/env python3
"""Phase 1c: Run LLM annotation pipeline on unified dataset.

Usage:
    # Full pipeline (requires ANTHROPIC_API_KEY)
    python scripts/run_annotation_pipeline.py

    # Dry run — validate prompts, print samples, no API calls
    python scripts/run_annotation_pipeline.py --dry-run

    # Use mock LLM (no API key needed — for testing)
    python scripts/run_annotation_pipeline.py --mock-llm

    # Annotate a specific split only
    python scripts/run_annotation_pipeline.py --split train

    # Custom counterfactual ratio
    python scripts/run_annotation_pipeline.py --counterfactual-ratio 0.3

    # Resume from checkpoint (already-cached frames are skipped)
    python scripts/run_annotation_pipeline.py --resume

    # Format only — convert existing annotated manifest to SFT format
    python scripts/run_annotation_pipeline.py --format-only

    # Combine flags
    python scripts/run_annotation_pipeline.py --mock-llm --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from drivesense.data.annotation import (  # noqa: E402
    AnnotationPromptBuilder,
    LLMAnnotationPipeline,
    MockLLMClient,
    SFTDataFormatter,
    _load_manifest,
)
from drivesense.utils.config import load_config  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_annotation_pipeline")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Phase 1c: LLM annotation pipeline for DriveSense unified dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", default="configs/data.yaml", help="Path to data config YAML")
    p.add_argument(
        "--manifest",
        default=None,
        help="Path to unified manifest JSONL (default: from config unified.output_dir)",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for annotated manifests (default: from config annotation.output_dir)",  # noqa: E501
    )
    p.add_argument(
        "--sft-output-dir",
        default=None,
        help="Output directory for SFT-formatted JSONL files",
    )
    p.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="all",
        help="Annotate only this split (default: all)",
    )
    p.add_argument(
        "--counterfactual-ratio",
        type=float,
        default=None,
        help="Fraction of frames to augment with counterfactuals (overrides config)",
    )
    p.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Max concurrent API requests (overrides config annotation.max_concurrent_requests)",
    )
    p.add_argument("--dry-run", action="store_true", help="Build prompts for 3 frames, print, exit")
    p.add_argument("--mock-llm", action="store_true", help="Use MockLLMClient (no API key needed)")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-cached frames (caching is always on; this flag is informational)",
    )
    p.add_argument(
        "--format-only",
        action="store_true",
        help="Skip annotation; only convert existing annotated manifest to SFT format",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dry-run helpers
# ---------------------------------------------------------------------------

def run_dry_run(frames: list[dict], config: dict) -> None:
    """Print sample prompts and mock annotations for 3 frames, then exit."""
    builder = AnnotationPromptBuilder()
    mock = MockLLMClient()
    samples = frames[:3]
    for i, frame in enumerate(samples, 1):
        source = frame.get("source", "unknown")
        fid = frame.get("frame_id", f"frame_{i}")
        print(f"\n{'='*70}")
        print(f"  SAMPLE {i}: {fid}  (source={source})")
        print("="*70)

        system, user = builder.build_annotation_prompt(frame)
        print("\n--- SYSTEM PROMPT ---")
        print(system[:500] + ("..." if len(system) > 500 else ""))
        print("\n--- USER PROMPT ---")
        print(user[:800] + ("..." if len(user) > 800 else ""))

        raw = mock.annotate(frame, system, user)
        annotation = json.loads(raw)
        print("\n--- MOCK ANNOTATION ---")
        print(json.dumps(annotation, indent=2))

        # Also show a counterfactual prompt
        cf_system, cf_user, cf_meta = builder.build_counterfactual_prompt(frame)
        print(f"\n--- COUNTERFACTUAL PROMPT (scenario: {cf_meta['scenario_label']}) ---")
        print(cf_user[:600] + ("..." if len(cf_user) > 600 else ""))

    print(f"\n{'='*70}")
    print(f"Dry run complete — {len(samples)} sample frames shown.")
    print("No API calls were made. Remove --dry-run to run the full pipeline.")


# ---------------------------------------------------------------------------
# Format-only helper
# ---------------------------------------------------------------------------

def run_format_only(config: dict, args: argparse.Namespace) -> None:
    """Convert an existing annotated manifest to SFT JSONL format."""
    ann_cfg = config.get("annotation", {})
    output_dir = Path(args.output_dir or ann_cfg.get("output_dir", "outputs/data/annotated"))
    sft_output_dir = Path(
        args.sft_output_dir or ann_cfg.get("sft_output_dir", "outputs/data/sft_ready")
    )
    manifest_path = output_dir / "annotated_manifest.json"
    if not manifest_path.exists():
        logger.error("Annotated manifest not found at %s — run pipeline first.", manifest_path)
        sys.exit(1)

    formatter = SFTDataFormatter(config)
    out = formatter.format_dataset(manifest_path, sft_output_dir)
    logger.info("SFT data written to %s", out)
    _print_sft_stats(sft_output_dir)


def _print_sft_stats(sft_dir: Path) -> None:
    stats_path = sft_dir / "sft_format_stats.json"
    if stats_path.exists():
        with stats_path.open() as fh:
            stats = json.load(fh)
        print("\n--- SFT Dataset Statistics ---")
        for k, v in stats.items():
            print(f"  {k}: {v}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point."""
    args = parse_args()
    config = load_config(args.config)
    ann_cfg = config.get("annotation", {})
    unified_cfg = config.get("unified", {})

    # Resolve manifest path
    manifest_path_str = args.manifest or (
        str(Path(unified_cfg.get("output_dir", "outputs/data/unified")) / "manifest_all.jsonl")
    )
    manifest_path = Path(manifest_path_str)

    if args.format_only:
        run_format_only(config, args)
        return

    if not manifest_path.exists():
        logger.error("Manifest not found: %s", manifest_path)
        logger.info(
            "Run 'python scripts/run_build_unified_dataset.py' first "  # noqa: E501
            "to create the unified manifest."
        )
        sys.exit(1)

    frames = _load_manifest(manifest_path)
    logger.info("Loaded %d frames from %s", len(frames), manifest_path)

    # Filter by split if requested
    if args.split != "all":
        frames = [f for f in frames if f.get("split") == args.split]
        logger.info("Filtered to %d frames in split '%s'", len(frames), args.split)

    if args.dry_run:
        run_dry_run(frames, config)
        return

    # Build pipeline
    mock_client = MockLLMClient() if args.mock_llm else None
    pipeline = LLMAnnotationPipeline(config, mock_client=mock_client)

    output_dir = Path(args.output_dir or ann_cfg.get("output_dir", "outputs/data/annotated"))
    max_concurrent = args.max_concurrent or int(ann_cfg.get("max_concurrent_requests", 5))
    cf_ratio = args.counterfactual_ratio

    logger.info(
        "Starting annotation: %d frames, mock=%s, max_concurrent=%d",
        len(frames), args.mock_llm, max_concurrent,
    )
    result_dir = pipeline.run_full_pipeline(
        manifest_path=manifest_path,
        output_dir=output_dir,
        counterfactual_ratio=cf_ratio,
    )
    logger.info("Annotation complete → %s", result_dir)

    # Print quality report
    report_path = result_dir / "quality_report.json"
    if report_path.exists():
        with report_path.open() as fh:
            report = json.load(fh)
        print("\n--- Quality Report ---")
        for k, v in report.items():
            print(f"  {k}: {v}")

    # Format to SFT
    annotated_manifest = result_dir / "annotated_manifest.json"
    if annotated_manifest.exists():
        sft_dir = Path(  # noqa: E501
            args.sft_output_dir or ann_cfg.get("sft_output_dir", "outputs/data/sft_ready")
        )
        formatter = SFTDataFormatter(config)
        formatter.format_dataset(annotated_manifest, sft_dir)
        logger.info("SFT data written to %s", sft_dir)
        _print_sft_stats(sft_dir)


if __name__ == "__main__":
    main()
