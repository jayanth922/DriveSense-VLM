#!/usr/bin/env python3
"""Side-by-side eval comparison across N runs — "here's how the model evolved."

Takes multiple labeled ``eval_summary.json`` files (e.g. from different
training runs/checkpoints) and produces a comparison table of Level-1
grounding metrics, with a one-line verdict per metric versus the PREVIOUS
run in the sequence (not just versus a single baseline).

Usage:
    python scripts/compare_eval_runs.py \
        --run v1_stub=outputs/eval/v1/eval_summary.json \
        --run v2_3072frames=outputs/eval/v2/eval_summary.json \
        --run v3_9158frames=outputs/eval/v3/eval_summary.json

    # Markdown (for a PR description / docs artifact):
    python scripts/compare_eval_runs.py --run a=... --run b=... --format markdown \
        > docs/eval_history.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from drivesense.eval.regression import (  # noqa: E402
    DEFAULT_METRICS,
    build_comparison_rows,
    load_eval_summary,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Compare Level-1 eval metrics across N labeled runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--run", action="append", required=True, metavar="LABEL=PATH",
                   help="A labeled eval_summary.json (repeatable; order = comparison order).")
    p.add_argument("--format", choices=["console", "markdown"], default="console")
    return p.parse_args()


def load_runs(specs: list[str]) -> list[tuple[str, dict]]:
    """Parse ``LABEL=PATH`` specs and load each eval summary."""
    runs: list[tuple[str, dict]] = []
    for spec in specs:
        label, sep, path = spec.partition("=")
        if not sep:
            print(f"ERROR: --run must be LABEL=PATH, got {spec!r}", file=sys.stderr)
            sys.exit(2)
        p = Path(path)
        if not p.exists():
            print(f"ERROR: eval summary not found for {label!r}: {p}", file=sys.stderr)
            sys.exit(2)
        runs.append((label, load_eval_summary(p)))
    return runs


def _fmt(v: float | None) -> str:
    return "—" if v is None else f"{v:.4f}"


def _cell_text(cell: dict) -> str:
    value = _fmt(cell["value"])
    return value if cell["verdict"] == "n/a" else f"{value} {cell['verdict']}"


def render_console(rows: list[dict], labels: list[str]) -> str:
    """Render the comparison as a fixed-width console table.

    Column width must fit the widest CELL CONTENT (value + verdict, e.g.
    ``"0.3300 ↓ regressed"``), not just the label — labels are usually much
    shorter than a value+verdict cell, which previously made columns collide.
    """
    widest_cell = max(
        (len(_cell_text(c)) for row in rows for c in row["cells"]), default=0)
    col = max(10, max(len(lbl) for lbl in labels), widest_cell) + 2
    total_width = 28 + col * len(labels)
    lines = ["=" * total_width, "  EVAL COMPARISON", "=" * total_width]
    header = f"  {'metric':<26}" + "".join(f"{lbl:>{col}}" for lbl in labels)
    lines.append(header)
    lines.append("-" * total_width)
    for row in rows:
        cells = "".join(f"{_cell_text(c):>{col}}" for c in row["cells"])
        lines.append(f"  {row['metric']:<26}{cells}")
    lines.append("=" * (28 + col * len(labels)))
    return "\n".join(lines)


def render_markdown(rows: list[dict], labels: list[str]) -> str:
    """Render the comparison as a GitHub-flavoured markdown table."""
    lines = [
        "# Eval comparison\n",
        "| Metric | " + " | ".join(labels) + " |",
        "|---" * (len(labels) + 1) + "|",
    ]
    for row in rows:
        cells = [
            _fmt(c["value"]) + (f" ({c['verdict']})" if c["verdict"] != "n/a" else "")
            for c in row["cells"]
        ]
        lines.append(f"| {row['metric']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    runs = load_runs(args.run)
    labels = [label for label, _ in runs]
    rows = build_comparison_rows(runs, DEFAULT_METRICS)
    text = (render_markdown(rows, labels) if args.format == "markdown"
           else render_console(rows, labels))
    print(text)


if __name__ == "__main__":
    main()
