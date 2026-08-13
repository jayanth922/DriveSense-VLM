# DriveSense-VLM Observability Layer

Three pure-Python, GPU-free tools that sit around `run_evaluation.py`: a CI
regression gate, a cross-run comparison report, and a data-drift monitor. None
of them need a model, a GPU, or live production traffic — they operate on
`eval_summary.json` files and stratification metadata the pipeline already
produces.

## 1. CI regression gate — `scripts/run_regression_gate.py`

**What it does:** compares a new model's `eval_summary.json` against a
baseline (the last known-good checkpoint) on seven Level-1 grounding metrics —
`hazard_detection_rate`, `detection_rate_by_iou` at 0.1/0.3/0.5,
`mean_best_pair_iou`, `parse_failure_rate`, `classification_accuracy` — and
**fails (exit 1)** if any metric regresses beyond a configurable relative
tolerance (default 10%). Each metric has a direction: "higher is better" for
detection/accuracy metrics, "lower is better" for `parse_failure_rate` (more
parse failures is always worse, even though the raw number goes up).

```bash
python scripts/run_regression_gate.py \
    --baseline outputs/eval/v2/eval_summary.json \
    --new      outputs/eval/v3/eval_summary.json

# Loosen/tighten one metric's tolerance (repeatable):
python scripts/run_regression_gate.py --baseline ... --new ... \
    --tolerance "detection_rate_by_iou@0.1=0.05"

# Save the machine-readable report for a CI artifact:
python scripts/run_regression_gate.py --baseline ... --new ... \
    --output outputs/eval/regression_report.json
```

Exit codes: `0` pass, `1` a metric regressed beyond tolerance, `2` bad input
(missing file, unknown `--tolerance` metric name).

**Plugging into CI:** run this as a required check right after
`run_evaluation.py --level 1 --generate-predictions` on every candidate
checkpoint, comparing against the currently-deployed model's stored
`eval_summary.json`. A failing exit code blocks promotion — this is exactly
the check that would have caught a v2→v3 regression automatically instead of
someone noticing it by eye.

## 2. Eval comparison report — `scripts/compare_eval_runs.py`

**What it does:** takes N labeled `eval_summary.json` files and renders one
table with every run's value per metric, plus a verdict (`↑ improved` /
`↓ regressed` / `→ flat`) versus the *previous* run in the sequence — not just
versus a single baseline. This is the artifact for "here's how the model
evolved," e.g. across `v1_stub → v2_3072frames → v3_9158frames`.

```bash
python scripts/compare_eval_runs.py \
    --run v1_stub=outputs/eval/v1/eval_summary.json \
    --run v2_3072frames=outputs/eval/v2/eval_summary.json \
    --run v3_9158frames=outputs/eval/v3/eval_summary.json

# Markdown, for a PR description or a docs artifact:
python scripts/compare_eval_runs.py --run a=... --run b=... --format markdown \
    > docs/eval_history.md
```

It shares its metric definitions with the regression gate
(`drivesense.eval.regression.DEFAULT_METRICS`) so the two tools never disagree
about what a metric means or which direction is "better" — but the verdict
here is **not tolerance-gated** (any real movement is labeled, not just
movement large enough to fail CI), because "did anything change since last
time" and "did it change enough to block a release" are genuinely different
questions.

## 3. Drift-detection scaffold — `src/drivesense/monitoring/drift.py`

**What it does:** `DriftMonitor` compares a REFERENCE categorical
distribution (e.g. the training set's `weather` / `time_of_day` / `location` /
`hazard_class` frequencies, all present in `sft_test_enriched.jsonl`) against
an INCOMING batch's distribution, per dimension, using the **Population
Stability Index (PSI)** — a standard, dependency-free drift metric (no scipy;
it's weighted log-ratios over category proportions). Interpretation follows
the usual industry bands: PSI `< 0.10` no significant change, `0.10–0.20`
moderate (worth watching), `> 0.20` significant shift.

```python
from drivesense.monitoring.drift import DriftMonitor

# Built ONCE, e.g. at model-promotion time, from the training set.
monitor = DriftMonitor.from_records(training_records,
                                    dimensions=["weather", "time_of_day", "location", "hazard_class"])

# Scored on a rolling basis against incoming production batches — no ground
# truth needed, only the same categorical metadata.
report = monitor.check(incoming_batch_records)
if DriftMonitor.any_drifted(report):
    alert(report)  # report[dim] = {psi, severity, drifted, reference_distribution, incoming_distribution}
```

**Demo — `scripts/demo_drift_monitor.py`:** runs standalone, no production
data required. It splits a label set in half, treats one half as the
reference and the other as "incoming," and shows two cases:

- **Case A** — incoming = the other (unmodified) half of the same split:
  the monitor correctly reports **no drift** on any dimension.
- **Case B** — incoming = that half with `weather` forced to `"rain"` for
  every record (a clearly-labeled *synthetic* override, since the demo's
  fallback data has no real rain samples): the monitor correctly flags
  **only** the `weather` dimension as drifted; the other three dimensions
  stay clean.

```bash
python scripts/demo_drift_monitor.py                       # synthetic data, always runnable
python scripts/demo_drift_monitor.py --labels outputs/data/sft_ready/sft_test_enriched.jsonl
```

**Plugging into production:** this is a scaffold, not a deployed monitor — the
missing piece for real use is a scheduled job that (a) builds the reference
once from the training set at model-promotion time, (b) pulls a window of
recent inference request metadata (weather/time/location tags, however your
serving layer captures them) as the incoming batch, (c) calls `.check()` on a
cadence (hourly/daily), and (d) routes `any_drifted(report) == True` to an
alert — the same shape as the regression gate, just triggered by incoming
data statistics instead of a new eval run. PSI needs no labels, so it works
even when ground truth for production traffic isn't available yet.

## Design notes

- All three tools are pure Python — no `torch`/`transformers`/GPU. They run
  in any CI environment, including one with no model artifacts at all beyond
  the JSON files.
- `src/drivesense/eval/regression.py` is the single source of truth for
  metric definitions/directions/tolerances, shared by the gate and the
  comparison report, so they can't silently drift apart from each other.
- Metric lookups use a dotted path (`"detection_rate_by_iou.0.1"`) that splits
  on the *first* `.` only — the JSON key `"0.1"` itself contains a literal
  dot, and a naive full split would shred it into `["0", "1"]` and silently
  report the metric as missing. (Caught by actually running the gate against
  real `eval_summary.json` fixtures before shipping — see `tests/test_regression.py`.)
