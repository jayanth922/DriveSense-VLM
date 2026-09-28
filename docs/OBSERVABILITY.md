# Evaluation and monitoring tools

[Back to the project README](../README.md)

The repository includes three small, CPU-only tools for tracking model quality. They operate on
saved evaluation JSON and dataset metadata; they do not load model weights.

## Regression gate

`scripts/run_regression_gate.py` compares a candidate `eval_summary.json` with a baseline. It
checks:

- hazard detection rate;
- detection rate at IoU 0.1, 0.3, and 0.5;
- mean best-pair IoU;
- parse failure rate;
- classification accuracy.

Metrics have a direction. Higher is better for detection and accuracy; lower is better for parse
failure rate. The default relative tolerance is 10%, and individual tolerances can be overridden.

```bash
python scripts/run_regression_gate.py \
  --baseline outputs/eval/baseline/eval_summary.json \
  --new outputs/eval/candidate/eval_summary.json

python scripts/run_regression_gate.py \
  --baseline outputs/eval/baseline/eval_summary.json \
  --new outputs/eval/candidate/eval_summary.json \
  --tolerance 'detection_rate_by_iou@0.1=0.05' \
  --output outputs/eval/regression_report.json
```

Exit codes:

- `0`: pass;
- `1`: regression beyond tolerance;
- `2`: invalid input.

The command is suitable for a required CI check after evaluation. The project also maintains a
policy-level v3/v4 gate in `scripts/mlops_report.py`, backed by
`results/metrics_registry.json`.

## Compare evaluation runs

`scripts/compare_eval_runs.py` prints the same metrics for any number of labeled runs. Each value
is marked improved, regressed, or unchanged relative to the previous run.

```bash
python scripts/compare_eval_runs.py \
  --run v2=outputs/eval/v2/eval_summary.json \
  --run v3=outputs/eval/v3/eval_summary.json \
  --run v4=outputs/eval/v4/eval_summary.json

python scripts/compare_eval_runs.py \
  --run v3=outputs/eval/v3/eval_summary.json \
  --run v4=outputs/eval/v4/eval_summary.json \
  --format markdown > outputs/eval/comparison.md
```

The report marks any numerical movement. The regression gate applies tolerances because its job is
to decide whether a change is large enough to block promotion. Both tools share metric definitions
from `src/drivesense/eval/regression.py`.

## Distribution drift

`DriftMonitor` in `src/drivesense/monitoring/drift.py` compares categorical distributions between
a reference set and an incoming batch. Supported dimensions include weather, time of day,
location, and hazard class.

It uses Population Stability Index (PSI):

- below 0.10: no material change;
- 0.10 to 0.20: moderate change;
- above 0.20: significant change.

```python
from drivesense.monitoring.drift import DriftMonitor

monitor = DriftMonitor.from_records(
    training_records,
    dimensions=["weather", "time_of_day", "location", "hazard_class"],
)
report = monitor.check(incoming_records)

if DriftMonitor.any_drifted(report):
    send_alert(report)
```

The demonstration script works with either synthetic data or an enriched SFT label file:

```bash
python scripts/demo_drift_monitor.py
python scripts/demo_drift_monitor.py \
  --labels outputs/data/sft_ready/sft_test_enriched.jsonl
```

The demo first compares two samples from the same distribution, then forces the incoming weather
field to `rain` and confirms that the weather dimension is flagged.

## Production integration

The drift monitor is a library scaffold, not a deployed service. A production integration would:

1. save a reference distribution when a model is promoted;
2. collect the same metadata from recent inference requests;
3. run `DriftMonitor.check()` on a schedule;
4. send significant PSI results to the alerting system.

No labels are required for PSI. Ground truth is still required for the regression gate because it
compares model quality rather than input distribution.

One implementation detail is covered by a regression test: paths such as
`detection_rate_by_iou.0.1` split only at the first dot because `0.1` is itself a JSON key. See
`tests/test_regression.py`.
