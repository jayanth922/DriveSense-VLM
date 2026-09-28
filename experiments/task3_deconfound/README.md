# Box-provenance experiment

This experiment compares foundation-model-emitted boxes with boxes projected from nuScenes ground
truth while holding the frames and training recipe fixed. Read
[`../../docs/TASK3_DECONFOUND.md`](../../docs/TASK3_DECONFOUND.md) for the result and limitations.

Start with [`RUNBOOK.md`](RUNBOOK.md). It includes a no-cost preflight and mock run before any API
or GPU stage.

| File | Purpose |
|---|---|
| `reconstruct.py` | rebuild manifests and estimate API cost |
| `describe_manifest.py` | add severity, explanation, and action to GT boxes through the Batch API |
| `build_arms.py` | assemble matched FM and GT arms and check split leakage |
| `compare_arms.py` | produce the FM-versus-GT result table |
| `training_h100.yaml` | single-H100 training settings |
| `model.yaml`, `data.yaml` | experiment-specific model and data paths |

The workflow reuses the repository's v4 labeler, prediction generator, and full evaluator rather
than maintaining separate implementations.
