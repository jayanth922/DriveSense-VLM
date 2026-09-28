# Data flywheel

[Back to the project README](../README.md)

The flywheel turns evaluation failures into a new training candidate. Each stage writes an
artifact that can be inspected or reused, and the final regression gate decides whether the new
model is safe to promote.

```text
evaluate -> find weak conditions -> select frames -> mine images
   ^                                                |
   |                                                v
promote or block <- regression gate <- retrain <- label and validate
```

## Stages

| Stage | Command | Main output |
|---|---|---|
| Evaluate | `scripts/run_evaluation.py --level 1` | grounding metrics |
| Stratify failures | `scripts/analyze_failure_stratification.py` | weak size and condition buckets |
| Select targets | `scripts/select_mining_targets.py` | ranked shopping list |
| Mine images | `scripts/run_streaming_miner.py` or `scripts/v4/build_v4_manifest.py` | bounded-storage image set and manifest |
| Label | `scripts/regenerate_annotations_v2_colab.py` or `scripts/v4/v4_batch_label.py` | SFT records |
| Validate | `scripts/run_label_validation.py` | pass/fail report for schema and box quality |
| Train | `scripts/run_training.py` | LoRA adapter and training metrics |
| Re-evaluate | prediction, evaluation, and stratification scripts | candidate metrics |
| Gate | `scripts/run_regression_gate.py` | promote or block decision |

## Safeguards

### Split isolation

Frames are grouped by nuScenes scene token. A scene assigned to validation or test cannot enter
training through a later mining pass. The v4 build removed 986 candidate frames because they
shared a scene with the fixed evaluation sets.

### Label validation

The label gate checks schema, coordinate bounds, oversized boxes, box diversity, and repeated
coordinates. Training stops if the generated set looks collapsed or malformed.

### Checkpoint selection

Training selects the checkpoint with the best validation loss. Checkpoint retention is configured
so that the best epoch cannot be deleted before `load_best_model_at_end` runs.

### Candidate regression gate

The gate compares a candidate with the selected model on the conditions that already perform
poorly. A candidate is blocked if any protected metric falls beyond its tolerance. This happened
to v4, so v3 remained selected.

## What happened in the v4 turn

v4 added 1,442 rain and nighttime examples. Performance nevertheless fell on rain, night with
tiny objects, and tiny objects overall. The gate blocked the candidate.

A follow-up experiment showed that box provenance contributed substantially to the regression:
the added v4 labels used foundation-model-emitted boxes, while the main v2/v3 set used boxes
projected from nuScenes 3D ground truth. See
[`TASK3_DECONFOUND.md`](TASK3_DECONFOUND.md) and
[`FLYWHEEL_V4_FINDINGS.md`](FLYWHEEL_V4_FINDINGS.md).

## Automation boundary

The stages are idempotent and communicate through JSON or JSONL artifacts. They could be wrapped
in Airflow, Prefect, or a Make target, but the repository keeps them as separate commands so that
costly API and GPU stages remain explicit. CI currently runs the test suite, smoke checks, and the
regression gate in [`.github/workflows/ci.yml`](../.github/workflows/ci.yml).
