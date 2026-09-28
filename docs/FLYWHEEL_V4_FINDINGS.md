# v4 flywheel findings

[Back to the project README](../README.md)

v4 was the first complete failure-driven training turn. It selected adverse-condition frames from
the v3 evaluation, labeled them, merged them without scene leakage, retrained the model, and
compared the candidate with v3 on the same test set.

The candidate did not improve the weak conditions. The regression gate blocked it, and v3 remained
the selected model.

## What changed in v4

1. The v3 robustness report identified rain, nighttime scenes, and tiny objects as weak conditions.
2. Mining selected 4,160 candidate frames from relevant nuScenes scenes.
3. Scene-token checks removed 986 frames that overlapped validation or test scenes.
4. The final adverse pool contained 2,231 frames. About 30% were actually clear or daytime,
   showing that the earlier metadata tags were broad rarity indicators rather than strict weather
   labels.
5. Claude Sonnet 5 labeled the pool through the Batch API. A 10-frame pilot exposed ordinary
   vehicles mislabeled as unusual objects, weather regions boxed as hazards, and edge-degenerate
   boxes. Prompt constraints and deterministic box repair were added before the full job.
6. The full labeling pass cost $10.98.
7. The final addition kept 1,226 positive frames and capped `no_hazard` examples at 216, producing
   1,442 new training records.
8. The merged set contained 8,670 train, 889 validation, and 1,041 test records.
9. Qwen2.5-VL-3B was trained for three epochs with LoRA and selected by validation loss.
10. Evaluation reused the fixed v3 test set and the same processor resolution as training.

## Training and evaluation results

### Validation loss

| Model | Training frames | Validation loss | Notes |
|---|---:|---:|---|
| v2 | 2,754 | 0.31 | different test set; best validation loss |
| v3 | 7,228 | 0.66 | five-epoch scale-up overfit |
| v4 | 8,670 | 0.694 | loss fell across three epochs |

### Grounding at IoU >= 0.5

| Metric | v3 | v4 | Change |
|---|---:|---:|---:|
| Precision | 0.40 | 0.37 | -0.03 |
| Recall | 0.24 | 0.19 | -0.05 |
| F1 | 0.30 | 0.25 | -0.05 |
| Mean matched IoU | 0.67 | 0.656 | -0.014 |
| Classification accuracy | 0.94 | 0.946 | +0.006 |
| Parse rate | 98.7% | 97.4% | -1.3 pp |

### Detection rate by condition

| Bucket | v3 | v4 | Change |
|---|---:|---:|---:|
| Overall | 28.0% | 23.0% | -5.0 pp |
| Tiny | 22.8% | 17.2% | -5.6 pp |
| Small | 46.4% | 42.9% | -3.5 pp |
| Medium | 52.6% | 52.6% | 0.0 pp |
| Rain | 12.5% | 7.4% | -5.1 pp |
| Night and tiny | 12.7% | 10.7% | -2.0 pp |
| Clear and medium | 69.0% | 69.0% | 0.0 pp |

The candidate regressed on all three conditions it was intended to improve. Stable mean IoU and
classification accuracy indicate that this was not a repeat of the coordinate-system bug.

## Why the result regressed

### New negative examples

v4 introduced 216 `no_hazard` records; v3 had none. This may have made the model more conservative
and contributed to the broad recall drop. The proposed v4b ablation removes those records while
keeping the positive addition.

### Different box source

The main v2/v3 labels use boxes projected from nuScenes 3D ground truth. The v4 addition instead
used `bbox_2d` values emitted by Claude and repaired only for basic geometric validity. This changed
the supervision convention on the hardest frames.

The later Task 3 experiment isolated this variable. With matched data and training settings, the
GT-box arm reached F1 0.222 compared with 0.128 for the FM-box arm. See
[`TASK3_DECONFOUND.md`](TASK3_DECONFOUND.md).

### Tiny-object limit

Tiny boxes make up 78% of test hazards. Their detection rate is much lower than the medium-object
rate in both v3 and v4. The current 3B model and image resolution appear to be a stronger constraint
than the number of additional adverse-condition frames.

## Gate decision

The protected buckets in `results/metrics_registry.json` are rain, night with tiny objects, and
tiny objects overall. All three became worse, so the gate returned `BLOCK`. v4 was not promoted.

This is the expected behavior of the pipeline: a targeted data addition is treated as a candidate,
not assumed to be an improvement.

## Limitations

- Rain and night subsets are smaller than the full test set.
- v4 changed both box source and class composition, so its first result alone could not attribute
  the regression to one variable.
- Task 3 used a reduced reconstruction and one seed. It established a clear directional result but
  not a full-scale confidence interval.

## Artifacts

The original training and prediction files lived on an ephemeral RunPod volume and are no longer
available. Their final metrics are versioned in
[`results/metrics_registry.json`](../results/metrics_registry.json), and compact v4 evaluation
artifacts are committed under [`results/v4/`](../results/v4/).

Historical volume paths are listed here only to identify the producing stage:

- `sft_train_ready_v4/sft_train.jsonl`: 1,442-record addition;
- `sft_train_ready_v4_merged/`: merged train, validation, and test sets;
- `v4_train_out/lora_adapter`: trained adapter;
- `v4_eval/`: predictions and stratified evaluation.

Current follow-up work is listed once in the
[`README`](../README.md#current-status-and-next-work).
