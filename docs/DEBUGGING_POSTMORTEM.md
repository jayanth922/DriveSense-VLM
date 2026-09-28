# Debugging postmortem

[Back to the project README](../README.md)

This document records three failures that changed the project. The first was an evaluation bug;
the other two were model and data regressions. All v3/v4 comparisons use the fixed 1,041-frame
test set.

## 1. IoU collapsed because training and inference used different image settings

**Observed behavior:** every class had an IoU close to zero. The initial result looked like a
complete localization failure.

**Cause:** prediction generation loaded the base processor without the training pixel limits.
Inference therefore used Qwen's larger default image representation instead of the resolution
used during fine-tuning. The model started emitting coordinates outside the labels' normalized
0-1000 range; 17% of sampled coordinates were above 1000.

**Fix:** generation now applies the same processor limits as training: `min_pixels=200704` and
`max_pixels=602112`. After the fix, none of 64 sampled coordinates exceeded 1000. The corrected
v3 result was precision 0.40, recall 0.24, F1 0.30, and mean matched IoU 0.67.

The evaluator also gained an all-zero-IoU check. If predictions and labels both contain boxes but
every overlap is exactly zero, evaluation stops instead of publishing a misleading score.

**Takeaway:** image preprocessing and coordinate conventions are part of the model interface. They
must match between training and inference.

## 2. The first data scale-up reduced generalization

**Observed behavior:** increasing the training set from 2,754 to 7,228 examples raised validation
loss from 0.31 to 0.66 while training loss continued to fall.

**Cause:** the larger dataset was not uniformly better, and five epochs overfit it. Two pipeline
issues made the result harder to diagnose:

- `save_total_limit` could delete the best checkpoint before it was restored;
- the reasoning judge returned a score of 1.0 after API errors, hiding failed judge calls.

**Fixes:** training uses early stopping and fewer epochs where appropriate, checkpoint retention
keeps every candidate best epoch, and judge errors now return `None` rather than a fabricated
score.

**Takeaway:** dataset size and training loss are not enough to judge a training run. Validation
loss and a fixed evaluation set must decide whether a new checkpoint is better.

## 3. Targeted adverse-weather data did not improve v4

**Hypothesis:** because rain, nighttime scenes, and tiny objects were the weakest buckets, adding
more examples from those conditions should improve them.

**Result:** v4 regressed on all three protected buckets:

| Bucket | v3 | v4 |
|---|---:|---:|
| Rain | 12.5% | 7.4% |
| Night and tiny | 12.7% | 10.7% |
| Tiny | 22.8% | 17.2% |

Overall recall fell from 0.24 to 0.19. Mean matched IoU stayed near 0.66, so the change was not
another coordinate failure.

Several variables changed in v4. It introduced 216 `no_hazard` negatives, which may have made the
model more conservative. More importantly, its targeted labels used foundation-model-emitted
boxes, unlike the GT-projected boxes in the main training set. The controlled Task 3 experiment
later confirmed that this provenance change materially reduced localization performance.

Tiny objects also remain a model-side limitation: they make up 78% of test hazards, and their
detection rate stays far below the medium-object rate. More examples alone did not overcome the
resolution and capacity limit.

**Takeaway:** targeted sampling does not help if the added labels use a different localization
convention. The regression gate prevented the weaker candidate from replacing v3.

## Preventive checks added

- shared processor pixel limits for training and inference;
- an all-zero-IoU abort in grounding evaluation;
- checkpoint retention compatible with best-model restoration;
- explicit judge failure handling;
- scene-level leakage checks;
- label diversity and schema validation;
- a candidate-versus-baseline regression gate in CI.

The detailed v4 analysis is in [`FLYWHEEL_V4_FINDINGS.md`](FLYWHEEL_V4_FINDINGS.md). The
box-provenance experiment is in [`TASK3_DECONFOUND.md`](TASK3_DECONFOUND.md).
