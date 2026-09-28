# Box-provenance experiment

[Back to the project README](../README.md)

The v4 training set changed two things at once. It added 1,442 targeted rain and nighttime frames,
and those new records used boxes emitted by a foundation model instead of boxes projected from
nuScenes ground truth. Because both variables changed together, the v4 result could not show which
one caused the regression.

This experiment isolates box source. Two Qwen2.5-VL-3B LoRA models use the same base data,
validation data, test data, targeted frame IDs, descriptions, and training settings. Only the
targeted boxes differ:

- **FM arm:** foundation-model-emitted boxes;
- **GT arm:** boxes projected from nuScenes 3D annotations.

## Dataset and training setup

The original per-frame v3/v4 artifacts were lost when their RunPod volume was reclaimed. The study
therefore reconstructs the design from the surviving nuScenes tables and CAM_FRONT images. It is
not a byte-for-byte replay of v3 or v4.

| Split | Frames |
|---|---:|
| Shared base train set | 2,652 |
| Shared validation set | 889 |
| Targeted addition | 1,162 |
| Image-available test subset | 402 of the original 1,041 |

Both arms use LoRA rank 32, alpha 64, three epochs, and effective batch size 16 on one H100. The
assembly step verifies that train scenes do not overlap validation or test scenes. Approximate API
cost for description and FM-label passes was $22.

## Results

| Metric | FM arm | GT arm |
|---|---:|---:|
| Recall | 0.101 | 0.167 |
| Precision | 0.176 | 0.330 |
| F1 | 0.128 | 0.222 |
| False-positive rate | 0.488 | 0.169 |
| Mean best-pair IoU | 0.244 | 0.458 |
| Frame detection at IoU 0.5 | 0.266 | 0.566 |
| No-hazard accuracy | 0.512 | 0.831 |
| Mean IoU for matched boxes | 0.632 | 0.641 |
| Class accuracy for matched boxes | 0.962 | 0.954 |

The two arms classify matched boxes at almost the same rate. The large differences are in whether a
box is produced and where it is placed. This identifies localization supervision as the main
source of the gap.

### Results by condition

| Condition | Frames | FM arm | GT arm |
|---|---:|---:|---:|
| Day | 283 | 0.130 | 0.171 |
| Night | 119 | 0.000 | 0.151 |
| Clear | 298 | 0.127 | 0.197 |
| Rain | 104 | 0.000 | 0.050 |

The FM arm detected no hazards at night or in rain. The GT arm was still weak in rain, but it did
not collapse to zero. On the rain subset, the FM arm's false-positive rate was 1.00 compared with
0.069 for the GT arm.

## Interpretation

Within this controlled reconstruction, GT-projected supervision improved both localization and
calibration. It roughly doubled F1, reduced false positives, and raised mean best-pair IoU from
0.244 to 0.458.

This supports the conclusion that the box-source change contributed materially to the v4
regression. It does not show that targeted mining itself was harmful, because the experiment was
designed to isolate provenance rather than compare targeted and untargeted data.

## Limitations

- The study reconstructs the original design but does not replay the original v3/v4 files.
- The base set is smaller than v3's 7,228-frame training set.
- Evaluation covers 402 of the fixed 1,041 test frames because only those images remained.
- The rain subset contains 104 frames.
- Each arm was trained once, so there are no confidence intervals across seeds.

The relative FM-versus-GT result is more informative than the absolute scores under this reduced
setup.

## Reproduction

Committed metrics are under
[`results/task3_deconfound/`](../results/task3_deconfound/). The directory contains both arms'
grounding and robustness reports plus `deconfound_result.json`.

The full procedure is in
[`experiments/task3_deconfound/RUNBOOK.md`](../experiments/task3_deconfound/RUNBOOK.md). Its main
steps are:

1. reconstruct manifests from nuScenes;
2. describe GT boxes through the Batch API;
3. generate the FM-boxed targeted set;
4. assemble matched arms and verify split isolation;
5. train both adapters;
6. generate predictions;
7. run Level 1 and Level 4 evaluation;
8. compare the two result directories.
