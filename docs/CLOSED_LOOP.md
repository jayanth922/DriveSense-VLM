# Failure-driven data selection

[Back to the project README](../README.md)

This part of the pipeline uses evaluation results to choose the next frames to mine. It replaces
the rule "collect more examples from a weak class" with a more specific question: under which
conditions does the model fail?

The selector can target combinations such as tiny objects in rain or small objects at night. It
does not train a model or download data by itself.

## Workflow

```text
predictions + enriched ground truth
             |
             v
failure stratification
             |
             v
ranked target conditions
             |
             v
candidate scoring from global metadata
             |
             v
streaming-miner shopping list
```

The resulting list feeds the existing mining, annotation, validation, training, and evaluation
commands. A later evaluation becomes the input to the next iteration.

## Analyze grounding failures

`scripts/analyze_failure_stratification.py` runs on saved predictions and labels. It does not load
the model or require a GPU.

```bash
python scripts/analyze_failure_stratification.py \
  --predictions outputs/predictions/test_predictions.jsonl \
  --ground-truth outputs/data/sft_ready_v2_merged/sft_test_enriched.jsonl \
  --output outputs/eval/failure_stratification.json
```

For each ground-truth hazard with a box, the script calculates:

- box area as a percentage of the frame;
- aspect ratio;
- size tier: tiny below 1%, small from 1-5%, medium from 5-15%, and large above 15%;
- best IoU against any prediction in the frame.

It then reports detection rates at IoU 0.1, 0.3, and 0.5, plus mean best-pair IoU, grouped by
size, weather, time of day, and location. IoU comes from the same
`drivesense.eval.grounding.compute_iou` function used by the main evaluator.

An early tool-validation run used an older prediction file and ranked large objects as the worst
tier. That ordering is not a current project result. On the fixed v3/v4 test set, tiny objects are
the weakest size tier. Current values are stored in
[`results/metrics_registry.json`](../results/metrics_registry.json).

## Select new mining targets

`scripts/select_mining_targets.py` reads the stratification report and the global nuScenes
metadata file, then scores frames that have not already been mined.

```bash
python scripts/select_mining_targets.py \
  --report outputs/eval/failure_stratification.json \
  --metadata outputs/data/spark_processed/metadata.jsonl \
  --have-manifest outputs/data/have_basenames.txt \
  --output outputs/data/mining_shoppinglist.jsonl \
  --target-count 2000
```

The global metadata does not include rendered images or projected boxes, so candidate scoring uses
proxies:

| Target | Available proxy |
|---|---|
| Size tier | annotation distance from the ego vehicle |
| Weather | keywords in the scene description |
| Time of day | keywords in the scene description |
| Location | unavailable; removed with a warning |

Each candidate receives a match score for the requested condition. The output uses the same JSONL
shopping-list format as the streaming miner and adds `mining_score` for traceability.

Use `--bucket` to select a condition other than the worst-ranked one. When passing the generated
list to the miner, preserve it with `--no-rebuild-list`:

```bash
python scripts/run_streaming_miner.py --no-rebuild-list
```

Without that flag, the miner may rebuild its default rarity sample and replace the targeted list.

## What has been validated

Both tools have unit tests and were run against real saved evaluation artifacts and the
34,149-frame metadata set. The selector produced a shopping list accepted by the existing miner,
and unsupported dimensions were reported rather than silently ignored.

The repository has not run a separate full mine-label-train cycle solely from these two tools.
That would require additional API and GPU spending. The implemented and tested contribution here
is the failure-to-shopping-list path.
