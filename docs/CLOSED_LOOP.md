# Closed-loop, failure-driven mining

> Part of **[DriveSense-VLM](../README.md)** — see the README for status, the full results tables, and the canonical [What's left](../README.md#whats-left-future-work).
> Detection numbers trace to [`results/metrics_registry.json`](../results/metrics_registry.json); inference numbers to [`INFERENCE_OPTIMIZATION.md` §7](../INFERENCE_OPTIMIZATION.md).


**The problem this closes:** Level-1 eval showed uniform near-zero
box-grounding across ALL hazard classes, regardless of how many training
examples each class had (24 to 841 examples per class). That rules out "just
mine more examples of class X" as the fix — the real gap wasn't captured by
per-class counts at all. This introduces two tools that let *measured
failure* drive the next mining pass, instead of guessing.

> ⚠️ **Historical premise — read with the dates in mind.** That "uniform
> near-zero grounding" was measured *before* the coordinate-convention bug was
> found and fixed (see
> [`DEBUGGING_POSTMORTEM.md`](../DEBUGGING_POSTMORTEM.md), Failure 1). Post-fix,
> v3 grounds at P 0.40 / R 0.24 / F1 0.30 with mean IoU 0.67 — not near-zero.
> The *motivation* for failure-driven mining still holds and the tooling below
> is unchanged, but this paragraph describes the state of the world when the
> tools were written, not the current model. Canonical numbers:
> [`results/metrics_registry.json`](../results/metrics_registry.json).

## The loop

```
 ┌──────────┐   ┌──────────┐   ┌────────┐   ┌───────┐   ┌──────┐   ┌───────┐   ┌────────┐
 │ analyze  │──▶│  select  │──▶│  mine  │──▶│ label │──▶│ gate │──▶│ train │──▶│  eval   │──┐
 │ failure  │   │ targets  │   │(stream)│   │(batch │   │(valid)│  │(LoRA) │   │(level 1)│  │
 │ (tool 1) │   │(tool 2)  │   │        │   │ API)  │   │       │  │       │   │         │  │
 └──────────┘   └──────────┘   └────────┘   └───────┘   └──────┘   └───────┘   └────────┘  │
      ▲                                                                                      │
      └──────────────────────────────────────────────────────────────────────────────────────┘
```

1. **analyze** (`scripts/analyze_failure_stratification.py`) — cross-tabulates
   per-hazard grounding quality by GT box **size tier** (a distance/scale
   proxy) against **weather / time_of_day / location**, on the existing
   `predictions.jsonl` + enriched ground truth. Finds which condition is
   *actually* hard — reuses `grounding.compute_iou`, not reimplemented.
2. **select** (`scripts/select_mining_targets.py`) — reads the #1 report's
   worst bucket and the global `metadata.jsonl` (34,149 keyframes), scores
   every un-mined candidate frame by how well it matches that worst
   condition, and writes a new shopping list — same schema the existing
   miner already consumes.
3. **mine** — the existing `scripts/run_streaming_miner.py`, unmodified,
   fetching images for the new targeted list.
4. **label** — the existing `scripts/regenerate_annotations_v2_colab.py`
   (Batch API describe pass), unmodified.
5. **gate** — the existing `scripts/run_label_validation.py`, unmodified.
6. **train** — the existing LoRA SFT trainer, unmodified.
7. **eval** — the existing `scripts/run_evaluation.py` (Level 1), unmodified
   — its output feeds step 1 again on the *next* iteration.

Every step from "mine" onward is existing, already-built pipeline. The two
new pieces are purely the failure→target translation at the front of the loop.

## 1. `scripts/analyze_failure_stratification.py`

Runs standalone on existing eval artifacts — no model, no GPU.

```bash
python scripts/analyze_failure_stratification.py \
    --predictions outputs/predictions/test_predictions.jsonl \
    --ground-truth outputs/data/sft_ready_v2_merged/sft_test_enriched.jsonl \
    --output outputs/eval/failure_stratification.json
```

For each GT hazard with a `bbox_2d`: box area (% of frame), aspect ratio, and
a size tier — `tiny <1%`, `small 1-5%`, `medium 5-15%`, `large >15%`. Each
hazard's **best IoU** against any prediction in its frame is computed via
`grounding.compute_iou` (the same primitive `compute_grounding_metrics`
already uses for its frame-level `best_pair_ious` — kept per-hazard here
instead of collapsed to one value per frame, which is what makes slicing by
individual hazard size possible when a frame holds hazards of different
sizes). The report cross-tabulates size tier against weather / time_of_day /
location and ranks every bucket (with enough samples to be reliable) from
worst to best `mean_best_pair_iou`.

**What we found when validating the tooling**, running it against the
pre-v3 local eval artifacts available at the time: the `large` size tier came
out uniformly worst — 0% detection at every IoU threshold — while `small` was
the best-performing tier. That set had no weather/time-of-day variety
(100% clear/day), so the actionable signal collapsed to size tier alone.

> ⚠️ **This ordering is superseded and must not be quoted as a result.** It
> came from an older, smaller predictions file used to prove the tool runs on
> real data — *not* the fixed 1,041-frame v3/v4 test set. The canonical L4
> result reverses it: detection **scales with hazard size**, with `tiny` the
> worst tier (22.8% v3 / 17.2% v4 @0.5) and `medium` the best (52.6%), and rain
> and night+tiny as the weak conditions. See the
> [README results](../README.md#results-v2--v3--v4-fixed-test-set) and
> [`results/metrics_registry.json`](../results/metrics_registry.json). What this
> section demonstrates is that the *tooling* correctly surfaces a worst bucket
> from real data — not what that bucket turned out to be.

## 2. `scripts/select_mining_targets.py`

```bash
python scripts/select_mining_targets.py \
    --report outputs/eval/failure_stratification.json \
    --metadata outputs/data/spark_processed/metadata.jsonl \
    --have-manifest outputs/data/have_basenames.txt \
    --output outputs/data/mining_shoppinglist.jsonl \
    --target-count 2000
```

Candidate frames only have `metadata.jsonl` fields — no rendered image, no
projected 2D box yet — so every match signal here is an honestly-labeled
**proxy**, not a measurement:

| Target dimension | Proxy | Source |
|---|---|---|
| `size_tier` | `distance_to_ego` per annotation, bucketed into distance bands | pinhole-projection heuristic: closer ⇒ bigger |
| `weather` | keyword search over `scene_description` | mirrors the existing `configs/data.yaml` / `scene_meta()` convention |
| `time_of_day` | keyword search over `scene_description` | same |
| `location` | **not supported** — no such field exists in `metadata.jsonl` | dropped with a logged warning, not silently ignored |

A candidate's score is the average per-axis match fraction over whichever
axes the worst bucket specifies (hazard-relevant annotations only, via
`box_sourcing.nuscenes_category_to_hazard` — reused, not reimplemented). The
output reuses the existing shopping-list schema exactly
(`streaming_miner.write_shoppinglist`), plus one extra `mining_score` field
for traceability. `--bucket` can target the 2nd/3rd-worst bucket explicitly
instead of always chasing #1.

**Operational note:** after writing the new list, run the miner with
`--no-rebuild-list` pointing at the same `shoppinglist_path` — otherwise the
miner's implicit-rebuild-when-populated default (a prior fix in this same
pipeline) will silently regenerate a plain rarity-sampled list and overwrite
this targeted one.

```bash
python scripts/run_streaming_miner.py --no-rebuild-list
```

## What this session did and didn't do

**Built and validated:** both tools, end-to-end, against real local eval
artifacts (`test_predictions.jsonl`, `sft_test_enriched.jsonl`) and the real
34,149-frame `metadata.jsonl` — not just synthetic unit tests. The
stratification report correctly identified `large` as the worst tier on real
data; the target selector correctly dropped the unsupported `location` axis
with a warning, scored real candidates, and produced a shopping list the
existing miner loads without modification.

**Explicitly NOT done, and shouldn't be read as done:** running a full
additional mine → label → gate → train → eval cycle using these tools. That
needs real GPU/API budget for the describe pass and training, which is a
deliberate, separate decision — this session only builds and validates the
*tooling* that makes that next cycle failure-driven instead of guesswork,
when someone chooses to spend that budget.
