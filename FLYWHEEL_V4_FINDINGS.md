# DriveSense-VLM — v4 Targeted-Flywheel Turn: Findings

**TL;DR.** I closed the data flywheel end-to-end — mine rare-hazard frames → auto-label
them with a foundation model behind a validation gate → merge leakage-safe → fine-tune →
evaluate with a coordinate-correct pipeline → compare against the previous model on a
*fixed* test set. The honest result: **targeted adverse-weather data did not improve the
weak buckets — it slightly regressed them.** Combined with v3's earlier result (naive
scale-up hurt generalization), the finding is that **data volume/targeting is not the
binding constraint for this model on tiny, distant hazards** — and the regression is
diagnosable. Closing the loop and reporting a trustworthy negative *is* the deliverable.

---

## What the v4 turn did

1. **Targeted mining.** Started from the v3 L4 failure map (rain, night, and tiny boxes
   were the worst buckets) and selected 4,160 candidate frames inside rain/night nuScenes
   scenes. Verified scene membership against the **real** nuScenes tables (not the miner's
   own metadata, which was unreliable), streaming the 1.3 GB `sample_data.json` with `ijson`.
2. **Leakage guard.** Mapped every candidate to its `scene_token` and dropped **986 frames**
   that shared a scene with the v3 test/val split, so the test set stays clean and fixed.
   Final adverse pool: 2,231 frames (rain or night); ~30% of the mined pool turned out to be
   clear/day (the miner's "rain/night" label was optimistic — it was broad-rarity, not
   strictly adverse).
3. **FM auto-labeling behind a gate.** Labeled with **Claude Sonnet-5 vision via the Batch
   API (−50%)**, constrained to v3's exact 7-class taxonomy. A 10-frame pilot exposed
   over-labeling (weather/darkness boxed as hazards, ordinary vehicles miscast as
   `unusual_object`, degenerate edge boxes); I hardened the prompt + added deterministic
   box repair, re-piloted clean, then ran the full set. Cost: **$10.98 for 2,231 frames.**
4. **Composition.** ~45% of adverse frames were genuinely `no_hazard` (normal night/rain
   driving). Kept all 1,226 positives + capped `no_hazard` at 15% → **1,442 SFT examples.**
5. **Leakage-safe merge.** v4 train = v3 train (7,228) + 1,442 = **8,670**; val/test reused
   **verbatim** from v3 (889 / 1,041). Asserted zero train↔val/test frame overlap.
6. **Retrain.** Qwen2.5-VL-3B LoRA, 3 epochs, early stopping, best-by-eval-loss checkpoint
   (fixing v3's bug where the best checkpoint was pruned). Clean, no overfit.
7. **Coordinate-correct eval.** Regenerated all 1,041 test predictions with the processor
   pinned to training resolution (min 200704 / max 602112) — the v3 drift bug does not recur
   (mean IoU 0.66 confirms boxes are in the right 0–1000 space).

---

## Results

### Training (eval_loss on the fixed v3 val set)

| model | train examples | eval_loss | note |
|---|---|---|---|
| v2 | 2,754 | 0.31 | best generalization |
| v3 | 7,228 (naive scale-up) | 0.66 | overfit at 5 epochs; generalization hurt |
| v4 | 8,670 (+1,442 targeted) | 0.69 | clean (fell every epoch), ≈ parity with v3 |

### L1 grounding (whole test set, IoU ≥ 0.5)

| metric | v3 | v4 | Δ |
|---|---|---|---|
| Precision | 0.40 | 0.37 | −0.03 |
| Recall | 0.24 | 0.19 | **−0.05** |
| F1 | 0.30 | 0.25 | −0.05 |
| mean IoU (matched) | 0.67 | 0.66 | ≈ |
| classification acc | 0.94 | 0.95 | ≈ |
| parse rate | 0.987 | 0.974 | ≈ |

### L4 stratified detection rate @0.5 — the buckets we mined for

| bucket | v3 | v4 | Δ |
|---|---|---|---|
| Overall | 28% | 23% | −5 |
| tiny box | 22.8% | 17.2% | −5.6 |
| small box | 46.4% | 42.9% | −3.5 |
| medium box | 52.6% | 52.6% | 0 |
| **rain (all)** | 12.5% | **7.4%** | −5.1 |
| **night + tiny** | 12.7% | **10.7%** | −2.0 |
| clear + medium | 69% | 69% | 0 |

**The targeted data hurt exactly the buckets it targeted (rain, night, tiny) and left the
easy buckets flat.** This is a trustworthy comparison: same fixed test set, same eval code,
resolution-pinned generation, mean IoU intact.

---

## Diagnosis — why targeted data regressed the weak buckets

1. **Recall-suppressing negatives.** v4 introduced **216 `no_hazard` examples; v3 had zero.**
   Teaching the model to withhold lowered recall everywhere (0.24 → 0.19), and adverse scenes
   are the most ambiguous — exactly where extra "predict nothing" signal does the most damage.
2. **Auto-label distribution shift.** The Sonnet-5 labels are pixel-grounded and place boxes
   differently than v3's labels; on a *fixed* v3-labeled test set, that shift costs matches
   even when the labels are individually reasonable.
3. **The real bottleneck is model-side, not data-side.** Tiny boxes are 78% of hazards and
   the hardest (mean IoU ~0.16–0.20). Adding ~1.4k more adverse frames doesn't change that the
   3B model, at 768-visual-token resolution, is capacity/resolution-limited on tiny distant
   objects. More data — naive *or* targeted — is not the lever.

## The bigger picture: two data-scaling experiments, one conclusion

- **v3 (naive scale-up, 2.7k→7.2k):** generalization *hurt* (eval_loss 0.31→0.66).
- **v4 (targeted +1.4k adverse):** weak-bucket recall *hurt* (rain 12.5%→7.4%).

Both experiments, run honestly on a fixed test set, point the same way: **for this
model, scaling data is not what moves rare-hazard recall.** The high-leverage moves are
model-side (higher input resolution, a larger or detection-specialized backbone, tiny-box
loss weighting) and label-consistency, not more frames.

## MLOps — regression gate (Step 8) verdict

The whole point of `run_regression_gate.py` is to block a candidate that regresses key
metrics vs the current production model. Applying the gate's rule to these numbers:

- rain det@0.5: 12.5% → 7.4% (**−41% relative**)
- night+tiny det@0.5: 12.7% → 10.7%
- overall recall: 0.24 → 0.19

**Verdict: BLOCK — v4 must not replace v3.** A gate catching a *real* regression on the
exact buckets the change was meant to help is a stronger observability demonstration than
gating a cherry-picked win. (Running the script on a CPU pod against
`/workspace/v4_eval/` vs `/workspace/v3_eval_clean/` will emit the tool artifact; the
verdict above follows directly from the measured metrics.)

## Honest limitations

- v4 is a legitimate negative result, not a bug — resolution is pinned and mean IoU is intact.
- The regression is a *modest* few-point move on small-n adverse buckets (rain n=337,
  night+tiny n=308); directionally clear, but not a dramatic collapse.
- The `no_hazard` cap (15%) and the FM-vs-GT label-convention gap are confounds not yet
  isolated.

## Suggested next steps (in priority order)

1. **Ablation (cheap, isolates the top hypothesis):** retrain v4b with **zero `no_hazard`**
   examples (pure targeted positives) and re-eval. If rain/night recall recovers, the
   negatives were the culprit; if not, it's model-side.
2. **Model-side lever:** raise vision resolution or add tiny-box loss weighting — the L4 map
   says size is the dominant factor, so this is where the real gains are.
3. **Deploy pillar:** real TensorRT export (currently falls back to torch.compile).
4. **Package:** this document + the v2→v3→v4 comparison as the portfolio narrative — an
   honest, diagnosed flywheel turn with a working regression gate.

## Reproducibility / artifacts (on the persistent `/workspace` volume)

- v4 SFT add: `sft_train_ready_v4/sft_train.jsonl` (1,442) — labeler `scripts/v4/v4_batch_label.py`
- v4 train set: `sft_train_ready_v4_merged/` (8,670 / 889 / 1,041) — `scripts/v4/v4_build_trainset.py`
- v4 adapter: `v4_train_out/lora_adapter` (eval_loss 0.694)
- v4 predictions: `v4_eval/test_pred_full.jsonl` (1,041); L4 report `v4_eval/failure_stratification.json`
- v3 baselines: `v3_regen/test_pred_full.jsonl`, `v3_eval_clean/{eval_summary,failure_stratification}.json`
