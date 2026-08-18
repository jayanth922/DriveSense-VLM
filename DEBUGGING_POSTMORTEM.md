# DriveSense-VLM — Debugging Postmortem

Three real failures found and diagnosed across v2→v3→v4. The value of this project is not a
high accuracy number — it is a rigorous lifecycle that *surfaces* failures and roots them out.
Every number below is measured on the fixed 1,041-frame test set.

---

## Failure 1 — All-zero IoU: a coordinate-convention bug (not the model)

**Symptom.** Official L1 grounding returned IoU ≈ 0 across every class. Easy misread:
"the model can't localize."

**Root cause.** A Qwen2-VL vs Qwen2.5-VL convention footgun in the *generation* path. The
prediction script loaded the processor from the base model with **no `min/max_pixels`**, so
inference ran at Qwen's ~1 MP default instead of training's 602112. The model saw
out-of-distribution image sizes; Qwen2.5-VL's native **absolute-pixel** grounding prior then
overrode the learned **0–1000 normalized** convention, and predicted boxes drifted off-scale
(17% of coords > 1000) while GT stayed 0–1000. IoU on mismatched units collapses to zero.

**Fix.** Pin the processor to training resolution (`min_pixels 256 / max_pixels 768` →
200704 / 602112) in the generation path. Coordinate drift went to **0/64 sampled coords > 1000**;
clean L1 emerged (P 0.40 / R 0.24 / F1 0.30 @0.5, mean IoU 0.67). Added a
`_assert_iou_not_all_zero()` guard so this class of bug fails loudly forever.

**Lesson.** A metric of exactly zero is a systems bug until proven otherwise. Resolution is
part of the model contract for VLMs — train and inference must match to the pixel budget.

## Failure 2 — Naive data scaling *hurt* generalization (v2→v3)

**Symptom.** Scaling SFT data 2,754 → 7,228 examples **doubled eval_loss** (0.31 → 0.66)
while train_loss fell — a textbook overfitting/quality-dilution signature.

**Root cause.** More data was not better data: the naive scale-up diluted label quality and
over-fit at 5 epochs. Two secondary bugs compounded it: the best checkpoint was **pruned by
`save_total_limit`** so `load_best_model_at_end` restored the *final* (overfit) state, and the
LLM judge silently scored 1.0 on every call because Sonnet rejected a `temperature` param and
the exception defaulted to score=1.

**Fix.** 2–3 epochs + early stopping on eval_loss; `save_total_limit ≥ epochs` so the best
checkpoint survives; judge hardened to return `None` on error (not a fake 1.0).

**Lesson.** "Add more data" is a hypothesis, not a strategy. Measure generalization, not
train loss; and guard every silent-default path in eval tooling.

## Failure 3 — Targeted data *also* didn't help — the honest flywheel result (v3→v4)

**Hypothesis.** The L4 failure map said rain / night / tiny boxes were the weak buckets, so a
*targeted* flywheel turn (mine adverse frames → FM auto-label behind a gate → leakage-safe
retrain) should lift exactly those buckets.

**Result.** It did not. On the fixed test set, det@0.5 **regressed** on the mined buckets:
rain 12.5% → 7.4%, night+tiny 12.7% → 10.7%, tiny 22.8% → 17.2%; overall recall 0.24 → 0.19.
Mean IoU held at 0.66, so this is a real result, not a drift artifact.

**Diagnosis.** (1) v4 introduced 216 `no_hazard` negatives (v3 had zero) → the model learned
to withhold → recall dropped everywhere, worst on the ambiguous adverse scenes. (2) The
Sonnet-5 pixel-grounded labels place boxes differently than v3's labels → distribution shift
on a v3-labeled test set. (3) The dominant factor is size: tiny boxes are 78% of hazards at
mean IoU ~0.16 — a 3B model at 768 visual tokens is capacity/resolution-limited there, and
~1.4k more frames doesn't move that.

**Conclusion.** Two data-scaling experiments — naive (v3) and targeted (v4) — both show that,
for this model, **scaling data is not the lever for rare-hazard recall**. The high-leverage
moves are model-side (input resolution, tiny-box loss weighting, a detection-specialized head)
and label-convention consistency. The flywheel *loop* is the deliverable; the honest negative
is the finding, and the regression gate is what catches it before promotion.

---

## What makes these "senior" findings

Each failure was (a) reproduced on a fixed benchmark, (b) root-caused to a specific mechanism
rather than hand-waved as "the model is bad," (c) fixed with a guard that prevents recurrence,
and (d) reported honestly even when the answer was "our idea didn't work." That loop — surface,
diagnose, guard, report — is the perception-gap discipline these roles are hiring for.
