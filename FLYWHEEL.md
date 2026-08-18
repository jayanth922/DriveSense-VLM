# DriveSense-VLM — The Data Flywheel (self-improving loop)

A repeatable, mostly-automated loop that turns a model's *measured weaknesses* back into
*targeted training data*, then verifies whether the change helped — with a gate that blocks
regressions. This is the "Data & Test Flywheel" pattern: select → label → train → test →
repeat, with minimal human intervention.

```
        ┌─────────────────────────────────────────────────────────────┐
        │                                                             │
        ▼                                                             │
  (1) EVAL + STRATIFY ──► (2) SELECT weak-bucket frames ──► (3) MINE  │
   L1/L4 failure map        from the failure map            adverse   │
        ▲                                                    frames   │
        │                                                      │      │
  (6) REGRESSION GATE ◄── (5) RE-EVAL ◄── (4b) RETRAIN ◄── (4a) AUTO- ┘
   promote or BLOCK        fixed test      LoRA SFT          LABEL + GATE
                                                            (FM behind a
                                                             validation gate)
```

## The loop, one stage per command

| stage | command | output |
|---|---|---|
| 1. Eval + stratify | `run_evaluation.py --level 1` ; `analyze_failure_stratification.py` | `failure_stratification.json` (the weak buckets) |
| 2. Select targets | `select_mining_targets.py` (reads the failure map) | ranked mining shopping list |
| 3. Mine | `scripts/v4/build_v4_manifest.py` | leakage-safe candidate manifest (verified vs real nuScenes tables) |
| 4a. Auto-label + gate | `scripts/v4/v4_batch_label.py` | SFT labels (FM behind the validation gate; 7-class constrained) |
| 4b. Build + retrain | `scripts/v4/v4_build_trainset.py` ; `run_training.py` | new adapter (best-by-eval-loss) |
| 5. Re-eval | `run_generate_predictions.py` ; `run_evaluation.py` ; `analyze_failure_stratification.py` | new predictions + failure map |
| 6. Gate | `run_regression_gate.py` | **PROMOTE or BLOCK** vs the current production model |

## Guardrails that make it trustworthy (not just automated)

- **Leakage-safe by construction.** Every mined frame is mapped to its nuScenes `scene_token`
  and dropped if it shares a scene with the *fixed* val/test split. The test set never moves,
  so v(n) vs v(n+1) is always apples-to-apples. (v4 dropped 986 leaking frames.)
- **Validation gate on labels.** The foundation-model labeler is constrained to the 7-class
  taxonomy, prompt-hardened against over-labeling, and every box is repaired/validated before
  it becomes training data. Gate pass/fail counts are logged.
- **Best-checkpoint discipline.** Early stopping on eval_loss; `save_total_limit ≥ epochs` so
  the best checkpoint is never pruned.
- **Regression gate closes the loop.** A candidate that degrades weak-bucket recall is
  **blocked**, not promoted — the loop can run unattended without silently shipping a worse
  model. (v4 was correctly blocked.)

## What the v3→v4 turn taught us

The loop ran end-to-end and returned an **honest negative**: targeted adverse data regressed
the very buckets it mined for (rain det@0.5 12.5%→7.4%). That is the flywheel working as
designed — it produced a trustworthy measurement and the gate caught the regression. The next
turn's lever, per the diagnosis, is **model-side** (resolution / tiny-box weighting), not more
data. A self-improving system that can conclude "this change didn't help, don't ship it" is
more valuable than one that only knows how to add data.

## Toward one-command operation

The stages above are pure functions over the persistent data volume, so the loop is a thin
orchestrator away from `make flywheel` (or an Airflow/Prefect DAG): each stage is idempotent,
writes a JSON artifact, and the next stage consumes it. The regression gate's exit code is the
promotion signal for CI/CD. See `.github/workflows/ci.yml` for the gate wired into CI.
