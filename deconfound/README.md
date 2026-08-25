# deconfound/ — Task 3 (FM-vs-GT de-confound) pipeline

Regenerated 2026-08-24 from the `Task3_Deconfound_Pipeline` design + live repo
internals (the original `deconfound_pipeline.tgz` was not in git). Drop this
folder at the repo root: `/workspace/DriveSense/deconfound/`.

Files:
- `reconstruct.py`     — `--preflight` ($0 gate) and `--build` (manifests + cost_estimate.json)
- `describe_manifest.py` — GT-describe (severity/reasoning/action on GT boxes) via Batch API, resumable
- `build_arms.py`      — `prep` (GT manifest for finalized ids) + `assemble` (arm_fm/arm_gt, leak asserts)
- `compare_arms.py`    — FM-vs-GT table -> deconfound_result.json
- `training_h100.yaml` — H100 training config (faithful recipe, eff-batch 16)
- `model.yaml`, `data.yaml` — configs run_training.py loads from this dir (SFT_DIR/OUT_DIR/ATTN_IMPL env-driven)
- `RUNBOOK.md`         — phases 0–7 with $0 gates + cost checkpoints. **Start here.**

Reuses existing repo scripts for FM labeling/finalize: `scripts/v4/v4_batch_label.py`,
`scripts/v4/v4_finalize_sft.py`, and eval via `scripts/run_generate_predictions.py`
+ `scripts/run_full_evaluation.py`.

Run the Phase 2.5 mock dry-run once before spending — it validates the whole chain at $0.
