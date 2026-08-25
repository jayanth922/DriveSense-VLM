# Task 3 de-confound — RUNBOOK (RunPod H100)

Regenerated 2026-08-24 from the `Task3_Deconfound_Pipeline` design + the live repo
internals (box_sourcing, batch_describe, v4 chain, SFT formatter, eval grounding /
failure_stratification, model/training configs). These wrapper scripts are a
faithful rebuild; **the $0 gates below (pre-flight, cost_estimate, leak asserts,
and the mock dry-run) exist so a bug is caught before any spend.** Do the mock
dry-run once before real money.

Experiment: two SFT arms sharing an IDENTICAL base+val + IDENTICAL targeted
frame_ids; they differ ONLY in targeted boxes (FM-labeled vs nuScenes-GT). Both
eval on the fixed 1,041 test set. Headline: does GT beat FM on rain/night det@0.5?

```
REPO=/workspace/DriveSense       # your clone
NUSC_ROOT=/workspace/nuscenes    # tables in v1.0-trainval/, images in samples/CAM_FRONT/
WORK=/workspace/deconfound_work  # all outputs land here
```
Everything lives on the network volume (/workspace) so a pod reclaim can't wipe it.

---

## Phase 0 — Environment (once; ~$0 + GPU clock)

```bash
cd /workspace/DriveSense
export PYTHONPATH=/workspace/DriveSense/src
export HF_HOME=/workspace/hf_cache            # cache base model on the volume
export REPO=/workspace/DriveSense
export NUSC_ROOT=/workspace/nuscenes
export WORK=/workspace/deconfound_work
export ANTHROPIC_API_KEY="sk-ant-..."
export SONNET="claude-sonnet-5"               # canonical model id used throughout the repo

# deps into base Python (torch 2.8/cu128 + numpy 2.1.2 already correct on this image)
pip install -q nuscenes-devkit pyquaternion ijson
pip install -q --force-reinstall --no-deps "numpy>=2.1,<2.3"   # ABI: re-pin AFTER devkit
pip install -q "transformers<5" peft accelerate bitsandbytes pillow pyyaml qwen-vl-utils anthropic tqdm
python3 -c "import torch,nuscenes,anthropic,numpy;print('env ok', torch.cuda.is_available(), numpy.__version__)"
```
Put `deconfound/` in the repo root (this folder). `model.yaml` and `data.yaml`
ship inside it — `run_training.py` reads them from the config's directory.

---

## Phase 1 — Pre-flight ($0 GATE — nothing spends before this is green)

```bash
python deconfound/reconstruct.py --preflight
```
Require the last line: **`PREFLIGHT: PASS (UNRESOLVED=0)`**. It also saves the
test scene tokens to `$WORK/test_scene_tokens.json` for exclusion.
If it FAILs, stop — `$WORK/unresolved_ids.json` lists the ids that don't map to
nuScenes; the eval GT can't be rebuilt until that mapping is fixed.

---

## Phase 2 — Build manifests + cost gate ($0)

```bash
python deconfound/reconstruct.py --build
cat $WORK/cost_estimate.json     # require "gate_pass": true  (total_api_usd <= 40)
```
Writes `$WORK/{eval_gt, base_val, targeted}/` and `cost_estimate.json`. Expected
total ~$30–38. If `gate_pass` is false, do NOT proceed to describe/label.

Sanity-check reconstruction against the authoritative v4 numbers before spending:
`$WORK/targeted/*_summary` weather/time should be rain|day-heavy, and base_val
train≈7228 / val≈889. (Sizes are tunable via `N_BASE/N_VAL/N_TARGETED`.)

---

## Phase 2.5 — MOCK dry-run ($0 — validate the whole chain end-to-end)

Run the API + assembly steps in mock mode so a wiring bug surfaces for free:

```bash
python deconfound/describe_manifest.py --manifest $WORK/base_val/annotated_manifest.json \
    --out $WORK/base_val --mock
python scripts/v4/v4_batch_label.py --manifest $WORK/targeted/v4_manifest.jsonl \
    --out /workspace/v4/annotated --sft-out /workspace/sft_train_ready_v4 --mock
python scripts/v4/v4_finalize_sft.py
mkdir -p $WORK/targeted_fm && cp /workspace/sft_train_ready_v4/sft_train.jsonl $WORK/targeted_fm/
python deconfound/build_arms.py prep
python deconfound/describe_manifest.py --manifest $WORK/targeted_gt/annotated_manifest.json \
    --out $WORK/targeted_gt --mock
python deconfound/build_arms.py assemble
```
Success = `arm_fm/` and `arm_gt/` exist with **zero leakage** and identical train
ids. Then delete the mock outputs (`rm -rf $WORK/base_val/sft_*.jsonl
$WORK/targeted_fm $WORK/targeted_gt /workspace/v4 /workspace/sft_train_ready_v4`)
and do the real runs below.

---

## Phase 3 — GT-describe base+val (API; ~$22, downsized)

```bash
python deconfound/describe_manifest.py \
    --manifest $WORK/base_val/annotated_manifest.json \
    --out $WORK/base_val --state $WORK/base_val/describe_batches.json \
    --downsize 768 --model "$SONNET"
```
Resumable: re-running drains in-flight batches and skips cached frames (no double
charge). Produces `$WORK/base_val/sft_{train,val}.jsonl`.

---

## Phase 4 — FM-label targeted + finalize (API; ~$9, full-res)

```bash
# 4a. FM boxes on the 2,231 adverse frames (Batch API; prints batch_cost_usd)
python scripts/v4/v4_batch_label.py \
    --manifest $WORK/targeted/v4_manifest.jsonl --model "$SONNET" \
    --out /workspace/v4/annotated --sft-out /workspace/sft_train_ready_v4
# 4b. finalize: clean to 7 classes, keep positives, cap no_hazard to 15% -> ~1,442
python scripts/v4/v4_finalize_sft.py
cat /workspace/v4/annotated/v4_sft_final_report.json   # check label dist vs authoritative
# 4c. hand the finalized FM SFT to the arm builder
mkdir -p $WORK/targeted_fm && cp /workspace/sft_train_ready_v4/sft_train.jsonl $WORK/targeted_fm/
```
Validate `v4_sft_final_report.json` label distribution against the known add mix
(construction 566, occ_ped 439, high_density 279, no_hazard 216, jay 122,
unusual 89, cyclist 75). Big deviations mean the adverse pool drifted.

---

## Phase 5 — GT targeted (same ids) + describe + assemble arms (API; ~$4)

```bash
# 5a. emit GT-boxed manifest for exactly the FM-finalized ids
python deconfound/build_arms.py prep
# 5b. describe those GT boxes (so both arms' prose comes from the same pass)
python deconfound/describe_manifest.py \
    --manifest $WORK/targeted_gt/annotated_manifest.json \
    --out $WORK/targeted_gt --downsize 768 --model "$SONNET"
# 5c. assemble both arms + leak asserts
python deconfound/build_arms.py assemble
```
`assemble` must print zero leakage and identical train ids across arms. Now
`$WORK/arm_fm/` and `$WORK/arm_gt/` each hold `sft_{train,val,test}.jsonl`.

---

## Phase 6 — Train both arms (GPU; ~$4–10)

```bash
# TF32 on H100
export NVIDIA_TF32_OVERRIDE=1
# ATTN: sdpa is default & safe. If flash-attn is built: export ATTN_IMPL=flash_attention_2

SFT_DIR=$WORK/arm_fm OUT_DIR=$WORK/out_fm \
  python scripts/run_training.py --config deconfound/training_h100.yaml
SFT_DIR=$WORK/arm_gt OUT_DIR=$WORK/out_gt \
  python scripts/run_training.py --config deconfound/training_h100.yaml
```
Tip: `--dry-run` first to time one micro-batch. If OOM at `pdb=16`, edit
`training_h100.yaml` to `per_device_train_batch_size: 8` +
`gradient_accumulation_steps: 2` (same effective batch 16).

---

## Phase 7 — Generate predictions, evaluate, compare

```bash
for arm in fm gt; do
  SFT_DIR=$WORK/arm_$arm python scripts/run_generate_predictions.py \
      --adapter-path $WORK/out_$arm --split test \
      --output $WORK/preds_$arm.jsonl --config configs/eval.yaml
  python scripts/run_full_evaluation.py --level 1 4 \
      --predictions $WORK/preds_$arm.jsonl \
      --ground-truth $WORK/eval_gt/sft_test.jsonl \
      --output-dir $WORK/results_$arm
done

python deconfound/compare_arms.py \
    --fm $WORK/results_fm --gt $WORK/results_gt \
    --out $WORK/deconfound_result.json
```
(Confirm the two eval script flags with `--help` on your build.) `compare_arms.py`
prints the FM-vs-GT table and the headline verdict, and writes
`deconfound_result.json`.

**Ship back for the write-up:** `deconfound_result.json` +
`$WORK/results_fm/failure_stratification.json` +
`$WORK/results_gt/failure_stratification.json`.

---

## Cost ledger (gate ≤ $40)
| phase | what | est |
|------|------|-----|
| 3 | describe base+val (~8,117 × downsized) | ~$22 |
| 4 | FM-label targeted (2,231 × full-res) | ~$9 |
| 5 | describe targeted-GT (~1,442 × downsized) | ~$4 |
| — | **API total** | **~$35** |
| 6 | 2× H100 training | ~$4–10 |

## Honest framing
Faithful-*family* reconstruction — the exact v3/v4 frames were lost, so this is
NOT a 1:1 replay of the published rows. The FM-vs-GT contrast is internally clean
(identical base/val/test + identical targeted frames; only boxes differ).
