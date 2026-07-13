#!/usr/bin/env bash
# One-command retrain + eval on the v2 labels for a fresh GPU pod (e.g. RunPod A100).
#
# Assumes: repo already cloned; deps installed directly (NOT `pip install -e .` —
# that fails on this repo's build backend):
#   pip install transformers peft accelerate bitsandbytes pillow pyyaml qwen-vl-utils numpy scipy tqdm
#
# Usage:
#   bash scripts/train_and_eval_v2.sh <DATAROOT> <BUNDLE_TGZ>
#     <DATAROOT>    where CAM_FRONT images will live, e.g. /workspace/nuscenes_trainval
#     <BUNDLE_TGZ>  the tarball built in Colab (labels/ + images/), e.g. /workspace/v2_bundle.tar.gz
#
# It unpacks the bundle, rewrites the baked-in image paths to this box, wires the
# trainer to the v2 labels, runs a SANITY GATE (aborts if the labels look collapsed
# or images are missing), then LoRA SFT -> generate predictions -> Level-1 eval.
set -euo pipefail

DATAROOT="${1:?usage: train_and_eval_v2.sh <DATAROOT> <BUNDLE_TGZ>}"
BUNDLE="${2:?usage: train_and_eval_v2.sh <DATAROOT> <BUNDLE_TGZ>}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$REPO/src"
export WANDB_DISABLED=true
V2="$REPO/outputs/data/sft_ready_v2"

echo "== 1/6 unpack bundle =="
mkdir -p "$DATAROOT/samples/CAM_FRONT" "$V2" /tmp/v2extract
tar -xzf "$BUNDLE" -C /tmp/v2extract
mv /tmp/v2extract/images/* "$DATAROOT/samples/CAM_FRONT/"
cp /tmp/v2extract/labels/*.jsonl "$V2/"
echo "   images on disk: $(ls "$DATAROOT/samples/CAM_FRONT" | wc -l)"

echo "== 2/6 rewrite image paths -> $DATAROOT =="
# Replace whatever prefix the labels were written with (the Colab Drive path) so the
# paths point at this box. Matches any '<prefix>/samples/CAM_FRONT/' inside a string.
sed -i "s#[^\"]*/samples/CAM_FRONT/#${DATAROOT}/samples/CAM_FRONT/#g" "$V2"/sft_*_enriched.jsonl

echo "== 3/6 wire trainer to v2 + sanity gate =="
( cd "$V2" && for s in train val test; do ln -sf "sft_${s}_enriched.jsonl" "sft_${s}.jsonl"; done )
sed -i 's#sft_output_dir:.*#sft_output_dir: "outputs/data/sft_ready_v2"#' "$REPO/configs/data.yaml"

python - "$V2" << 'PY'
import json, os, sys
d = sys.argv[1]; bad = False
for sp in ("train", "val", "test"):
    recs = [json.loads(l) for l in open(f"{d}/sft_{sp}.jsonl")]
    boxes = [tuple(h["bbox_2d"]) for r in recs
             for h in json.loads(r["messages"][-1]["content"]).get("hazards", []) if "bbox_2d" in h]
    ratio = len(set(boxes)) / len(boxes) if boxes else 0.0
    print(f"  {sp:5s}: {len(recs):4d} records | {len(boxes):4d} boxes | unique_box_ratio={ratio:.3f}")
    if sp == "train" and ratio < 0.5:
        print("  ABORT: train unique_box_ratio < 0.5 — labels look collapsed/wrong."); bad = True
r0 = json.loads(open(f"{d}/sft_train.jsonl").readline()); ip = r0["images"][0]
print("  sample image exists:", os.path.exists(ip), "->", ip)
if not os.path.exists(ip):
    print("  ABORT: sample image not found — path rewrite / transfer is wrong."); bad = True
sys.exit(1 if bad else 0)
PY

echo "== 4/6 A100 batch settings + LoRA SFT =="
sed -i 's/per_device_train_batch_size: .*/per_device_train_batch_size: 8/' "$REPO/configs/training.yaml"
sed -i 's/per_device_eval_batch_size: .*/per_device_eval_batch_size: 8/'   "$REPO/configs/training.yaml"
sed -i 's/gradient_accumulation_steps: .*/gradient_accumulation_steps: 2/'  "$REPO/configs/training.yaml"
python "$REPO/scripts/run_training.py" --config "$REPO/configs/training.yaml"

echo "== 5/6 generate predictions with the trained adapter =="
python "$REPO/scripts/run_generate_predictions.py" --split test \
  --adapter-path "$REPO/outputs/training/lora_adapter" \
  --ground-truth "$V2/sft_test_enriched.jsonl" \
  --output "$REPO/outputs/predictions/test_predictions.jsonl"

echo "== 6/6 Level-1 grounding eval =="
python "$REPO/scripts/run_evaluation.py" --level 1 \
  --predictions "$REPO/outputs/predictions/test_predictions.jsonl" \
  --ground-truth "$V2/sft_test_enriched.jsonl"

echo ""
echo "== DONE =="
echo "Copy these off the pod, then STOP it:"
echo "  outputs/training/lora_adapter/              (trained adapter)"
echo "  outputs/predictions/test_predictions.jsonl  (predictions)"
echo "  outputs/eval/level1/                        (grounding report + metrics)"
