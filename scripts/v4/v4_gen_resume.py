#!/usr/bin/env python3
# Resume-safe v4 prediction generation: generate in chunks, commit each to the
# main output so a pod restart continues instead of restarting. Idempotent.
import json, os, subprocess, sys
GT      = "/workspace/sft_ready_v3_merged/sft_test_enriched.jsonl"
OUT     = "/workspace/v4_eval/test_pred_full.jsonl"
ADAPTER = "/workspace/v4_train_out/lora_adapter"
TMP_GT  = "/workspace/v4_eval/_missing_gt.jsonl"
TMP_OUT = "/workspace/v4_eval/_part.jsonl"
CHUNK   = 200

def fids(path):
    s = set()
    if os.path.exists(path):
        for l in open(path):
            l = l.strip()
            if not l: continue
            try: d = json.loads(l)
            except Exception: continue
            if d.get("frame_id"): s.add(d["frame_id"])
    return s

gt = [json.loads(l) for l in open(GT) if l.strip()]
total = len(gt)
while True:
    done = fids(OUT)
    missing = [g for g in gt if g.get("frame_id") not in done]
    print(f"[resume] done={len(done)}/{total}  missing={len(missing)}", flush=True)
    if not missing:
        print(f"[resume] COMPLETE: {len(done)}/{total}", flush=True); break
    chunk = missing[:CHUNK]
    with open(TMP_GT, "w") as f:
        for g in chunk: f.write(json.dumps(g) + "\n")
    if os.path.exists(TMP_OUT): os.remove(TMP_OUT)
    rc = subprocess.run([sys.executable, "scripts/run_generate_predictions.py",
        "--split", "test", "--adapter-path", ADAPTER,
        "--ground-truth", TMP_GT, "--output", TMP_OUT,
        "--max-new-tokens", "768"]).returncode
    if rc != 0 or not os.path.exists(TMP_OUT):
        print(f"[resume] chunk failed (rc={rc}); rerun this script to retry", flush=True); sys.exit(1)
    with open(OUT, "a") as fout, open(TMP_OUT) as fin:
        for l in fin:
            if l.strip(): fout.write(l.rstrip("\n") + "\n")
    print(f"[resume] committed +{len(chunk)}", flush=True)
