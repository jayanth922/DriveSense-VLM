import os, sys, json, time
from pathlib import Path
sys.path.insert(0, 'src')
from drivesense.utils.config import load_config, merge_configs
from drivesense.eval.grounding import GroundingEvaluator
from drivesense.eval.reasoning import (
    _build_judge_prompt, _parse_judge_response, LLMJudge, compute_reasoning_metrics,
)
import anthropic

N   = int(sys.argv[1]) if len(sys.argv) > 1 else 0          # <=0 => all pairs
TAG = sys.argv[2] if len(sys.argv) > 2 else "run"
OUT = Path("/workspace/v3_l2"); OUT.mkdir(exist_ok=True)
IDF = OUT / f"batch_id_{TAG}.txt"
PRED = Path("/workspace/v3_regen/test_pred_full.jsonl")
GT   = Path("/workspace/sft_ready_v3_merged/sft_test_enriched.jsonl")
DIMS = ["correctness", "completeness", "action_relevance"]

c = Path("configs")
cfg = merge_configs(load_config(c/"model.yaml"), load_config(c/"data.yaml"),
                    load_config(c/"training.yaml"), load_config(c/"eval.yaml"))
model = cfg.get("reasoning", {}).get("judge", {}).get("model", "claude-sonnet-5")
SYS = LLMJudge.JUDGE_SYSTEM_PROMPT
DIMDESC = LLMJudge.JUDGE_DIMENSIONS

ev = GroundingEvaluator(cfg)
preds = ev.load_predictions(PRED)
gt = ev.load_ground_truth(GT)
gtby = {g["frame_id"]: g for g in gt if "frame_id" in g}
pairs = []
for p in preds:
    if p.get("parse_failure", False):
        continue
    g = gtby.get(p.get("frame_id", ""))
    if g:
        pairs.append((p, g))
if N > 0:
    pairs = pairs[:N]

# deterministic index -> (frame_id, dimension); rebuilt identically on resume
order = [(p["frame_id"], d) for (p, g) in pairs for d in DIMS]
print(f"model={model}  pairs={len(pairs)}  calls={len(order)}", flush=True)

client = anthropic.Anthropic()

if IDF.exists():
    batch_id = IDF.read_text().strip()
    print("resuming batch", batch_id, flush=True)
else:
    reqs, k = [], 0
    for (p, g) in pairs:
        for d in DIMS:
            prompt = _build_judge_prompt(p, g, d, DIMDESC.get(d, d))
            reqs.append({
                "custom_id": f"r{k}",
                "params": {"model": model, "max_tokens": 256, "system": SYS,
                           "messages": [{"role": "user", "content": prompt}]},
            })
            k += 1
    print(f"submitting {len(reqs)} requests...", flush=True)
    batch = client.messages.batches.create(requests=reqs)
    batch_id = batch.id
    IDF.write_text(batch_id)
    print("batch id", batch_id, flush=True)

while True:
    b = client.messages.batches.retrieve(batch_id)
    print("status", b.processing_status, "counts", b.request_counts, flush=True)
    if b.processing_status == "ended":
        break
    time.sleep(20)

scores_by_frame, tin, tout, ok, err = {}, 0, 0, 0, 0
for r in client.messages.batches.results(batch_id):
    idx = int(r.custom_id[1:]); fid, d = order[idx]
    if r.result.type == "succeeded":
        msg = r.result.message
        tin += msg.usage.input_tokens; tout += msg.usage.output_tokens
        _txt = next((b.text for b in msg.content if getattr(b,"type",None)=="text"), "")
        parsed = _parse_judge_response(_txt)
        just = parsed.get("justification", "")
        if isinstance(just, str) and just.startswith("<parse error"):
            err += 1; continue
        scores_by_frame.setdefault(fid, {})[d] = {"score": parsed.get("score"), "justification": just}
        ok += 1
    else:
        err += 1
print(f"results: ok={ok} err={err}", flush=True)

judge_results = [{"frame_id": fid, "scores": sc} for fid, sc in scores_by_frame.items()]
metrics = compute_reasoning_metrics(judge_results)
metrics.pop("judge_results_raw", None)
(OUT/f"l2_metrics_{TAG}.json").write_text(json.dumps(metrics, indent=2))
(OUT/f"l2_judge_results_{TAG}.json").write_text(json.dumps(judge_results, indent=2))

cost = tin*1.0/1e6 + tout*5.0/1e6       # Sonnet 5 batch: $1 in / $5 out per M
print("TOKENS in/out:", tin, tout, flush=True)
print(f"BATCH COST (Sonnet5 batch $1/$5): ${round(cost,4)}", flush=True)
print(json.dumps(metrics, indent=2), flush=True)
