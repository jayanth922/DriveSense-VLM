#!/usr/bin/env python3
# Step 4: build v4 train set = v3 train + v4 adverse add; reuse v3 val/test verbatim.
# Enriches v4 lines with weather/time_of_day/scene_token, dedups by frame_id vs ALL
# v3 splits, and asserts zero train<->val/test leakage.
import json, collections
from pathlib import Path

V3 = "/workspace/sft_train_ready_v3"
V4ADD = "/workspace/sft_train_ready_v4/sft_train.jsonl"
MANI = "/workspace/v4/v4_manifest.jsonl"
OUT = Path("/workspace/sft_train_ready_v4_merged"); OUT.mkdir(parents=True, exist_ok=True)

def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]

def dump(rows, p):
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def labels(rows):
    c = collections.Counter()
    for r in rows:
        try:
            ann = json.loads(r["messages"][-1]["content"])
        except Exception:
            continue
        for h in ann.get("hazards", []):
            c[h.get("label")] += 1
    return c

v3_tr = load(f"{V3}/sft_train.jsonl")
v3_va = load(f"{V3}/sft_val.jsonl")
v3_te = load(f"{V3}/sft_test.jsonl")
add = load(V4ADD)
meta = {r["frame_id"]: r for r in load(MANI)}

v3_all_ids = {r.get("frame_id") for r in (v3_tr + v3_va + v3_te)}

enr, dropped = [], 0
for r in add:
    fid = r.get("frame_id")
    if fid in v3_all_ids:
        dropped += 1
        continue
    m = meta.get(fid, {})
    r = dict(r)
    r["split"] = "train"
    r["scene_token"] = m.get("scene_token")
    r["weather"] = m.get("weather")
    r["time_of_day"] = m.get("time_of_day")
    r["location"] = None
    enr.append(r)

train = v3_tr + enr
dump(train, OUT / "sft_train.jsonl")
dump(v3_va, OUT / "sft_val.jsonl")
dump(v3_te, OUT / "sft_test.jsonl")

tr_ids = {r.get("frame_id") for r in train}
va_ids = {r.get("frame_id") for r in v3_va}
te_ids = {r.get("frame_id") for r in v3_te}
leak_te = len(tr_ids & te_ids)
leak_va = len(tr_ids & va_ids)

add_wt = collections.Counter((r.get("weather"), r.get("time_of_day")) for r in enr)
rep = {
    "v3_train": len(v3_tr), "v4_add_input": len(add), "v4_add_dropped_dup": dropped,
    "v4_add_kept": len(enr), "v4_train_total": len(train),
    "val": len(v3_va), "test": len(v3_te),
    "leak_train_vs_test": leak_te, "leak_train_vs_val": leak_va,
    "add_weather_time": {f"{w}|{t}": c for (w, t), c in add_wt.items()},
    "add_label_dist": dict(labels(enr).most_common()),
    "merged_train_label_dist": dict(labels(train).most_common()),
    "out_dir": str(OUT),
}
json.dump(rep, open(OUT / "v4_merge_report.json", "w"), indent=2)
print(json.dumps(rep, indent=2))
assert leak_te == 0 and leak_va == 0, "!!! LEAKAGE DETECTED — do not train"
print("\nOK: zero train<->val/test leakage. v4 train set ready at", OUT)
