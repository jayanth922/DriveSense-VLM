#!/usr/bin/env python3
import json, os, random
from pathlib import Path
from collections import defaultdict, Counter

V4 = Path("/workspace/v4"); META = V4 / "v4_frames_meta.jsonl"
N_TARGET = int(os.environ.get("N_TARGET", "1600")); SEED = 42

recs = [json.loads(l) for l in META.open() if l.strip()]
adverse = [r for r in recs if r["weather"] in ("rain","fog") or r["time_of_day"]=="night"]
print(f"[pool] total={len(recs)} adverse={len(adverse)}")

random.seed(SEED)
byscene = defaultdict(list)
for r in adverse: byscene[r["scene_token"]].append(r)
for st in byscene: random.shuffle(byscene[st])
order = list(byscene); random.shuffle(order)
picked = []
if N_TARGET<=0 or N_TARGET>=len(adverse): picked=list(adverse)
else:
    while len(picked)<N_TARGET:
        prog=False
        for st in order:
            if byscene[st]:
                picked.append(byscene[st].pop()); prog=True
                if len(picked)>=N_TARGET: break
        if not prog: break

for r in picked:
    r["scene_description"] = r.get("source_metadata",{}).get("description","")

with (V4/"v4_manifest.jsonl").open("w") as f:
    for r in picked: f.write(json.dumps(r)+"\n")
wc=Counter((r["weather"],r["time_of_day"]) for r in picked)
summary={"mode":"adverse_only","adverse_pool":len(adverse),"picked":len(picked),
         "picked_scenes":len({r["scene_token"] for r in picked}),
         "weather_time_breakdown":{f"{w}|{t}":c for (w,t),c in wc.items()}}
json.dump(summary, open(V4/"v4_manifest_summary.json","w"), indent=2)
print("[summary]", json.dumps(summary, indent=2))
