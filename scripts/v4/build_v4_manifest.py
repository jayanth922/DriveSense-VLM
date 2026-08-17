#!/usr/bin/env python3
import json, os, random
from pathlib import Path
from collections import defaultdict, Counter
import ijson

NUSC = Path("/workspace/nuscenes/v1.0-trainval")
IMG_DIR = Path("/workspace/nuscenes/samples/CAM_FRONT")
V4 = Path("/workspace/v4")
FRAMES_TXT = V4 / "v4_frames.txt"
N_TARGET = int(os.environ.get("N_TARGET", "1600"))
SEED = 42

HOLDOUT_FILES = [
    "/workspace/sft_train_ready_v3/sft_test.jsonl",
    "/workspace/sft_train_ready_v3/sft_val.jsonl",
    "/workspace/sft_ready_v3_merged/sft_test_enriched.jsonl",
    "/workspace/sft_ready_v3_merged/sft_val_enriched.jsonl",
]
V3_TRAIN = "/workspace/sft_train_ready_v3/sft_train.jsonl"

# 1. target basenames on disk
want = {l.strip() for l in FRAMES_TXT.read_text().splitlines() if l.strip()}
on_disk = {b for b in want if (IMG_DIR / b).exists()}
print(f"[frames] unique={len(want)} on_disk={len(on_disk)} missing={len(want)-len(on_disk)}")

# 2. stream sample_data.json (1.3GB) -> basename -> (sd_token, sample_token)
print("[sample_data] streaming...")
bn2sd, bn2sample = {}, {}
n = 0
with open(NUSC / "sample_data.json", "rb") as f:
    for item in ijson.items(f, "item"):
        n += 1
        bn = item.get("filename", "").rsplit("/", 1)[-1]
        if bn in on_disk:
            bn2sd[bn] = item.get("token")
            bn2sample[bn] = item.get("sample_token")
        if n % 2_000_000 == 0:
            print(f"  scanned {n:,} rows, matched {len(bn2sd)}")
print(f"[sample_data] scanned {n:,} rows, matched {len(bn2sd)}/{len(on_disk)}")

# 3-4. small tables
s2scene = {s["token"]: s["scene_token"] for s in json.load(open(NUSC / "sample.json"))}
scene_desc = {s["token"]: (s.get("name",""), s.get("description","")) for s in json.load(open(NUSC / "scene.json"))}

# 5. holdout scenes (v3 test+val) + v3 train dedup ids
def scene_tokens(p):
    p = Path(p); out = set()
    if p.exists():
        for line in p.open():
            line = line.strip()
            if line:
                st = json.loads(line).get("scene_token")
                if st: out.add(st)
    return out
holdout = set().union(*[scene_tokens(p) for p in HOLDOUT_FILES]) if HOLDOUT_FILES else set()
v3_train_ids = scene_tokens.__self__ if False else set()
p = Path(V3_TRAIN)
if p.exists():
    for line in p.open():
        line = line.strip()
        if line: v3_train_ids.add(json.loads(line).get("frame_id"))
print(f"[holdout] scenes={len(holdout)}  v3_train_ids={len(v3_train_ids)}")

# 6. build candidate records
records = []
d_no_scene = d_leak = d_dup = 0
for bn in on_disk:
    sd, samp = bn2sd.get(bn), bn2sample.get(bn)
    scene = s2scene.get(samp) if samp else None
    if scene is None: d_no_scene += 1; continue
    if scene in holdout: d_leak += 1; continue
    if sd in v3_train_ids: d_dup += 1; continue
    name, desc = scene_desc.get(scene, ("", ""))
    dl = desc.lower()
    tod = "night" if "night" in dl else "day"
    weather = "rain" if "rain" in dl else ("fog" if "fog" in dl else "clear")
    signals = []
    if weather != "clear": signals.append(f"adverse_weather:{weather}")
    if tod == "night": signals.append("time:night")
    records.append({
        "frame_id": sd, "image_path": str(IMG_DIR / bn), "source": "nuscenes",
        "split": "train", "scene_token": scene, "scene_name": name,
        "weather": weather, "time_of_day": tod, "road_type": "urban",
        "source_metadata": {"rarity_signals": signals, "description": desc},
        "basename": bn,
    })
print(f"[filter] candidates={len(records)} dropped: no_scene={d_no_scene} leak={d_leak} dup={d_dup}")

# 7. scene-diverse subsample
random.seed(SEED)
byscene = defaultdict(list)
for r in records: byscene[r["scene_token"]].append(r)
for st in byscene: random.shuffle(byscene[st])
order = list(byscene); random.shuffle(order)
picked = []
if N_TARGET <= 0 or N_TARGET >= len(records):
    picked = list(records)
else:
    while len(picked) < N_TARGET:
        progressed = False
        for st in order:
            if byscene[st]:
                picked.append(byscene[st].pop()); progressed = True
                if len(picked) >= N_TARGET: break
        if not progressed: break
print(f"[subsample] picked={len(picked)} across {len({r['scene_token'] for r in picked})} scenes")

# 8. write outputs
V4.mkdir(parents=True, exist_ok=True)
with (V4 / "v4_frames_meta.jsonl").open("w") as f:
    for r in records: f.write(json.dumps(r) + "\n")
with (V4 / "v4_manifest.jsonl").open("w") as f:
    for r in picked: f.write(json.dumps(r) + "\n")
wc = Counter((r["weather"], r["time_of_day"]) for r in picked)
summary = {
    "target_basenames": len(want), "on_disk": len(on_disk),
    "matched_sample_data": len(bn2sd), "candidates_after_filter": len(records),
    "dropped_no_scene": d_no_scene, "dropped_leak_v3_testval": d_leak,
    "dropped_dup_v3_train": d_dup, "holdout_scenes": len(holdout),
    "picked": len(picked), "picked_scenes": len({r['scene_token'] for r in picked}),
    "weather_time_breakdown": {f"{w}|{t}": c for (w, t), c in wc.items()},
}
json.dump(summary, open(V4 / "v4_manifest_summary.json", "w"), indent=2)
print("[summary]", json.dumps(summary, indent=2))
