#!/usr/bin/env python3
# Finalize v4 SFT: clean hazards to the 7-class set, keep all positive frames,
# cap no_hazard-only frames to ~NH_FRAC, rewrite sft_train_ready_v4.
import sys, json, random, collections
from pathlib import Path
sys.path.insert(0, "src")
from drivesense.utils.config import load_config
from drivesense.data.annotation import SFTDataFormatter

VALID7 = {"occluded_pedestrian", "jaywalking", "cyclist_proximity",
          "construction_zone", "high_density", "unusual_object", "no_hazard"}
NH_FRAC = 0.15
SEED = 42
SRC = "/workspace/v4/annotated/annotated_manifest.json"
OUTDIR = Path("/workspace/v4/annotated")
SFT_OUT = "/workspace/sft_train_ready_v4"

def clean_hz(hz):
    out = []
    if isinstance(hz, list):
        for h in hz:
            if isinstance(h, dict) and h.get("label") in VALID7 \
               and isinstance(h.get("bbox_2d"), list) and len(h["bbox_2d"]) == 4:
                out.append(h)
    return out

src = json.load(open(SRC))
positives, negatives = [], []
for fr in src:
    ann = dict(fr["annotation"])
    hz = clean_hz(ann.get("hazards"))
    real = [h for h in hz if h.get("label") != "no_hazard"]
    fr2 = dict(fr)
    if real:
        ann["hazards"] = real
        fr2["annotation"] = ann; positives.append(fr2)
    else:
        nh = [h for h in hz if h.get("label") == "no_hazard"][:1] or [{
            "bbox_2d": [0, 0, 1000, 1000], "label": "no_hazard", "severity": "low",
            "reasoning": "No safety-critical hazard from the target classes is present.",
            "action": "Proceed with normal caution."}]
        ann["hazards"] = nh
        fr2["annotation"] = ann; negatives.append(fr2)

random.seed(SEED); random.shuffle(negatives)
keep_neg = min(len(negatives), round(NH_FRAC / (1 - NH_FRAC) * len(positives)))
final = positives + negatives[:keep_neg]
random.shuffle(final)

man = OUTDIR / "annotated_manifest_final.json"
json.dump(final, open(man, "w"), ensure_ascii=False, indent=2)

cfg = load_config("configs/data.yaml")
SFTDataFormatter(cfg).format_dataset(man, Path(SFT_OUT))

lab = collections.Counter()
hzc = []
for fr in final:
    hs = fr["annotation"]["hazards"]
    hzc.append(len(hs))
    for h in hs:
        lab[h["label"]] += 1
rep = {"frames_final": len(final), "positive": len(positives),
       "negative_available": len(negatives), "negative_kept": keep_neg,
       "nohazard_frac": round(keep_neg / len(final), 3),
       "avg_hazards_per_frame": round(sum(hzc) / len(final), 2),
       "hazard_label_distribution": dict(lab.most_common()),
       "sft_out": SFT_OUT}
json.dump(rep, open(OUTDIR / "v4_sft_final_report.json", "w"), indent=2)
print(json.dumps(rep, indent=2))
