#!/usr/bin/env python3
# v4 auto-labeler: Sonnet-5 vision via Batch API (-50%), constrained to v3's 7-class set.
# Reuses repo AnnotationPromptBuilder + validator + SFTDataFormatter. Logs gate stats + cost.
import sys, json, time, base64, argparse
from pathlib import Path
sys.path.insert(0, "src")
from drivesense.utils.config import load_config
import drivesense.data.annotation as A
from drivesense.data.annotation import AnnotationPromptBuilder, SFTDataFormatter, _load_manifest

BANNED = ("adverse_weather", "emergency_vehicle")
CHUNK = 700

GUIDANCE = (
    "\n\n## CRITICAL LABELING RULES\n"
    "- Box only DISCRETE, LOCALIZED hazards from the allowed classes.\n"
    "- NEVER create a hazard for environmental conditions themselves: rain, wet road, glare, "
    "headlight glare, fog, or darkness are NOT hazards and must NEVER be boxed. Never output a "
    "near-full-frame box such as [0,0,1000,1000].\n"
    "- Do NOT label ordinary vehicles (cars, trucks, buses) that are merely driving or stopped in "
    "normal traffic. A single lead car is NOT high_density; use high_density ONLY when 5+ agents "
    "crowd the scene.\n"
    "- 'unusual_object' is ONLY for genuinely out-of-place road objects (debris, fallen object, "
    "animal, object blocking a lane) - never a vehicle, never weather, never darkness.\n"
    "- Prefer occluded_pedestrian / jaywalking / cyclist_proximity / construction_zone / "
    "high_density when they truly apply. If NO allowed hazard is genuinely present, output exactly "
    "one hazard with label 'no_hazard'.\n"
    "- Every bounding box MUST have x2 > x1 and y2 > y1. If an object is cut off at an edge, keep "
    "the near edge at least 15 units inside the frame (e.g., 985 not 1000).\n"
)

def constrain(t):
    for b in BANNED:
        t = t.replace(", " + b, "").replace(b + ", ", "").replace(b, "")
    return t

def repair_box(b):
    if not (isinstance(b, list) and len(b) == 4):
        return b
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in b]
    except Exception:
        return b
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    x1, x2 = max(0, min(1000, x1)), max(0, min(1000, x2))
    y1, y2 = max(0, min(1000, y1)), max(0, min(1000, y2))
    if x2 - x1 < 5:
        x1, x2 = (x2 - 15, x2) if x2 >= 995 else (x1, x1 + 15)
    if y2 - y1 < 5:
        y1, y2 = (y2 - 15, y2) if y2 >= 995 else (y1, y1 + 15)
    return [max(0, x1), max(0, y1), min(1000, x2), min(1000, y2)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/workspace/v4/v4_manifest.jsonl")
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--tag", default="full")
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--out", default="/workspace/v4/annotated")
    ap.add_argument("--sft-out", default="/workspace/sft_train_ready_v4")
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()

    cfg = load_config(a.config)
    OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
    IDF = OUT / ("batch_ids_%s.json" % a.tag)
    pb = AnnotationPromptBuilder()
    V = next(v for v in vars(A).values() if isinstance(v, type)
             and hasattr(v, "validate_annotation") and hasattr(v, "parse_llm_response"))

    frames = _load_manifest(Path(a.manifest))
    if a.n and a.n > 0:
        frames = frames[:a.n]
    order = [f["frame_id"] for f in frames]
    fmap = {f["frame_id"]: f for f in frames}
    print("model=%s frames=%d tag=%s mock=%s" % (a.model, len(frames), a.tag, a.mock), flush=True)

    raw, tin, tout = {}, 0, 0
    if a.mock:
        canned = json.dumps({"hazards": [{"bbox_2d": [120, 210, 340, 640], "label": "jaywalking",
            "severity": "high", "reasoning": "Pedestrian entering the roadway in low visibility; "
            "elevated collision risk for the ego vehicle.", "action": "Brake and prepare to stop."}],
            "scene_summary": "Mock.", "ego_context": {"weather": "rain", "time_of_day": "night", "road_type": "urban"}})
        for fid in order:
            raw[fid] = canned
    else:
        import anthropic
        client = anthropic.Anthropic()
        reqs = []
        for i, f in enumerate(frames):
            system, user = pb.build_annotation_prompt(f)
            system, user = constrain(system), constrain(user) + GUIDANCE
            content = []
            ip = f.get("image_path")
            if ip and Path(ip).exists():
                data = base64.standard_b64encode(Path(ip).read_bytes()).decode()
                mt = "image/png" if ip.lower().endswith(".png") else "image/jpeg"
                content.append({"type": "image", "source": {"type": "base64", "media_type": mt, "data": data}})
            content.append({"type": "text", "text": user})
            reqs.append({"custom_id": "r%d" % i, "params": {"model": a.model,
                "max_tokens": a.max_tokens, "system": system,
                "messages": [{"role": "user", "content": content}]}})  # no temperature: Sonnet 5 rejects it
        if IDF.exists():
            batch_ids = json.loads(IDF.read_text()); print("resuming", batch_ids, flush=True)
        else:
            batch_ids = []
            for j in range(0, len(reqs), CHUNK):
                b = client.messages.batches.create(requests=reqs[j:j + CHUNK])
                batch_ids.append(b.id); print("submitted chunk %d -> %s" % (j // CHUNK, b.id), flush=True)
            IDF.write_text(json.dumps(batch_ids))
        while True:
            stt = [client.messages.batches.retrieve(b).processing_status for b in batch_ids]
            print("statuses", stt, flush=True)
            if all(s == "ended" for s in stt):
                break
            time.sleep(30)
        for bid in batch_ids:
            for r in client.messages.batches.results(bid):
                i = int(r.custom_id[1:]); fid = order[i]
                if r.result.type == "succeeded":
                    m = r.result.message
                    tin += m.usage.input_tokens; tout += m.usage.output_tokens
                    raw[fid] = next((bl.text for bl in m.content if getattr(bl, "type", None) == "text"), "")

    annotated = []
    st = {"total": len(order), "parsed": 0, "valid_clean": 0, "auto_fixed": 0,
          "still_invalid": 0, "dropped_banned_hazards": 0, "boxes_repaired": 0,
          "empty_after_filter": 0, "failed": 0}
    ld = {}
    for fid in order:
        txt = raw.get(fid)
        if not txt:
            st["failed"] += 1; continue
        ann = V.parse_llm_response(txt)
        if ann is None:
            st["failed"] += 1; continue
        st["parsed"] += 1
        hz = ann.get("hazards")
        if isinstance(hz, list):
            kept = [h for h in hz if str(h.get("label", "")).lower() not in BANNED]
            st["dropped_banned_hazards"] += len(hz) - len(kept)
            ann["hazards"] = kept
        for h in ann.get("hazards", []) or []:
            if isinstance(h, dict) and "bbox_2d" in h:
                nb = repair_box(h["bbox_2d"])
                if nb != h["bbox_2d"]:
                    st["boxes_repaired"] += 1
                h["bbox_2d"] = nb
        ok, _ = V.validate_annotation(ann)
        if ok:
            st["valid_clean"] += 1
        else:
            ann = V.fix_common_issues(ann)
            ok2, _ = V.validate_annotation(ann)
            if ok2:
                st["auto_fixed"] += 1
            else:
                st["still_invalid"] += 1
                if not ann.get("hazards"):
                    st["empty_after_filter"] += 1
        for h in ann.get("hazards", []):
            ld[h.get("label", "?")] = ld.get(h.get("label", "?"), 0) + 1
        fr = dict(fmap[fid]); fr["annotation"] = ann
        annotated.append(fr)

    man = OUT / "annotated_manifest.json"
    json.dump(annotated, open(man, "w"), ensure_ascii=False, indent=2)
    SFTDataFormatter(cfg).format_dataset(man, Path(a.sft_out))
    st["label_distribution"] = dict(sorted(ld.items(), key=lambda kv: -kv[1]))
    st["tokens_in"] = tin; st["tokens_out"] = tout
    st["batch_cost_usd"] = round(tin / 1e6 + tout * 5.0 / 1e6, 4)
    st["sft_out"] = a.sft_out
    json.dump(st, open(OUT / ("v4_annotation_report_%s.json" % a.tag), "w"), indent=2)
    print("[report]", json.dumps(st, indent=2), flush=True)

if __name__ == "__main__":
    main()
