#!/usr/bin/env python3
"""Task 3 de-confound — manifest reconstruction + pre-flight gate.

The historical v3/v4 per-frame training data did not survive the RunPod
deletion, so this script rebuilds a *faithful-family* dataset directly from the
surviving nuScenes trainval tables + CAM_FRONT images. Two modes:

  --preflight   $0 gate. Resolves the fixed 1,041 test frame_ids (from
                results/v4/test_pred_full.jsonl) against nuScenes and prints
                PASS (UNRESOLVED=0) or FAIL. NOTHING may be spent until PASS.

  --build       Reconstructs, GT-boxes, and writes every manifest:
                  eval_gt/   – the 1,041 test frames, GT boxes + templated
                               text (FREE), formatted to sft_test.jsonl for
                               use as --ground-truth at eval time.
                  base_val/  – ~7,228 train + ~889 val keyframes (any weather,
                               test scenes excluded), GT-boxed, NO describe
                               text yet -> feed to describe_manifest.py.
                  targeted/  – ~2,231 adverse (rain|fog|night) keyframes ->
                               feed to the FM labeler AND the GT-box arm.
                Also writes cost_estimate.json (API $, gated <= $40).

frame_id convention (matches build_v4_manifest.py): frame_id IS the CAM_FRONT
sample_data token. Token bridge to GT boxes:
    frame_id -> nusc.get("sample_data", frame_id)["sample_token"] -> sample.

Env overrides (all have RunPod defaults):
  REPO        repo root                (default /workspace/DriveSense)
  NUSC_ROOT   nuScenes dataroot        (default /workspace/nuscenes)
  WORK        output dir               (default /workspace/deconfound_work)
  TEST_PRED   1,041-id source jsonl    (default $REPO/results/v4/test_pred_full.jsonl)
  N_BASE / N_VAL / N_TARGETED  set sizes (defaults 7228 / 889 / 2231)
  NH_FRAC     no_hazard-only cap       (default 0.15)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(os.environ.get("REPO", "/workspace/DriveSense"))
NUSC_ROOT = Path(os.environ.get("NUSC_ROOT", "/workspace/nuscenes"))
WORK = Path(os.environ.get("WORK", "/workspace/deconfound_work"))
TEST_PRED = Path(os.environ.get("TEST_PRED", REPO / "results/v4/test_pred_full.jsonl"))

N_BASE = int(os.environ.get("N_BASE", "7228"))
N_VAL = int(os.environ.get("N_VAL", "889"))
N_TARGETED = int(os.environ.get("N_TARGETED", "2231"))
NH_FRAC = float(os.environ.get("NH_FRAC", "0.15"))
SEED = 42

# per-frame API rates from the cost model (Sonnet vision via Batch API, -50%)
RATE_DESCRIBE_DOWNSIZED = 0.0027   # $/frame, GT-describe at --downsize 768
RATE_FM_FULLRES = 0.00385          # $/frame, FM labeling at full resolution
COST_GATE = 40.0

sys.path.insert(0, str(REPO / "src"))


# --------------------------------------------------------------------------- #
# nuScenes bootstrap (numpy ABI + MapMask stub, per the Kaggle/RunPod fixes)
# --------------------------------------------------------------------------- #
def load_nusc():
    import nuscenes.nuscenes as _nn
    # tables ship no map PNGs -> stub MapMask so NuScenes() does not try to load them
    _nn.MapMask = type("_M", (), {"__init__": lambda s, *a, **k: None})
    from nuscenes.nuscenes import NuScenes
    print(f"[nusc] loading v1.0-trainval from {NUSC_ROOT} ...", flush=True)
    return NuScenes(version="v1.0-trainval", dataroot=str(NUSC_ROOT), verbose=False)


def read_test_ids() -> list[str]:
    ids = []
    for line in TEST_PRED.open():
        line = line.strip()
        if line:
            ids.append(json.loads(line)["frame_id"])
    return ids


def resolve_frame(nusc, fid: str):
    """Return (sample_token, how) or (None, None). Tries sample_data, then
    sample token directly, then CAM_FRONT basename lookup."""
    try:
        sd = nusc.get("sample_data", fid)
        return sd["sample_token"], "sample_data"
    except Exception:
        pass
    try:
        nusc.get("sample", fid)
        return fid, "sample"
    except Exception:
        pass
    return None, None


def scene_conditions(desc: str):
    dl = (desc or "").lower()
    tod = "night" if "night" in dl else "day"
    weather = "rain" if "rain" in dl else ("fog" if "fog" in dl else "clear")
    return weather, tod


# --------------------------------------------------------------------------- #
# templated describe text for the FREE eval-GT (L1/L4 do not score prose)
# --------------------------------------------------------------------------- #
_SEV = {
    "occluded_pedestrian": "high", "jaywalking": "high", "cyclist_proximity": "high",
    "construction_zone": "medium", "unusual_object": "medium",
    "high_density": "low", "no_hazard": "low",
}
_REASON = {
    "occluded_pedestrian": "A partially occluded pedestrian is present and may enter the ego path.",
    "jaywalking": "A pedestrian is crossing outside a marked crosswalk near the ego path.",
    "cyclist_proximity": "A cyclist is close to the ego lane and may change position.",
    "construction_zone": "Construction objects narrow or obstruct the drivable path.",
    "unusual_object": "An out-of-place object on the road could obstruct the lane.",
    "high_density": "Many agents crowd the scene, raising the chance of sudden conflicts.",
    "no_hazard": "No safety-critical hazard from the target classes is present.",
}
_ACTION = {
    "no_hazard": "Proceed with normal caution.",
}


def templated_annotation(hazards, weather, tod):
    out = []
    for h in hazards:
        lab = h["label"]
        item = {"label": lab, "severity": _SEV.get(lab, "medium"),
                "reasoning": _REASON.get(lab, "Potential hazard in the ego path."),
                "action": _ACTION.get(lab, "Slow down and be prepared to stop.")}
        if "bbox_2d" in h:
            item["bbox_2d"] = h["bbox_2d"]
        out.append(item)
    return {"hazards": out,
            "scene_summary": f"Reconstructed {weather}/{tod} nuScenes keyframe.",
            "ego_context": {"weather": weather, "time_of_day": tod, "road_type": "urban"}}


def gt_hazards(nusc, sample_token):
    from drivesense.data.box_sourcing import source_boxes_for_frame
    kept, _ = source_boxes_for_frame(nusc, sample_token)
    # keep only fields the SFT target needs; box-exempt (high_density/no_hazard) carry no bbox
    clean = []
    for h in kept:
        item = {"label": h["label"]}
        if "bbox_2d" in h:
            item["bbox_2d"] = h["bbox_2d"]
        clean.append(item)
    return clean


# --------------------------------------------------------------------------- #
# PRE-FLIGHT
# --------------------------------------------------------------------------- #
def preflight():
    nusc = load_nusc()
    ids = read_test_ids()
    resolved, unresolved, via = 0, [], Counter()
    scenes = set()
    for fid in ids:
        st, how = resolve_frame(nusc, fid)
        if st is None:
            unresolved.append(fid)
            continue
        resolved += 1
        via[how] += 1
        try:
            scenes.add(nusc.get("sample", st)["scene_token"])
        except Exception:
            pass

    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "test_scene_tokens.json").write_text(json.dumps(sorted(scenes)))
    print(f"[preflight] test_ids={len(ids)} RESOLVED={resolved} "
          f"UNRESOLVED={len(unresolved)} via={dict(via)} test_scenes={len(scenes)}")
    if unresolved:
        (WORK / "unresolved_ids.json").write_text(json.dumps(unresolved))
        print("PREFLIGHT: FAIL — spend nothing. First 10 unresolved:", unresolved[:10])
        print("The 1,041 test ids could not be mapped to nuScenes. Fix the id->token "
              "mapping before --build. Unresolved ids saved to", WORK / "unresolved_ids.json")
        sys.exit(1)
    print(f"PREFLIGHT: PASS (UNRESOLVED=0). {len(scenes)} test scenes saved for exclusion. "
          f"Safe to run --build.")


# --------------------------------------------------------------------------- #
# BUILD
# --------------------------------------------------------------------------- #
def _scene_diverse(records, n, seed=SEED):
    """Round-robin over scenes for a scene-diverse subsample (build_v4 style)."""
    if n <= 0 or n >= len(records):
        return list(records)
    rng = random.Random(seed)
    byscene = defaultdict(list)
    for r in records:
        byscene[r["scene_token"]].append(r)
    for st in byscene:
        rng.shuffle(byscene[st])
    order = list(byscene)
    rng.shuffle(order)
    picked = []
    while len(picked) < n:
        progressed = False
        for st in order:
            if byscene[st]:
                picked.append(byscene[st].pop())
                progressed = True
                if len(picked) >= n:
                    break
        if not progressed:
            break
    return picked


def _cap_no_hazard(records, frac=NH_FRAC, seed=SEED):
    """Keep every positive frame; cap no_hazard-only frames to `frac` of the set."""
    pos, neg = [], []
    for r in records:
        (pos if any(h["label"] != "no_hazard" for h in r["_hz"]) else neg).append(r)
    rng = random.Random(seed)
    rng.shuffle(neg)
    keep_neg = min(len(neg), round(frac / (1 - frac) * len(pos))) if pos else 0
    out = pos + neg[:keep_neg]
    rng.shuffle(out)
    return out, len(pos), keep_neg


def build():
    nusc = load_nusc()
    WORK.mkdir(parents=True, exist_ok=True)

    # test scenes to exclude (written by --preflight; recompute if absent)
    tsf = WORK / "test_scene_tokens.json"
    if tsf.exists():
        test_scenes = set(json.loads(tsf.read_text()))
    else:
        test_scenes = set()
        for fid in read_test_ids():
            st, _ = resolve_frame(nusc, fid)
            if st:
                try:
                    test_scenes.add(nusc.get("sample", st)["scene_token"])
                except Exception:
                    pass
    print(f"[build] excluding {len(test_scenes)} test scenes")

    # ---- 1. EVAL GT: the exact 1,041 test frames, GT boxes + templated text ----
    test_ids = read_test_ids()
    eval_frames = []
    miss = 0
    for fid in test_ids:
        st, _ = resolve_frame(nusc, fid)
        if st is None:
            miss += 1
            continue
        sd = nusc.get("sample_data", nusc.get("sample", st)["data"]["CAM_FRONT"])
        img = NUSC_ROOT / sd["filename"]
        scene = nusc.get("scene", nusc.get("sample", st)["scene_token"])
        w, t = scene_conditions(scene.get("description", ""))
        hz = gt_hazards(nusc, st)
        if not hz:
            hz = [{"label": "no_hazard"}]
        eval_frames.append({
            "frame_id": fid, "image_path": str(img), "split": "test",
            "scene_token": scene["token"], "weather": w, "time_of_day": t,
            "location": None, "annotation": templated_annotation(hz, w, t),
        })
    _write_manifest_and_sft(eval_frames, WORK / "eval_gt", tag="eval_gt")
    print(f"[eval_gt] frames={len(eval_frames)} unresolved={miss}")

    # ---- 2. Sweep all non-test keyframes once, GT-box them ----
    pool = []
    n_scan = n_ondisk = 0
    for samp in nusc.sample:
        if samp["scene_token"] in test_scenes:
            continue
        n_scan += 1
        sd = nusc.get("sample_data", samp["data"]["CAM_FRONT"])
        img = NUSC_ROOT / sd["filename"]
        if not img.exists():
            continue
        n_ondisk += 1
        scene = nusc.get("scene", samp["scene_token"])
        w, t = scene_conditions(scene.get("description", ""))
        hz = gt_hazards(nusc, samp["token"])
        if not hz:
            hz = [{"label": "no_hazard"}]
        pool.append({
            "frame_id": samp["data"]["CAM_FRONT"], "image_path": str(img),
            "scene_token": samp["scene_token"], "weather": w, "time_of_day": t,
            "location": None, "_hz": hz,
        })
    print(f"[pool] non-test keyframes scanned={n_scan} on_disk={n_ondisk} boxed={len(pool)}")

    adverse = [r for r in pool if r["weather"] in ("rain", "fog") or r["time_of_day"] == "night"]
    print(f"[pool] adverse={len(adverse)}")

    # ---- 3. TARGETED (~2,231 adverse, scene-diverse) ----
    targeted = _scene_diverse(adverse, N_TARGETED)
    targeted_ids = {r["frame_id"] for r in targeted}
    _write_targeted(targeted, WORK / "targeted")
    print(f"[targeted] picked={len(targeted)} "
          f"wt={dict(Counter((r['weather'], r['time_of_day']) for r in targeted))}")

    # ---- 4. BASE + VAL from the remaining (non-targeted) frames ----
    rest = [r for r in pool if r["frame_id"] not in targeted_ids]
    rest_capped, npos, nneg = _cap_no_hazard(rest)
    base_val = _scene_diverse(rest_capped, N_BASE + N_VAL)
    rng = random.Random(SEED)
    rng.shuffle(base_val)
    for i, r in enumerate(base_val):
        r["split"] = "val" if i < N_VAL else "train"
        r["annotation"] = templated_annotation(r["_hz"], r["weather"], r["time_of_day"])
        r.pop("_hz", None)
    _write_manifest_only(base_val, WORK / "base_val", tag="base_val")
    print(f"[base_val] total={len(base_val)} train={sum(r['split']=='train' for r in base_val)} "
          f"val={sum(r['split']=='val' for r in base_val)} (pos_pool={npos} nh_kept={nneg})")

    # ---- 5. cost estimate (gate) ----
    n_describe = N_BASE + N_VAL + N_TARGETED  # base+val + targeted-GT describe (~finalized subset)
    api_describe = n_describe * RATE_DESCRIBE_DOWNSIZED
    api_fm = len(targeted) * RATE_FM_FULLRES
    total = round(api_describe + api_fm, 2)
    est = {
        "describe_frames": n_describe, "describe_rate": RATE_DESCRIBE_DOWNSIZED,
        "describe_usd": round(api_describe, 2),
        "fm_label_frames": len(targeted), "fm_rate": RATE_FM_FULLRES,
        "fm_label_usd": round(api_fm, 2),
        "total_api_usd": total, "gate_usd": COST_GATE,
        "gate_pass": total <= COST_GATE,
        "note": "targeted-GT describe is over the finalized subset; here estimated at "
                "N_TARGETED as an upper bound.",
    }
    (WORK / "cost_estimate.json").write_text(json.dumps(est, indent=2))
    print("[cost]", json.dumps(est, indent=2))
    if not est["gate_pass"]:
        print(f"COST GATE: FAIL — estimate ${total} > ${COST_GATE}. Do NOT run describe/label.")
        sys.exit(2)
    print(f"COST GATE: PASS — estimate ${total} <= ${COST_GATE}.")
    print("\nBUILD COMPLETE. Next: describe_manifest.py (base_val + targeted-GT), "
          "then the FM labeler on targeted/, then build_arms.py.")


# --------------------------------------------------------------------------- #
# manifest writers (match the record shape SFTDataFormatter expects)
# --------------------------------------------------------------------------- #
def _sft_format(manifest_json: Path, out_dir: Path):
    from drivesense.utils.config import load_config
    from drivesense.data.annotation import SFTDataFormatter
    cfg = load_config(str(REPO / "configs/data.yaml"))
    SFTDataFormatter(cfg).format_dataset(manifest_json, out_dir)


def _write_manifest_and_sft(frames, out_dir: Path, tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    man = out_dir / "annotated_manifest.json"
    json.dump(frames, man.open("w"), ensure_ascii=False, indent=2)
    _sft_format(man, out_dir)
    (out_dir / f"{tag}_summary.json").write_text(json.dumps({"frames": len(frames)}, indent=2))


def _write_manifest_only(frames, out_dir: Path, tag: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    man = out_dir / "annotated_manifest.json"
    json.dump(frames, man.open("w"), ensure_ascii=False, indent=2)
    (out_dir / f"{tag}_summary.json").write_text(json.dumps({
        "frames": len(frames),
        "wt": {f"{w}|{t}": c for (w, t), c in
               Counter((r["weather"], r["time_of_day"]) for r in frames).items()},
    }, indent=2))


def _write_targeted(frames, out_dir: Path):
    """Targeted frames go to the FM labeler as a JSONL manifest (no annotation yet)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "v4_manifest.jsonl").open("w") as f:
        for r in frames:
            rec = {k: r[k] for k in ("frame_id", "image_path", "scene_token",
                                     "weather", "time_of_day", "location")}
            rec["split"] = "train"
            rec["source"] = "nuscenes"
            rec["scene_description"] = ""
            rec["_gt_hazards"] = r["_hz"]  # kept so build_arms can build the GT arm for free
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Task 3 de-confound reconstruction + preflight")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--preflight", action="store_true", help="$0 gate: resolve the 1,041 test ids")
    g.add_argument("--build", action="store_true", help="reconstruct all manifests + cost_estimate")
    a = ap.parse_args()
    if a.preflight:
        preflight()
    else:
        build()


if __name__ == "__main__":
    main()
