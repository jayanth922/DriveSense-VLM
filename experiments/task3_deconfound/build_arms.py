#!/usr/bin/env python3
"""Assemble the two SFT arms for the FM-vs-GT de-confound.

Both arms share an IDENTICAL base+val (GT-described reconstruction) and an
IDENTICAL set of targeted frame_ids (the FM-finalized ~1,442). They differ in
ONE thing: the targeted boxes — FM-labeled vs nuScenes-GT. Test is the fixed
1,041 eval set. Leakage (train ∩ val, train ∩ test) is asserted to be 0.

Two stages:

  prep      Read the FM-finalized targeted ids, and emit the targeted-GT
            annotated manifest (GT boxes from reconstruct's _gt_hazards) for
            exactly those ids -> describe it next with describe_manifest.py.

  assemble  Build arm_fm/ and arm_gt/ from:
              base_val/sft_train.jsonl + sft_val.jsonl   (shared)
              eval_gt/sft_test.jsonl                     (shared test)
              targeted_fm/sft_train.jsonl                (FM boxes, finalized)
              targeted_gt/sft_train.jsonl                (GT boxes, described)
            Aligns GT targeted to the FM finalized id set and asserts equality,
            then writes both arms and runs the leakage asserts.

Env: WORK (default /workspace/deconfound_work), REPO (default /workspace/DriveSense).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

WORK = Path(os.environ.get("WORK", "/workspace/deconfound_work"))


def load_jsonl(p: Path):
    return [json.loads(l) for l in p.open() if l.strip()]


def dump_jsonl(rows, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def ids_of(rows):
    return {r.get("frame_id") for r in rows}


# --------------------------------------------------------------------------- #
def prep():
    """Emit targeted-GT annotated manifest for the FM-finalized ids."""
    fm = load_jsonl(WORK / "targeted_fm" / "sft_train.jsonl")
    finalized = ids_of(fm)
    print(f"[prep] FM-finalized targeted frames = {len(finalized)}")

    # GT hazards were carried on the targeted manifest by reconstruct.py
    gt_src = {r["frame_id"]: r for r in load_jsonl(WORK / "targeted" / "v4_manifest.jsonl")}
    missing = [fid for fid in finalized if fid not in gt_src]
    if missing:
        print(f"[prep] WARNING: {len(missing)} finalized ids have no GT source record "
              f"(first 5: {missing[:5]})")

    frames = []
    for fid in finalized:
        r = gt_src.get(fid)
        if r is None:
            continue
        hz = r.get("_gt_hazards") or [{"label": "no_hazard"}]
        frames.append({
            "frame_id": fid, "image_path": r["image_path"], "split": "train",
            "scene_token": r.get("scene_token"), "weather": r.get("weather"),
            "time_of_day": r.get("time_of_day"), "location": r.get("location"),
            "annotation": {"hazards": hz,
                           "scene_summary": f"Reconstructed {r.get('weather')}/{r.get('time_of_day')} keyframe.",
                           "ego_context": {"weather": r.get("weather"),
                                           "time_of_day": r.get("time_of_day"),
                                           "road_type": "urban"}},
        })
    out = WORK / "targeted_gt"
    out.mkdir(parents=True, exist_ok=True)
    man = out / "annotated_manifest.json"
    json.dump(frames, man.open("w"), ensure_ascii=False, indent=2)
    print(f"[prep] wrote {man} ({len(frames)} frames). Next: describe_manifest.py on it, "
          f"then build_arms.py assemble.")


# --------------------------------------------------------------------------- #
def assemble():
    base_tr = load_jsonl(WORK / "base_val" / "sft_train.jsonl")
    base_va = load_jsonl(WORK / "base_val" / "sft_val.jsonl")
    test = load_jsonl(WORK / "eval_gt" / "sft_test.jsonl")
    fm_tgt = load_jsonl(WORK / "targeted_fm" / "sft_train.jsonl")
    gt_tgt = load_jsonl(WORK / "targeted_gt" / "sft_train.jsonl")

    fm_ids = ids_of(fm_tgt)
    gt_by_id = {r.get("frame_id"): r for r in gt_tgt}
    gt_aligned = [gt_by_id[i] for i in fm_ids if i in gt_by_id]
    gt_ids = ids_of(gt_aligned)

    print(f"[assemble] base_train={len(base_tr)} base_val={len(base_va)} test={len(test)}")
    print(f"[assemble] targeted FM={len(fm_ids)} GT_aligned={len(gt_ids)}")
    missing = fm_ids - gt_ids
    if missing:
        raise SystemExit(f"!!! GT arm missing {len(missing)} finalized ids "
                         f"(describe targeted_gt first). e.g. {list(missing)[:5]}")
    assert fm_ids == gt_ids, "FM/GT targeted id sets differ — arms would not be comparable"

    arm_fm_train = base_tr + fm_tgt
    arm_gt_train = base_tr + gt_aligned

    def write_arm(name, train):
        d = WORK / name
        dump_jsonl(train, d / "sft_train.jsonl")
        dump_jsonl(base_va, d / "sft_val.jsonl")
        dump_jsonl(test, d / "sft_test.jsonl")
        tr, va, te = ids_of(train), ids_of(base_va), ids_of(test)
        leak_te, leak_va = len(tr & te), len(tr & va)
        rep = {"arm": name, "train": len(train), "val": len(base_va), "test": len(test),
               "leak_train_vs_test": leak_te, "leak_train_vs_val": leak_va}
        (d / "arm_report.json").write_text(json.dumps(rep, indent=2))
        print("[assemble]", json.dumps(rep))
        assert leak_te == 0 and leak_va == 0, f"!!! LEAKAGE in {name} — do not train"
        return rep

    write_arm("arm_fm", arm_fm_train)
    write_arm("arm_gt", arm_gt_train)

    # both arms must differ ONLY on targeted boxes -> identical train id sets
    assert ids_of(arm_fm_train) == ids_of(arm_gt_train), \
        "!!! arm train id sets differ — base/targeted not aligned"
    print("[assemble] OK: identical train ids across arms, zero leakage. "
          "arm_fm/ and arm_gt/ ready to train.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "assemble"])
    a = ap.parse_args()
    prep() if a.stage == "prep" else assemble()


if __name__ == "__main__":
    main()
