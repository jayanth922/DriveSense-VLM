#!/usr/bin/env python3
"""Annotation v2 regeneration (run in Colab): GT boxes + describe-only VLM + gate.

Pipeline (see docs/annotation_v2.md):
  curate rarity frames → per-scene dedup → source tight GT boxes →
  describe-only Claude pass (severity/reasoning/action; NEVER boxes) →
  SFT JSONL (scene-level split) → HARD validation gate.

The VLM never localizes: label + bbox_2d come from nuScenes GT; the VLM only
fills severity/reasoning/action. VLM-failed frames are excluded, not templated.

Usage (Colab, after cloning the repo and mounting Drive):
    export ANTHROPIC_API_KEY=...          # required
    python scripts/regenerate_annotations_v2_colab.py \
        --dataroot /content/drive/MyDrive/.../nuscenes --version v1.0-trainval \
        --out-dir  /content/drive/MyDrive/.../sft_ready_v2 \
        --min-score 5 --frames-per-scene 3
    # dry / mechanics run on partial image mounts (skips the coverage assertion):
    python scripts/regenerate_annotations_v2_colab.py ... --max-frames 50 --dry
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_SYS_PROMPT = (
    "You are an AV safety analyst. You are given a dashcam image and a list of "
    "ALREADY-LOCALIZED hazards (fixed class + box). DO NOT invent, move, add or "
    "remove hazards/boxes. For EACH, give severity (low|medium|high|critical), "
    "reasoning (2-3 sentences on why it endangers the ego), and a driving action. "
    "Also scene_summary and ego_context. Output ONLY JSON: "
    '{"hazards":[{"label","severity","reasoning","action"}],"scene_summary",'
    '"ego_context":{"weather","time_of_day","road_type"}}'
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataroot", required=True, help="nuScenes dataset root.")
    p.add_argument("--version", default="v1.0-trainval")
    p.add_argument("--out-dir", required=True, help="Destination for sft_*_enriched.jsonl.")
    p.add_argument("--min-score", type=int, default=5, help="Rarity threshold (5 = genuinely rare).")
    p.add_argument("--frames-per-scene", type=int, default=3,
                   help="Cap kept frames per scene to remove near-duplicate keyframes.")
    p.add_argument("--max-frames", type=int, default=None, help="Cap total frames (dry runs).")
    p.add_argument("--model", default="claude-sonnet-4-5")
    p.add_argument("--min-coverage", type=float, default=0.95,
                   help="Required fraction of selected frames with images (non-dry).")
    p.add_argument("--dry", action="store_true", help="Skip the image-coverage assertion.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def curate_and_dedup(nusc: object, filt: object, args: argparse.Namespace) -> list[str]:
    """Select rarity frames with images and cap frames per scene (dedup)."""
    def img_path(tok: str) -> str:
        return nusc.get_sample_data_path(nusc.get("sample", tok)["data"]["CAM_FRONT"])  # type: ignore[attr-defined]

    rare = [f["sample_token"] for f in filt.filter_rare_frames(min_score=args.min_score)]  # type: ignore[attr-defined]
    # Dry runs use only image-covered frames (mechanics test); full runs use all
    # rare frames and rely on main()'s coverage assertion to reject a partial mount.
    if args.dry:
        covered = {s["token"] for s in nusc.sample if os.path.exists(img_path(s["token"]))}  # type: ignore[attr-defined]
        sel = [t for t in rare if t in covered]
    else:
        sel = rare

    # Per-scene cap: keep at most frames_per_scene, evenly spaced in time.
    by_scene: dict[str, list[str]] = defaultdict(list)
    for t in sel:
        by_scene[nusc.get("sample", t)["scene_token"]].append(t)  # type: ignore[attr-defined]
    deduped: list[str] = []
    k = max(1, args.frames_per_scene)
    for toks in by_scene.values():
        if len(toks) <= k:
            deduped += toks
        else:
            idx = sorted({round(i * (len(toks) - 1) / (k - 1)) for i in range(k)}) if k > 1 else [0]
            deduped += [toks[i] for i in idx]

    random.shuffle(deduped)
    if args.max_frames:
        deduped = deduped[: args.max_frames]
    print(f"curated {len(rare)} rare → {len(sel)} covered → {len(deduped)} after per-scene dedup")
    return deduped


def scene_meta(nusc: object, tok: str) -> tuple[str, dict]:
    """Return (scene_token, {time_of_day, weather, location}) for a sample."""
    s = nusc.get("sample", tok)  # type: ignore[attr-defined]
    scene = nusc.get("scene", s["scene_token"])  # type: ignore[attr-defined]
    log = nusc.get("log", scene["log_token"])  # type: ignore[attr-defined]
    d = scene.get("description", "").lower()
    loc = log.get("location", "")
    return s["scene_token"], {
        "time_of_day": "night" if "night" in d else "day",
        "weather": "rain" if "rain" in d else "fog" if "fog" in d else "clear",
        "location": "boston" if "boston" in loc else "singapore" if "singapore" in loc else (loc or "unknown"),
    }


def make_describe(model: str) -> object:
    """Build the describe-only VLM call (returns a function). Claude fills fields, not boxes."""
    import anthropic  # noqa: PLC0415

    client = anthropic.Anthropic(max_retries=5)

    def describe(img_path: str, hazards: list[dict]) -> dict | None:
        with open(img_path, "rb") as f:
            b64 = base64.standard_b64encode(f.read()).decode()
        given = [{"label": h["label"], **({"bbox_2d": h["bbox_2d"]} if "bbox_2d" in h else {})} for h in hazards]
        user = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
            {"type": "text", "text": "Given hazards (keep order): " + json.dumps(given)
             + "\nFill severity/reasoning/action for each in the SAME order. JSON only."},
        ]
        last = None
        for a in range(4):
            try:
                r = client.messages.create(model=model, max_tokens=4096, system=_SYS_PROMPT,
                                           messages=[{"role": "user", "content": user}])
                if r.stop_reason == "max_tokens":
                    last = "hit max_tokens"; continue
                m = re.search(r"\{.*\}", r.content[0].text, re.DOTALL)
                if m:
                    return json.loads(m.group(0))
                last = "no JSON"
            except Exception as exc:  # noqa: BLE001
                last = repr(exc); time.sleep(min(30, 3 * (2 ** a)))
        print("  describe() failed:", last)
        return None

    return describe


def build_records(nusc: object, tokens: list[str], describe: object) -> list[dict]:
    """Box-source + describe each frame; GT box/label authoritative. Returns SFT records."""
    from drivesense.data.box_sourcing import source_boxes_for_frame  # noqa: PLC0415
    from drivesense.data.annotation import SFTDataFormatter  # noqa: PLC0415

    fmt = SFTDataFormatter()
    records: list[dict] = []
    failed = missing = 0
    for i, tok in enumerate(tokens):
        s = nusc.get("sample", tok)  # type: ignore[attr-defined]
        img = nusc.get_sample_data_path(s["data"]["CAM_FRONT"])  # type: ignore[attr-defined]
        if not os.path.exists(img):
            missing += 1
            continue
        kept, _ = source_boxes_for_frame(nusc, tok)
        sc, meta = scene_meta(nusc, tok)
        ann = _annotate(img, kept, meta, describe)
        if ann is None:
            failed += 1
            continue
        rec = fmt.format_single_example({"image_path": img, "annotations": ann,
                                         "frame_id": tok, "source": "nuscenes"})
        rec.update({"split": "", "scene_token": sc, **meta})
        records.append(rec)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(tokens)} ({failed} VLM fail, {missing} missing img)")
    print(f"built {len(records)} records, {failed} VLM failures, {missing} skipped (missing image)")
    return records


def _annotate(img: str, kept: list[dict], meta: dict, describe: object) -> dict | None:
    """Assemble one frame's annotation (GT boxes + VLM-described fields)."""
    ctx = {"weather": meta["weather"], "time_of_day": meta["time_of_day"], "road_type": "urban"}
    if not kept:
        return {"hazards": [], "scene_summary": "No annotatable hazard in the front camera view.",
                "ego_context": ctx}
    vlm = describe(img, kept)  # type: ignore[operator]
    if vlm is None:
        return None
    by_label = vlm.get("hazards", [])
    haz: list[dict] = []
    for j, h in enumerate(kept):
        d = by_label[j] if j < len(by_label) else {}
        entry = {"label": h["label"], "severity": d.get("severity", "medium"),
                 "reasoning": d.get("reasoning", ""), "action": d.get("action", "reduce speed")}
        if "bbox_2d" in h:
            entry["bbox_2d"] = h["bbox_2d"]
        haz.append(entry)
    return {"hazards": haz, "scene_summary": vlm.get("scene_summary", ""),
            "ego_context": vlm.get("ego_context", ctx)}


def split_and_write(records: list[dict], out_dir: Path, seed: int) -> dict[str, Path]:
    """Assign a scene-level 80/10/10 split and write per-split JSONL."""
    scenes = sorted({r["scene_token"] for r in records})
    random.Random(seed).shuffle(scenes)
    n = len(scenes)
    split_of = {**{s: "train" for s in scenes[: max(1, int(.8 * n))]},
                **{s: "val" for s in scenes[max(1, int(.8 * n)): max(2, int(.9 * n))]},
                **{s: "test" for s in scenes[max(2, int(.9 * n)):]}}
    for r in records:                       # stamp the split onto each record
        r["split"] = split_of[r["scene_token"]]
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for sp in ("train", "val", "test"):
        p = out_dir / f"sft_{sp}_enriched.jsonl"
        paths[sp] = p
        with p.open("w", encoding="utf-8") as f:
            for r in records:
                if r["split"] == sp:
                    f.write(json.dumps(r) + "\n")
        print(f"  {sp}: {sum(1 for r in records if r['split'] == sp)} → {p}")
    return paths


def run_gate(paths: dict[str, Path]) -> bool:
    """Run the hard validation gate on each non-empty split. Returns overall pass."""
    ok = True
    for sp, p in paths.items():
        if p.stat().st_size == 0:
            continue
        print(f"\n=== gate: {sp} ===")
        g = subprocess.run([sys.executable, str(_REPO / "scripts" / "run_label_validation.py"),
                            "--labels", str(p)], capture_output=True, text=True)
        print(g.stdout[-1500:])
        if g.returncode != 0:
            print(g.stderr[-500:]); ok = False
    return ok


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ERROR: ANTHROPIC_API_KEY not set.")
    random.seed(args.seed)

    from drivesense.utils.config import load_config  # noqa: PLC0415
    from drivesense.data.nuscenes_loader import NuScenesRarityFilter  # noqa: PLC0415

    cfg = load_config(str(_REPO / "configs" / "data.yaml"))
    cfg["nuscenes"]["version"] = args.version
    filt = NuScenesRarityFilter(Path(args.dataroot), cfg)
    nusc = filt.nusc

    tokens = curate_and_dedup(nusc, filt, args)

    # Non-dry runs must have (near) full image coverage or the dataset is biased.
    if not args.dry:
        cov = sum(os.path.exists(nusc.get_sample_data_path(nusc.get("sample", t)["data"]["CAM_FRONT"]))
                  for t in tokens) / max(1, len(tokens))
        if cov < args.min_coverage:
            sys.exit(f"ERROR: image coverage {cov:.1%} < {args.min_coverage:.0%}. "
                     "Mount all trainval blobs, or use --dry for a mechanics run.")

    records = build_records(nusc, tokens, make_describe(args.model))
    paths = split_and_write(records, Path(args.out_dir), args.seed)
    if run_gate(paths):
        print("\n✅ ALL SPLITS PASSED THE GATE. Labels are safe to train on (retrain is a separate step).")
    else:
        sys.exit("\n❌ GATE FAILED — do not retrain; fix and re-run.")


if __name__ == "__main__":
    main()
