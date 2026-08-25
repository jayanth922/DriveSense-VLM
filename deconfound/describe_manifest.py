#!/usr/bin/env python3
"""GT-describe a reconstructed manifest — fill severity/reasoning/action onto
already-localized GT boxes via the Message Batches API (resumable, -50%).

Used for the base+val sets and for the targeted-GT arm: the boxes come from
nuScenes GT (box_sourcing), and ONLY the prose is model-generated, so the two
arms differ purely in box provenance, never in the describe pass.

Reuses drivesense.data.batch_describe (build_request/chunk_jobs/BatchState/
submit_new/drain_existing) verbatim — the box-exempt, size-capped GT hazards are
passed straight through as the "given hazards".

  python deconfound/describe_manifest.py \
      --manifest /workspace/deconfound_work/base_val/annotated_manifest.json \
      --out      /workspace/deconfound_work/base_val \
      --state    /workspace/deconfound_work/base_val/describe_batches.json \
      --downsize 768 --model "$SONNET"

Resumable: batch ids are persisted in --state before polling, so a restart
drains existing batches instead of resubmitting (no double charge). Per-frame
described results are cached in <out>/describe_cache.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(os.environ.get("REPO", "/workspace/DriveSense"))
sys.path.insert(0, str(REPO / "src"))

DESCRIBE_SYSTEM = (
    "You are an expert ADAS driving-scene analyst. You are given an image and a "
    "list of already-localized hazards (label + optional bbox_2d in [0,1000] "
    "coords) for the front camera of an ego vehicle. Do NOT add, remove, or "
    "re-localize hazards. For EACH given hazard, in the SAME order, output:\n"
    "  severity : one of low | medium | high\n"
    "  reasoning: one concise sentence on why it is a hazard to the ego vehicle\n"
    "  action   : one concise driving action the ego vehicle should take\n"
    "Respond with JSON only: {\"hazards\": [{\"severity\":..,\"reasoning\":..,"
    "\"action\":..}, ...]} with exactly one entry per given hazard, same order."
)


def downsized_path(img_path: str, max_side: int, cache_dir: Path) -> str:
    """Return a path to a JPEG whose longest side is <= max_side (cached)."""
    if not max_side:
        return img_path
    from PIL import Image
    cache_dir.mkdir(parents=True, exist_ok=True)
    dst = cache_dir / (Path(img_path).stem + f"_{max_side}.jpg")
    if dst.exists():
        return str(dst)
    try:
        im = Image.open(img_path).convert("RGB")
        w, h = im.size
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            im = im.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
        im.save(dst, "JPEG", quality=88)
        return str(dst)
    except Exception:
        return img_path  # fall back to full-res on any image error


def merge_described(hazards: list[dict], vlm: dict) -> list[dict]:
    """Merge model severity/reasoning/action onto GT hazards by order."""
    desc = vlm.get("hazards") if isinstance(vlm, dict) else None
    if not isinstance(desc, list):
        return hazards
    out = []
    for i, h in enumerate(hazards):
        d = desc[i] if i < len(desc) and isinstance(desc[i], dict) else {}
        merged = dict(h)
        merged["severity"] = d.get("severity", h.get("severity", "medium"))
        merged["reasoning"] = d.get("reasoning", h.get("reasoning", "Potential hazard in the ego path."))
        merged["action"] = d.get("action", h.get("action", "Slow down and be prepared to stop."))
        out.append(merged)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="annotated_manifest.json (GT-boxed)")
    ap.add_argument("--out", required=True, help="output dir (writes described manifest + SFT)")
    ap.add_argument("--state", default=None, help="batch-id state file (resume)")
    ap.add_argument("--downsize", type=int, default=768, help="max image side (0 = full res)")
    ap.add_argument("--model", default=os.environ.get("SONNET", "claude-sonnet-4-20250514"))
    ap.add_argument("--mock", action="store_true", help="no API; fill neutral text")
    a = ap.parse_args()

    from drivesense.data import batch_describe as BD
    from drivesense.utils.config import load_config
    from drivesense.data.annotation import SFTDataFormatter

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    state_path = Path(a.state) if a.state else out / "describe_batches.json"
    cache_path = out / "describe_cache.jsonl"
    dcache = out / "_downsized"

    frames = json.load(open(a.manifest))
    fmap = {f["frame_id"]: f for f in frames}

    # jobs: describe only box-bearing frames (no_hazard-only frames need no prose gen)
    jobs = []
    for f in frames:
        hz = f["annotation"]["hazards"]
        jobs.append((f["frame_id"], downsized_path(f["image_path"], a.downsize, dcache), hz))
    print(f"[describe] frames={len(frames)} model={a.model} downsize={a.downsize} mock={a.mock}",
          flush=True)

    # resume cache: frame_ids already described
    done = {}
    if cache_path.exists():
        for line in cache_path.open():
            line = line.strip()
            if line:
                r = json.loads(line)
                done[r["frame_id"]] = r["vlm"]
    cache_f = cache_path.open("a")

    def on_result(custom_id: str, vlm: dict):
        f = fmap.get(custom_id)
        if f is None:
            return
        f["annotation"]["hazards"] = merge_described(f["annotation"]["hazards"], vlm)
        cache_f.write(json.dumps({"frame_id": custom_id, "vlm": vlm}) + "\n")
        cache_f.flush()

    if a.mock:
        for fid, _, hz in jobs:
            on_result(fid, {"hazards": [{"severity": "medium",
                "reasoning": "Mock described hazard.", "action": "Proceed with caution."}
                for _ in hz]})
    else:
        import anthropic
        client = anthropic.Anthropic()
        state = BD.BatchState(state_path)
        # apply any cached results first (free), then only submit the remainder
        for fid, vlm in done.items():
            on_result(fid, vlm)
        BD.drain_existing(client, state, on_result)   # collect completed batches FIRST (free)
        got = set()
        if cache_path.exists():
            for line in cache_path.open():
                line = line.strip()
                if line:
                    got.add(json.loads(line)["frame_id"])
        remaining = [(fid, img, hz) for (fid, img, hz) in jobs if fid not in got]
        print(f"[describe] recovered={len(got)} to_submit={len(remaining)}", flush=True)
        if remaining:
            BD.submit_new(client, remaining, state, a.model, DESCRIBE_SYSTEM, on_result)

    cache_f.close()
    described = out / "annotated_manifest_described.json"
    json.dump(frames, described.open("w"), ensure_ascii=False, indent=2)
    cfg = load_config(str(REPO / "configs/data.yaml"))
    SFTDataFormatter(cfg).format_dataset(described, out)
    print(f"[describe] wrote {described} and SFT splits under {out}")


if __name__ == "__main__":
    main()
