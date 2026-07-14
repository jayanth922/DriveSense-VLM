#!/usr/bin/env python3
"""Annotation v2 regeneration (run in Colab): GT boxes + describe-only VLM + gate.

Pipeline (see docs/annotation_v2.md):
  curate rarity frames (or --shopping-list) → source tight GT boxes →
  describe-only Claude pass via the Message Batches API (50% cheaper; async) →
  SFT JSONL (scene-level split) → HARD validation gate.

The VLM never localizes: label + bbox_2d come from nuScenes GT; the VLM only
fills severity/reasoning/action. VLM-failed frames are excluded, not templated.
The describe pass is cost-optimised (batch, not real-time) and resumable: a
per-frame cache holds finished annotations and a batch-id state file lets a
restart poll in-flight batches instead of resubmitting them.

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
import json
import os
import random
import subprocess
import sys
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
    p.add_argument("--model", default="claude-sonnet-5",
                   help="Describe model (default claude-sonnet-5). A/B a few frames vs "
                        "claude-haiku-4-5-20251001 with --max-frames.")
    p.add_argument("--min-coverage", type=float, default=0.95,
                   help="Required fraction of selected frames with images (non-dry).")
    p.add_argument("--dry", action="store_true", help="Skip the image-coverage assertion.")
    p.add_argument("--shopping-list", default=None,
                   help="Use this mining_shoppinglist.jsonl as the frame source and SKIP "
                        "rarity re-curation (filter_rare_frames) AND per-scene dedup — the "
                        "miner already selected these frames. Labels exactly the mined set.")
    p.add_argument("--no-cache", action="store_true",
                   help="Disable the per-frame describe cache (default: cache under "
                        "<out-dir>/describe_cache so a killed run resumes, not restarts).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _cam_front_path(nusc: object, tok: str) -> str:
    """Return the CAM_FRONT image path for a sample token."""
    return nusc.get_sample_data_path(nusc.get("sample", tok)["data"]["CAM_FRONT"])  # type: ignore[attr-defined]


def _tokens_from_shoppinglist(nusc: object, args: argparse.Namespace) -> list[str]:
    """Frame source = mining shopping list. Rarity + per-scene dedup are SKIPPED.

    Uses the ``sample_token`` of each shopping-list row directly, so exactly the
    mined frames are labelled (not a re-filtered subset). Tokens absent from the
    loaded ``--version`` tables are dropped with a count.
    """
    valid = {s["token"] for s in nusc.sample}  # type: ignore[attr-defined]
    raw: list[str] = []
    for line in Path(args.shopping_list).read_text().splitlines():
        line = line.strip()
        if line:
            tok = json.loads(line).get("sample_token", "")
            if tok:
                raw.append(tok)
    seen: set[str] = set()
    ordered = [t for t in raw if not (t in seen or seen.add(t))]
    tokens = [t for t in ordered if t in valid]
    dropped = len(ordered) - len(tokens)
    if args.dry:  # partial mounts: mechanics test on image-covered frames only
        tokens = [t for t in tokens if os.path.exists(_cam_front_path(nusc, t))]
    random.shuffle(tokens)
    if args.max_frames:
        tokens = tokens[: args.max_frames]
    print(f"shopping-list: {len(ordered)} tokens, {dropped} not in {args.version} tables → "
          f"{len(tokens)} frames (rarity re-curation + per-scene dedup SKIPPED)")
    return tokens


def curate_and_dedup(nusc: object, filt: object, args: argparse.Namespace) -> list[str]:
    """Select rarity frames with images and cap frames per scene (dedup).

    When ``--shopping-list`` is set, delegate to :func:`_tokens_from_shoppinglist`
    and bypass rarity scoring + the per-scene cap entirely.
    """
    if args.shopping_list:
        return _tokens_from_shoppinglist(nusc, args)

    def img_path(tok: str) -> str:
        return _cam_front_path(nusc, tok)

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
    tag = "image-covered" if args.dry else "selected (coverage checked next)"
    print(f"curated {len(rare)} rare → {len(sel)} {tag} → {len(deduped)} after per-scene dedup")
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


def _read_cache(cache_dir: str | None, tok: str) -> dict | None:
    """Load a cached per-frame annotation, or ``None`` (missing/corrupt)."""
    if not cache_dir:
        return None
    cf = Path(cache_dir) / f"{tok}.json"
    if not cf.exists():
        return None
    try:
        return json.loads(cf.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _write_cache(cache_dir: str | None, tok: str, ann: dict) -> None:
    """Persist a per-frame annotation to the resume cache (no-op if disabled)."""
    if not cache_dir:
        return
    cf = Path(cache_dir) / f"{tok}.json"
    cf.parent.mkdir(parents=True, exist_ok=True)
    cf.write_text(json.dumps(ann))


def _no_hazard_ann(meta: dict) -> dict:
    """Annotation for a frame with no box-sourced hazards (no API call needed)."""
    return {"hazards": [], "scene_summary": "No annotatable hazard in the front camera view.",
            "ego_context": {"weather": meta["weather"], "time_of_day": meta["time_of_day"],
                            "road_type": "urban"}}


def _merge_hazards(kept: list[dict], meta: dict, vlm: dict) -> dict:
    """Merge GT boxes/labels (authoritative) with the VLM-described fields."""
    ctx = {"weather": meta["weather"], "time_of_day": meta["time_of_day"], "road_type": "urban"}
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


def _plan_frames(nusc: object, tokens: list[str], cache_dir: str | None) -> tuple[dict, dict, list]:
    """Box-source every frame locally; classify into cached / no-hazard / needs-describe.

    Returns ``(plan, anns, jobs)`` — ``plan`` maps token→{img,sc,meta,kept};
    ``anns`` holds finished annotations (from cache or the no-hazard shortcut);
    ``jobs`` is the ``(token, img, kept)`` list needing a paid describe call.
    """
    from drivesense.data.box_sourcing import source_boxes_for_frame  # noqa: PLC0415

    plan: dict[str, dict] = {}
    anns: dict[str, dict] = {}
    jobs: list = []
    missing = 0
    for tok in tokens:
        img = _cam_front_path(nusc, tok)
        if not os.path.exists(img):
            missing += 1
            continue
        sc, meta = scene_meta(nusc, tok)
        plan[tok] = {"img": img, "sc": sc, "meta": meta}
        cached = _read_cache(cache_dir, tok)
        if cached is not None:
            anns[tok] = cached
            continue
        kept, _ = source_boxes_for_frame(nusc, tok)
        if not kept:
            anns[tok] = _no_hazard_ann(meta)
            _write_cache(cache_dir, tok, anns[tok])
            continue
        plan[tok]["kept"] = kept
        jobs.append((tok, img, kept))
    print(f"planned {len(plan)} frames ({missing} missing image, {len(anns)} already done, "
          f"{len(jobs)} to describe via Batch API)")
    return plan, anns, jobs


def _batch_describe(args: argparse.Namespace, cache_dir: str | None,
                    plan: dict, anns: dict, jobs: list) -> None:
    """Run the Message Batches describe pass; resume prior batches, then submit new."""
    import anthropic  # noqa: PLC0415
    from drivesense.data import batch_describe as bd  # noqa: PLC0415

    state = bd.BatchState(Path(args.out_dir) / "batch_state.json")
    if not jobs and not state.ids:
        return
    client = anthropic.Anthropic()

    def on_result(tok: str, vlm: dict) -> None:
        p = plan.get(tok)
        if p is None or "kept" not in p:
            return
        anns[tok] = _merge_hazards(p["kept"], p["meta"], vlm)
        _write_cache(cache_dir, tok, anns[tok])

    bd.drain_existing(client, state, on_result)                 # resume any in-flight batches
    remaining = [(t, img, hz) for (t, img, hz) in jobs if t not in anns]
    print(f"batch describe: submitting {len(remaining)} frames "
          f"({len(jobs) - len(remaining)} recovered from prior batches)")
    bd.submit_new(client, remaining, state, args.model, _SYS_PROMPT, on_result)


def build_records(nusc: object, tokens: list[str], args: argparse.Namespace,
                  cache_dir: str | None) -> list[dict]:
    """Box-source locally, describe via the Batch API (50% cheaper), assemble SFT records.

    GT box/label are authoritative; the VLM only fills severity/reasoning/action.
    Resumable: the per-frame cache holds finished annotations and a batch-id state
    file lets a restart poll in-flight batches instead of resubmitting them.
    """
    from drivesense.data.annotation import SFTDataFormatter  # noqa: PLC0415

    fmt = SFTDataFormatter()
    plan, anns, jobs = _plan_frames(nusc, tokens, cache_dir)
    _batch_describe(args, cache_dir, plan, anns, jobs)

    records: list[dict] = []
    for tok, p in plan.items():
        ann = anns.get(tok)
        if ann is None:  # batch errored/expired for this frame — retried on next run
            continue
        rec = fmt.format_single_example({"image_path": p["img"], "annotations": ann,
                                         "frame_id": tok, "source": "nuscenes"})
        rec.update({"split": "", "scene_token": p["sc"], **p["meta"]})
        records.append(rec)
    print(f"built {len(records)} records ({len(plan) - len(records)} without annotation)")
    return records


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

    cache_dir = None if args.no_cache else str(Path(args.out_dir) / "describe_cache")
    if cache_dir:
        print(f"describe cache: {cache_dir} (resumable — re-run to continue after a drop)")
    print(f"describe pass: Message Batches API (50% off), model={args.model}")
    records = build_records(nusc, tokens, args, cache_dir)
    paths = split_and_write(records, Path(args.out_dir), args.seed)
    if run_gate(paths):
        print("\n✅ ALL SPLITS PASSED THE GATE. Labels are safe to train on (retrain is a separate step).")
    else:
        sys.exit("\n❌ GATE FAILED — do not retrain; fix and re-run.")


if __name__ == "__main__":
    main()
