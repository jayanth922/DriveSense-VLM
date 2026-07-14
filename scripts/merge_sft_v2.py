#!/usr/bin/env python3
"""Merge mined v2 labels + the existing 688-frame v2 set into one scene-split dataset.

Pools all records from both label dirs, de-dups by ``frame_id`` (the existing set
wins on any conflict), then re-assigns an 80/10/10 split at the SCENE level so no
scene_token ever appears in two splits. Writes ``sft_{train,val,test}_enriched.jsonl``
to ``--out-dir``.

Usage:
    python scripts/merge_sft_v2.py \
        --new-dir outputs/data/sft_ready_v2_mined \
        --old-dir outputs/data/sft_ready_v2 \
        --out-dir outputs/data/sft_ready_v2_merged
    # also run the hard validation gate on the merged splits:
    python scripts/merge_sft_v2.py ... --gate
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--new-dir", default="outputs/data/sft_ready_v2_mined",
                   help="Dir of newly mined sft_*_enriched.jsonl labels.")
    p.add_argument("--old-dir", default="outputs/data/sft_ready_v2",
                   help="Dir of the existing 688-frame v2 sft_*_enriched.jsonl labels.")
    p.add_argument("--out-dir", default="outputs/data/sft_ready_v2_merged",
                   help="Destination for the merged, re-split labels.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gate", action="store_true",
                   help="Run run_label_validation.py on each merged split after writing.")
    return p.parse_args()


def load_split_records(d: Path) -> list[dict]:
    """Load every record from ``sft_{train,val,test}_enriched.jsonl`` under ``d``."""
    records: list[dict] = []
    for sp in _SPLITS:
        p = d / f"sft_{sp}_enriched.jsonl"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def require_keys(records: list[dict]) -> None:
    """Abort if any record lacks ``frame_id`` or ``scene_token`` (can't merge/split)."""
    bad = [i for i, r in enumerate(records)
           if not r.get("frame_id") or not r.get("scene_token")]
    if bad:
        sys.exit(f"ERROR: {len(bad)} records missing frame_id/scene_token (first at index "
                 f"{bad[0]}). Are these v2 *_enriched.jsonl files?")


def dedup_by_frame_id(records: list[dict]) -> list[dict]:
    """Drop later duplicates by ``frame_id`` (first occurrence wins)."""
    seen: set[str] = set()
    out: list[dict] = []
    for r in records:
        fid = r["frame_id"]
        if fid in seen:
            continue
        seen.add(fid)
        out.append(r)
    return out


def assign_scene_split(records: list[dict], seed: int) -> dict[str, str]:
    """Assign an 80/10/10 split per SCENE (mutates ``records['split']`` in place)."""
    scenes = sorted({r["scene_token"] for r in records})
    random.Random(seed).shuffle(scenes)
    n = len(scenes)
    split_of = {**{s: "train" for s in scenes[: max(1, int(.8 * n))]},
                **{s: "val" for s in scenes[max(1, int(.8 * n)): max(2, int(.9 * n))]},
                **{s: "test" for s in scenes[max(2, int(.9 * n)):]}}
    for r in records:
        r["split"] = split_of[r["scene_token"]]
    return split_of


def verify_no_scene_leak(records: list[dict]) -> None:
    """Raise if any scene_token ended up in more than one split."""
    by_scene: dict[str, set[str]] = {}
    for r in records:
        by_scene.setdefault(r["scene_token"], set()).add(r["split"])
    leaked = {s: v for s, v in by_scene.items() if len(v) > 1}
    if leaked:
        example = next(iter(leaked.items()))
        raise AssertionError(f"scene leak: {len(leaked)} scenes span multiple splits "
                             f"(e.g. {example[0]} → {sorted(example[1])})")


def write_splits(records: list[dict], out_dir: Path) -> dict[str, Path]:
    """Write per-split JSONL to ``out_dir`` and print counts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for sp in _SPLITS:
        p = out_dir / f"sft_{sp}_enriched.jsonl"
        rows = [r for r in records if r["split"] == sp]
        with p.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        paths[sp] = p
        print(f"  {sp}: {len(rows)} → {p}")
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
            print(g.stderr[-500:])
            ok = False
    return ok


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    old = load_split_records(Path(args.old_dir))
    new = load_split_records(Path(args.new_dir))
    print(f"loaded {len(old)} existing ({args.old_dir}) + {len(new)} mined ({args.new_dir})")

    pool = old + new                       # old first → the proven 688 set wins on conflict
    require_keys(pool)
    records = dedup_by_frame_id(pool)
    print(f"de-duped by frame_id: {len(records)} unique ({len(pool) - len(records)} dropped)")

    assign_scene_split(records, args.seed)
    verify_no_scene_leak(records)
    print(f"scene-level split over {len({r['scene_token'] for r in records})} scenes:")
    paths = write_splits(records, Path(args.out_dir))

    if args.gate:
        if not run_gate(paths):
            sys.exit("\n❌ GATE FAILED on the merged set — do not train on it.")
        print("\n✅ merged set PASSED the gate. Safe to train.")
    else:
        print("\nNext — gate the merged set before training:")
        for sp in _SPLITS:
            print(f"  python scripts/run_label_validation.py --labels {paths[sp]}")


if __name__ == "__main__":
    main()
