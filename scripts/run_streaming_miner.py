#!/usr/bin/env python3
"""Streaming, bounded-storage nuScenes image miner.

Fetch CAM_FRONT keyframe images for shopping-list frames by processing the
trainval image blobs ONE AT A TIME (~250 GB total, but never more than a single
~25 GB tarball on disk at once). Per-frame metadata is already global, so this
only FETCHES images — it never re-scores.

Usage:
    # 1. Validate the plan without downloading anything (recommended first step):
    python scripts/run_streaming_miner.py --dry-run

    # 2. Build/refresh only the shopping list:
    python scripts/run_streaming_miner.py --build-list-only --target-count 4500

    # 3. Real run (needs nuScenes auth — see --dry-run output):
    NUSCENES_BLOB_URLS=urls.json python scripts/run_streaming_miner.py
    #   or:  python scripts/run_streaming_miner.py --blob-dir /data/tarballs

The blobs are NOT public. Run --dry-run to see exactly what auth to supply.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from drivesense.utils.config import load_config  # noqa: E402
from drivesense.data import streaming_miner as sm  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_streaming_miner")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Streaming bounded-storage nuScenes image miner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", default="configs/data.yaml", help="Path to data.yaml")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate shopping list + blob plan + auth; download nothing")
    p.add_argument("--build-list-only", action="store_true",
                   help="Build the shopping list and exit")
    p.add_argument("--rebuild-list", action="store_true",
                   help="Rebuild the shopping list even if it already exists")
    p.add_argument("--target-count", type=int, default=None,
                   help="Override mining.target_count (0 = whole band)")
    p.add_argument("--disk-cap-gb", type=float, default=None,
                   help="Override mining.disk_cap_gb")
    p.add_argument("--count-mode", choices=["hazard_class", "num_annotations"], default=None,
                   help="Override mining.count_mode")
    p.add_argument("--blobs", nargs="+", default=None,
                   help="Override the list of blobs to process")
    p.add_argument("--blob-dir", default=None,
                   help="Dir of pre-downloaded blob tarballs (offline mode)")
    p.add_argument("--blob-urls-file", default=None,
                   help="JSON map {blob: signed_url} (or set NUSCENES_BLOB_URLS)")
    p.add_argument("--metadata", default=None, help="Override mining.metadata_path")
    p.add_argument("--dataroot", default=None, help="Override mining.dataroot")
    return p.parse_args()


def resolve_cfg(args: argparse.Namespace) -> dict:
    """Merge config file + CLI overrides into a flat mining-params dict."""
    m = load_config(args.config).get("mining", {})
    dataroot = args.dataroot or m["dataroot"]
    return {
        "metadata_path": args.metadata or m["metadata_path"],
        "shoppinglist_path": m["shoppinglist_path"],
        "manifest_path": m["manifest_path"],
        "report_path": m["report_path"],
        "dataroot": dataroot,
        "cam_front_dir": str(Path(dataroot) / "samples" / "CAM_FRONT"),
        "temp_dir": m["temp_dir"],
        "band": tuple(m.get("hazard_band", [3, 20])),
        "count_mode": args.count_mode or m.get("count_mode", "hazard_class"),
        "strata_edges": list(m.get("strata_edges", [3, 6, 9, 13, 21])),
        "target_count": args.target_count if args.target_count is not None
        else int(m.get("target_count", 4500)),
        "seed": int(m.get("seed", 42)),
        "disk_cap_gb": args.disk_cap_gb if args.disk_cap_gb is not None
        else float(m.get("disk_cap_gb", 50)),
        "blobs": args.blobs or list(m.get("needed_blobs", sm.TRAINVAL_BLOBS)),
        "base_url": m.get("base_url", ""),
        "_mining": m,
    }


# ---------------------------------------------------------------------------
# Shopping list
# ---------------------------------------------------------------------------


def build_or_load_list(cfg: dict, args: argparse.Namespace) -> list[dict]:
    """Build the shopping list (band filter + stratified sample) or reuse it."""
    path = Path(cfg["shoppinglist_path"])
    if path.exists() and not args.rebuild_list:
        frames = sm.load_shoppinglist(path)
        logger.info("Reusing existing shopping list: %d frames (%s)", len(frames), path)
        return frames
    meta = Path(cfg["metadata_path"])
    if not meta.exists():
        logger.error("Metadata not found: %s", meta)
        logger.error("This file (34k keyframes) lives on Drive — copy it locally or "
                     "pass --metadata <path>.")
        sys.exit(1)
    logger.info("Scanning metadata for %s hazards in band %s ...", cfg["count_mode"], cfg["band"])
    candidates = sm.band_frames_without_images(
        meta, cfg["cam_front_dir"], cfg["band"], cfg["count_mode"])
    logger.info("In-band frames without local images: %d", len(candidates))
    sampled = sm.stratified_sample(
        candidates, cfg["target_count"], cfg["strata_edges"], cfg["seed"])
    sm.write_shoppinglist(sampled, path)
    logger.info("Wrote shopping list: %d frames → %s", len(sampled), path)
    return sampled


def print_strata(frames: list[dict], edges: list[int]) -> None:
    """Log the hazard-count stratum distribution of the shopping list."""
    hist = sm.stratum_histogram(frames, edges)
    logger.info("Stratum distribution (hazard-count bucket → frames):")
    for bucket, n in hist.items():
        logger.info("    %-10s %6d", bucket, n)


# ---------------------------------------------------------------------------
# Blob plan + auth
# ---------------------------------------------------------------------------


def plan_blobs(cfg: dict, args: argparse.Namespace) -> tuple[list[tuple[str, str, str]], bool]:
    """Resolve a source for each pending blob; return (plan, all_resolved).

    plan entries are ``(blob, kind, ref)``; completed blobs are marked ``kind="done"``.
    """
    manifest = sm.MiningManifest(cfg["manifest_path"])
    url_map = sm.load_blob_urls(cfg["_mining"], args.blob_urls_file)
    token = os.environ.get("NUSCENES_TOKEN")
    plan: list[tuple[str, str, str]] = []
    all_ok = True
    for blob in cfg["blobs"]:
        if manifest.is_done(blob):
            plan.append((blob, "done", ""))
            continue
        kind, ref = sm.resolve_blob_source(
            blob, args.blob_dir, url_map, token, cfg["base_url"])
        if kind == "missing":
            all_ok = False
        plan.append((blob, kind, ref))
    return plan, all_ok


def print_plan(plan: list[tuple[str, str, str]], cfg: dict, n_frames: int) -> None:
    """Print the blob-processing plan as a table."""
    print("\n" + "=" * 64)
    print("  BLOB PLAN")
    print("=" * 64)
    print(f"  shopping list : {n_frames} frames")
    print(f"  dataroot      : {cfg['dataroot']}")
    print(f"  temp dir      : {cfg['temp_dir']}")
    print(f"  disk cap      : {cfg['disk_cap_gb']:.0f} GB (one blob at a time)")
    print("-" * 64)
    print(f"  {'blob':<32}{'source'}")
    print("-" * 64)
    for blob, kind, _ in plan:
        print(f"  {blob:<32}{kind}")
    print("=" * 64 + "\n")


# ---------------------------------------------------------------------------
# Streaming loop
# ---------------------------------------------------------------------------


def process_blob(blob: str, kind: str, ref: str, cfg: dict, wanted: set[str]) -> tuple[int, float]:
    """Fetch (if needed) + stream-extract one blob, then clean up. Bounded storage.

    Returns ``(matched_count, footprint_gb)``. Guarantees the tarball is deleted
    before returning, so two blobs never coexist.
    """
    temp_dir = Path(cfg["temp_dir"])
    temp_dir.mkdir(parents=True, exist_ok=True)
    cap_bytes = int(cfg["disk_cap_gb"] * (1024 ** 3))
    tar_path = Path(ref) if kind == "local" else temp_dir / blob
    footprint_gb = 0.0
    try:
        if kind != "local":
            _guard_free_space(cfg)
            headers = ({"Authorization": f"token {os.environ['NUSCENES_TOKEN']}"}
                       if kind == "token" else None)
            logger.info("[%s] downloading ...", blob)
            sm.download_blob(ref, tar_path, headers=headers, cap_bytes=cap_bytes)
        footprint_gb = sm.dir_size_bytes(temp_dir) / (1024 ** 3)
        if footprint_gb > cfg["disk_cap_gb"]:
            raise RuntimeError(f"[{blob}] footprint {footprint_gb:.1f} GB exceeds cap")
        logger.info("[%s] extracting matching keyframes (footprint %.1f GB) ...",
                    blob, footprint_gb)
        written, skipped = sm.stream_extract_blob(tar_path, wanted, cfg["cam_front_dir"])
        logger.info("[%s] matched %d new (%d already present)", blob, len(written), skipped)
        return len(written), footprint_gb
    finally:
        if kind != "local" and tar_path.exists():
            tar_path.unlink()  # delete tarball immediately — never keep two blobs
            logger.info("[%s] deleted tarball", blob)


def _guard_free_space(cfg: dict) -> None:
    """Abort before download if free disk is below the one-blob cap."""
    avail = sm.free_gb(cfg["temp_dir"])
    if avail < cfg["disk_cap_gb"]:
        raise RuntimeError(
            f"only {avail:.1f} GB free at {cfg['temp_dir']} — need >= "
            f"{cfg['disk_cap_gb']:.0f} GB headroom for one blob")


def run_streaming(cfg: dict, plan: list[tuple[str, str, str]], frames: list[dict]) -> dict:
    """Execute the streaming loop over pending blobs; return the report dict."""
    manifest = sm.MiningManifest(cfg["manifest_path"])
    wanted = {f["basename"] for f in frames}
    running_total = manifest.total_matched()
    for blob, kind, ref in plan:
        if kind == "done":
            logger.info("[%s] already complete — skipping", blob)
            continue
        matched, footprint = process_blob(blob, kind, ref, cfg, wanted)
        running_total += matched
        manifest.mark_done(blob, matched, footprint)
        logger.info("[%s] running total fetched: %d | disk HWM: %.1f GB",
                    blob, running_total, manifest.disk_hwm_gb)
    return _build_report(cfg, manifest, len(wanted))


def _build_report(cfg: dict, manifest: sm.MiningManifest, list_size: int) -> dict:
    """Assemble the final mining report dict and count the on-disk dataset."""
    final_images = len(list(Path(cfg["cam_front_dir"]).glob("*.jpg"))) \
        if Path(cfg["cam_front_dir"]).exists() else 0
    return {
        "shopping_list_size": list_size,
        "completed_blobs": manifest.completed_blobs,
        "per_blob_matched": manifest.per_blob,
        "total_frames_added": manifest.total_matched(),
        "final_cam_front_images": final_images,
        "disk_high_water_mark_gb": round(manifest.disk_hwm_gb, 2),
    }


def print_report(report: dict) -> None:
    """Print the final summary table."""
    print("\n" + "=" * 64)
    print("  MINING REPORT")
    print("=" * 64)
    print(f"  {'blob':<32}{'frames matched':>16}")
    print("-" * 64)
    for blob in report["completed_blobs"]:
        print(f"  {blob:<32}{report['per_blob_matched'].get(blob, 0):>16}")
    print("-" * 64)
    print(f"  {'shopping list size':<32}{report['shopping_list_size']:>16}")
    print(f"  {'total frames added':<32}{report['total_frames_added']:>16}")
    print(f"  {'final CAM_FRONT images':<32}{report['final_cam_front_images']:>16}")
    print(f"  {'disk high-water mark (GB)':<32}{report['disk_high_water_mark_gb']:>16}")
    print("=" * 64 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    cfg = resolve_cfg(args)

    frames = build_or_load_list(cfg, args)
    print_strata(frames, cfg["strata_edges"])
    if args.build_list_only:
        return

    plan, all_ok = plan_blobs(cfg, args)
    print_plan(plan, cfg, len(frames))

    pending = [(b, k, r) for b, k, r in plan if k not in ("done",)]
    if not all_ok:
        logger.error("Cannot resolve a source for %d blob(s). STOPPING.",
                     sum(1 for _, k, _ in plan if k == "missing"))
        print("\n" + sm.auth_instructions())
        sys.exit(2)

    if args.dry_run:
        logger.info("DRY RUN: shopping list + blob plan + auth all validated. "
                    "%d blob(s) pending, %d already done. Nothing downloaded.",
                    len(pending), len(plan) - len(pending))
        return

    report = run_streaming(cfg, plan, frames)
    Path(cfg["report_path"]).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg["report_path"]).write_text(json.dumps(report, indent=2))
    logger.info("Wrote report → %s", cfg["report_path"])
    print_report(report)


if __name__ == "__main__":
    main()
