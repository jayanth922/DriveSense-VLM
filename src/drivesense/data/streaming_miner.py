"""Streaming, bounded-storage image miner for nuScenes trainval blobs.

The nuScenes trainval image set ships as 10 tarballs (~250 GB total). This module
fetches only the CAM_FRONT keyframe images we actually want (a "shopping list"),
processing ONE blob at a time so peak disk never holds more than a single
tarball. Per-frame metadata is already global (independent of which blob's images
are present), so no per-blob re-scoring happens here — the miner only FETCHES.

Pipeline:
    1. build_shoppinglist  — band-filter + stratified sample from metadata.jsonl
    2. stream_extract_blob — pull matching CAM_FRONT keyframes out of one tarball
    3. MiningManifest      — blob-level resume state (idempotent across restarts)

Auth is deliberately NOT baked in: the blobs need a nuScenes account + accepted
license and are served via expiring signed URLs. See ``resolve_blob_source``.
"""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
import tarfile
from pathlib import Path
from typing import Iterator

from drivesense.data.box_sourcing import nuscenes_category_to_hazard

logger = logging.getLogger(__name__)

# All 10 trainval image blobs, in canonical order.
TRAINVAL_BLOBS: tuple[str, ...] = tuple(
    f"v1.0-trainval{n:02d}_blobs.tgz" for n in range(1, 11)
)
# Only keyframe CAM_FRONT images live under this prefix (sweeps/ are non-keyframes).
CAM_FRONT_SAMPLES_PREFIX = "samples/CAM_FRONT/"
_BYTES_PER_GB = 1024 ** 3


# ---------------------------------------------------------------------------
# Shopping list: band filter + stratified sample
# ---------------------------------------------------------------------------


def frame_hazard_count(record: dict, mode: str = "hazard_class") -> int:
    """Count hazards in one metadata record.

    Args:
        record: A metadata.jsonl record with ``annotations`` (each carrying
            ``category_name`` and ``visibility_level``) and ``num_annotations``.
        mode:   ``"hazard_class"`` counts only annotations that map to a
            box-sourced hazard class (via :mod:`box_sourcing`); ``"num_annotations"``
            uses the raw agent count.

    Returns:
        The hazard count used for band filtering / stratification.
    """
    if mode == "num_annotations":
        return int(record.get("num_annotations", len(record.get("annotations", []))))
    count = 0
    for ann in record.get("annotations", []):
        vis = int(ann.get("visibility_level", 4) or 4)
        if nuscenes_category_to_hazard(ann.get("category_name", ""), vis) is not None:
            count += 1
    return count


def iter_metadata(path: str | Path) -> Iterator[dict]:
    """Yield records from a metadata JSONL file (skips blank lines)."""
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def in_band(count: int, band: tuple[int, int]) -> bool:
    """True if ``count`` is within the inclusive ``[lo, hi]`` hazard band."""
    return band[0] <= count <= band[1]


def local_image_exists(basename: str, cam_front_dir: str | Path) -> bool:
    """True if ``basename`` already exists under the local CAM_FRONT dir."""
    return (Path(cam_front_dir) / basename).exists()


def load_have_basenames(path: str | Path) -> set[str]:
    """Load already-have image basenames from a manifest (to subtract WITHOUT
    needing the physical image files on the box).

    Accepts a flexible format — one entry per line, each being a bare basename,
    a full path (basename taken), or a JSON object carrying ``basename`` or
    ``cam_front_path``. Blank lines are ignored.

    Args:
        path: Path to the have-manifest file.

    Returns:
        Set of image basenames already owned.
    """
    have: set[str] = set()
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line[0] == "{":
            rec = json.loads(line)
            ref = rec.get("basename") or rec.get("cam_front_path", "")
        else:
            ref = line
        if ref:
            have.add(Path(ref).name)
    return have


def band_frames_without_images(
    metadata_path: str | Path,
    cam_front_dir: str | Path,
    band: tuple[int, int],
    mode: str = "hazard_class",
    have_extra: set[str] | None = None,
) -> list[dict]:
    """Select in-band frames whose CAM_FRONT image is not already owned.

    A frame is subtracted if its basename is in ``have_extra`` (a manifest of
    already-owned basenames) OR its image physically exists under
    ``cam_front_dir``. Either path alone is sufficient — the manifest lets a box
    subtract the already-downloaded frames without transferring the image bytes.

    Args:
        metadata_path: Global keyframe metadata JSONL.
        cam_front_dir: Local ``samples/CAM_FRONT`` dir to test image presence.
        band:          Inclusive ``(lo, hi)`` hazard band.
        mode:          Hazard-count mode (see :func:`frame_hazard_count`).
        have_extra:    Optional set of already-owned basenames (from a manifest).

    Returns:
        Shopping-list dicts: ``basename``, ``sample_token``, ``scene_token``,
        ``hazard_count``, ``cam_front_path``.
    """
    have_extra = have_extra or set()
    out: list[dict] = []
    for rec in iter_metadata(metadata_path):
        count = frame_hazard_count(rec, mode)
        if not in_band(count, band):
            continue
        basename = Path(rec["cam_front_path"]).name
        if basename in have_extra or local_image_exists(basename, cam_front_dir):
            continue
        out.append({
            "basename": basename,
            "sample_token": rec.get("sample_token", ""),
            "scene_token": rec.get("scene_token", ""),
            "hazard_count": count,
            "cam_front_path": rec["cam_front_path"],
        })
    return out


def _bucket_index(count: int, edges: list[int]) -> int:
    """Return the stratum index for ``count`` given left-closed bucket ``edges``."""
    idx = 0
    for i in range(len(edges) - 1):
        if edges[i] <= count < edges[i + 1]:
            return i
        idx = i
    return idx


def stratified_sample(
    frames: list[dict],
    target_count: int,
    strata_edges: list[int],
    seed: int = 42,
) -> list[dict]:
    """Proportionally sample ``target_count`` frames across hazard-count strata.

    Proportional (not uniform) allocation keeps the band's natural, healthy
    distribution rather than skewing toward the densest frames. Deterministic
    for a fixed ``seed``.

    Args:
        frames:       Candidate shopping-list dicts (need ``hazard_count``).
        target_count: Desired sample size; ``<= 0`` or ``>= len(frames)`` returns all.
        strata_edges: Left-closed bucket edges, e.g. ``[3, 6, 9, 13, 21]``.
        seed:         RNG seed.

    Returns:
        The sampled frames (sorted by basename for reproducible output).
    """
    if target_count <= 0 or target_count >= len(frames):
        return sorted(frames, key=lambda f: f["basename"])
    rng = random.Random(seed)
    buckets: dict[int, list[dict]] = {}
    for f in frames:
        buckets.setdefault(_bucket_index(f["hazard_count"], strata_edges), []).append(f)
    total = len(frames)
    picked: list[dict] = []
    for idx in sorted(buckets):
        group = buckets[idx]
        take = round(target_count * len(group) / total)
        take = min(take, len(group))
        picked.extend(rng.sample(group, take) if take < len(group) else group)
    picked = _adjust_to_target(picked, frames, target_count, rng)
    return sorted(picked, key=lambda f: f["basename"])


def _adjust_to_target(
    picked: list[dict], frames: list[dict], target: int, rng: random.Random
) -> list[dict]:
    """Correct rounding drift so ``len(picked)`` matches ``target`` exactly."""
    if len(picked) > target:
        return rng.sample(picked, target)
    if len(picked) < target:
        chosen = {id(f) for f in picked}
        remaining = [f for f in frames if id(f) not in chosen]
        picked = picked + rng.sample(remaining, min(target - len(picked), len(remaining)))
    return picked


def write_shoppinglist(frames: list[dict], path: str | Path) -> None:
    """Write shopping-list dicts to a JSONL file (creating parent dirs)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as fh:
        for f in frames:
            fh.write(json.dumps(f) + "\n")


def load_shoppinglist(path: str | Path) -> list[dict]:
    """Load a shopping-list JSONL file into a list of dicts."""
    return list(iter_metadata(path))


def decide_rebuild_mode(
    list_exists: bool,
    have_count: int,
    have_source: str,
    force_rebuild: bool,
    no_rebuild: bool,
) -> tuple[bool, str]:
    """Decide whether to rebuild the shopping list or reuse a frozen one.

    Rebuild is IMPLICIT whenever there is something to subtract (a populated
    dataset or a have-manifest), so the denominator footgun — sampling from the
    full band and wasting target slots on frames you already own — cannot happen.
    ``no_rebuild`` is the explicit escape hatch for reusing a frozen list.

    Args:
        list_exists:   Whether a shopping-list file is already on disk.
        have_count:    Number of already-owned images available to subtract.
        have_source:   Human label for where ``have_count`` came from.
        force_rebuild: ``--rebuild-list`` was passed.
        no_rebuild:    ``--no-rebuild-list`` was passed (escape hatch).

    Returns:
        ``(rebuild, reason)`` — ``reason`` is a loud, human-readable mode string.
    """
    if no_rebuild:
        if list_exists:
            return False, "reusing FROZEN list (--no-rebuild-list; NOT subtracting owned images)"
        return True, "building fresh list (--no-rebuild-list set but no list on disk)"
    if force_rebuild:
        return True, f"rebuilding list against {have_count} already-owned images (--rebuild-list)"
    if have_count > 0:
        return True, f"rebuilding list against {have_count} already-owned images ({have_source})"
    if list_exists:
        return False, "reusing existing list (no owned images or --have-manifest to subtract)"
    return True, "building fresh list (no owned images or --have-manifest to subtract)"


def stratum_histogram(frames: list[dict], strata_edges: list[int]) -> dict[str, int]:
    """Return ``{"[lo,hi)": count}`` per stratum for reporting/logging."""
    hist: dict[str, int] = {}
    for f in frames:
        i = _bucket_index(f["hazard_count"], strata_edges)
        lo, hi = strata_edges[i], strata_edges[min(i + 1, len(strata_edges) - 1)]
        hist[f"[{lo},{hi})"] = hist.get(f"[{lo},{hi})", 0) + 1
    return dict(sorted(hist.items()))


# ---------------------------------------------------------------------------
# Bounded-storage streaming extraction
# ---------------------------------------------------------------------------


def stream_extract_blob(
    tar_path: str | Path,
    wanted: set[str],
    cam_front_dir: str | Path,
) -> tuple[list[str], int]:
    """Stream a blob tarball, extracting ONLY wanted CAM_FRONT keyframes.

    Uses tarfile streaming mode (``r|gz``) so the archive is read sequentially
    without loading it into memory or seeking. Only members under
    ``samples/CAM_FRONT/`` whose basename is in ``wanted`` are written; everything
    else is skipped. Already-present targets are not rewritten (idempotent).

    Args:
        tar_path:      Path to the ``.tgz`` blob.
        wanted:        Set of desired image basenames (the shopping list).
        cam_front_dir: Destination ``samples/CAM_FRONT`` dir.

    Returns:
        ``(newly_written_basenames, skipped_existing_count)``.
    """
    dest_dir = Path(cam_front_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    skipped = 0
    with tarfile.open(tar_path, "r|gz") as tar:  # streaming, bounded memory
        for member in tar:
            if not member.isfile() or CAM_FRONT_SAMPLES_PREFIX not in member.name:
                continue
            base = member.name.rsplit("/", 1)[-1]
            if base not in wanted:
                continue
            dest = dest_dir / base
            if dest.exists():
                skipped += 1
                continue
            src = tar.extractfile(member)
            if src is None:
                continue
            with open(dest, "wb") as fh:
                shutil.copyfileobj(src, fh, length=1 << 20)
            written.append(base)
    return written, skipped


def download_blob(
    url: str,
    dest: str | Path,
    headers: dict | None = None,
    cap_bytes: int | None = None,
    chunk: int = 8 << 20,
) -> int:
    """Stream-download a blob to ``dest``, aborting if it exceeds ``cap_bytes``.

    Args:
        url:       Signed (or token-authed) blob URL.
        dest:      Local tarball path to write.
        headers:   Optional request headers (e.g. token auth).
        cap_bytes: Abort if the download exceeds this many bytes (disk guard).
        chunk:     Streaming chunk size in bytes.

    Returns:
        Bytes written.
    """
    import requests  # noqa: PLC0415 — optional dep, only needed for real downloads

    written = 0
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, headers=headers or {}, stream=True, timeout=(30, 300)) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for block in r.iter_content(chunk):
                if not block:
                    continue
                fh.write(block)
                written += len(block)
                if cap_bytes is not None and written > cap_bytes:
                    raise RuntimeError(
                        f"blob exceeded disk cap ({written / _BYTES_PER_GB:.1f} GB) — aborting"
                    )
    return written


def dir_size_bytes(path: str | Path) -> int:
    """Total size in bytes of files directly under ``path`` (non-recursive is fine
    here — the temp dir only ever holds a single tarball)."""
    p = Path(path)
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.glob("*") if f.is_file())


def free_gb(path: str | Path) -> float:
    """Free space in GiB on the filesystem backing ``path`` (or its parent)."""
    target = Path(path)
    while not target.exists():
        target = target.parent
    return shutil.disk_usage(target).free / _BYTES_PER_GB


# ---------------------------------------------------------------------------
# nuScenes auth resolution (no hardcoded/guessed URLs)
# ---------------------------------------------------------------------------


def load_blob_urls(config: dict, cli_file: str | None) -> dict[str, str]:
    """Load the blob→signed-URL map from ``--blob-urls-file`` / env / config.

    Args:
        config:   The ``mining`` config dict.
        cli_file: Optional ``--blob-urls-file`` path (highest priority).

    Returns:
        ``{blob_filename: signed_url}`` (empty if none configured).
    """
    src = cli_file or os.environ.get("NUSCENES_BLOB_URLS") or config.get("blob_urls_file") or ""
    if not src:
        return {}
    path = Path(src)
    if not path.exists():
        logger.warning("blob_urls_file '%s' does not exist — ignoring", src)
        return {}
    return json.loads(path.read_text())


def blob_name_aliases(blob: str) -> list[str]:
    """Return ``blob`` plus its ``_blobs``/``_keyframes`` counterpart.

    nuScenes serves full ``_blobs.tgz`` (samples + sweeps) and, for some accounts,
    keyframe-only ``_keyframes.tgz``. Accepting either lets a ``urls.json`` or
    ``--blob-dir`` keyed by one name resolve a config that names the other, while
    the local tarball path stays a single consistent value (download = extract =
    delete), so the cleanup can't reference a path that was never written.

    Args:
        blob: A blob filename.

    Returns:
        ``[blob, counterpart]`` (counterpart omitted if neither suffix matches).
    """
    if "_blobs.tgz" in blob:
        return [blob, blob.replace("_blobs.tgz", "_keyframes.tgz")]
    if "_keyframes.tgz" in blob:
        return [blob, blob.replace("_keyframes.tgz", "_blobs.tgz")]
    return [blob]


def resolve_blob_source(
    blob: str,
    blob_dir: str | Path | None,
    url_map: dict[str, str],
    token: str | None,
    base_url: str,
) -> tuple[str, str]:
    """Resolve where a blob's bytes come from, in priority order.

    Priority: a local pre-downloaded tarball → a signed URL → token+base_url. Both
    the ``_blobs`` and ``_keyframes`` spellings are accepted (see
    :func:`blob_name_aliases`).

    Args:
        blob:     Blob filename, e.g. ``v1.0-trainval02_blobs.tgz``.
        blob_dir: Optional dir of pre-downloaded tarballs (offline mode).
        url_map:  ``{blob: signed_url}`` map.
        token:    Optional ``NUSCENES_TOKEN`` for legacy header auth.
        base_url: Base URL for token auth.

    Returns:
        ``(kind, ref)`` where kind is ``"local"``, ``"url"``, ``"token"``, or
        ``"missing"``. ``ref`` is a path, URL, or "" for missing.
    """
    aliases = blob_name_aliases(blob)
    if blob_dir:
        for name in aliases:
            local = Path(blob_dir) / name
            if local.exists():
                return "local", str(local)
    for name in aliases:
        if url_map.get(name):
            return "url", url_map[name]
    if token:
        return "token", f"{base_url.rstrip('/')}/{blob}"
    return "missing", ""


def auth_instructions() -> str:
    """Return the exact steps the user must follow to supply nuScenes blob access."""
    return (
        "nuScenes trainval blobs are NOT public — they need a nuScenes account, an\n"
        "accepted license, and are served via EXPIRING signed URLs. Supply ONE of:\n"
        "\n"
        "  OPTION 1 (recommended) — signed URLs:\n"
        "    1. Log in at https://www.nuscenes.org/nuscenes#download\n"
        "    2. Accept the Terms of Use / download agreement.\n"
        "    3. For each needed blob (trainval 02,03,05,06,07,09,10), right-click the\n"
        "       'Trainval blobs part NN' link and copy the signed URL.\n"
        "    4. Save a JSON map and pass it:\n"
        '         { \"v1.0-trainval02_blobs.tgz\": \"https://...signed...\", ... }\n'
        "       --blob-urls-file urls.json   (or export NUSCENES_BLOB_URLS=urls.json)\n"
        "    NOTE: signed URLs expire (hours) — refresh them if the job is slow.\n"
        "\n"
        "  OPTION 2 — official downloader, then point the miner at the tarballs:\n"
        "    Download the blobs with the nuScenes CLI / your browser into a dir, then\n"
        "       --blob-dir /path/to/tarballs   (fully offline; still one-blob-bounded)\n"
        "\n"
        "  OPTION 3 — legacy token header auth (may 403 in the signed-URL era):\n"
        "       export NUSCENES_TOKEN='...'   (base_url is set in configs/data.yaml)\n"
    )


# ---------------------------------------------------------------------------
# Resume manifest
# ---------------------------------------------------------------------------


class MiningManifest:
    """Blob-level resume state for the streaming miner.

    Tracks which blobs are fully processed so a re-run skips them. Per-frame
    idempotency is handled separately by on-disk image presence, so the manifest
    stays small. Persisted as a single JSON file.
    """

    def __init__(self, path: str | Path) -> None:
        """Load existing state from ``path`` if present, else start empty."""
        self.path = Path(path)
        self.completed_blobs: list[str] = []
        self.per_blob: dict[str, int] = {}
        self.disk_hwm_gb: float = 0.0
        if self.path.exists():
            data = json.loads(self.path.read_text())
            self.completed_blobs = data.get("completed_blobs", [])
            self.per_blob = data.get("per_blob", {})
            self.disk_hwm_gb = float(data.get("disk_hwm_gb", 0.0))

    def is_done(self, blob: str) -> bool:
        """True if ``blob`` was already fully processed."""
        return blob in self.completed_blobs

    def mark_done(self, blob: str, matched: int, hwm_gb: float) -> None:
        """Record a completed blob and persist immediately (crash-safe)."""
        if blob not in self.completed_blobs:
            self.completed_blobs.append(blob)
        self.per_blob[blob] = matched
        self.disk_hwm_gb = max(self.disk_hwm_gb, hwm_gb)
        self.save()

    def total_matched(self) -> int:
        """Total frames fetched across all recorded blobs."""
        return sum(self.per_blob.values())

    def save(self) -> None:
        """Persist manifest state atomically (temp file + rename)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({
            "completed_blobs": self.completed_blobs,
            "per_blob": self.per_blob,
            "disk_hwm_gb": round(self.disk_hwm_gb, 2),
        }, indent=2))
        tmp.replace(self.path)
