"""Closed-loop mining target selection: score un-mined nuScenes candidate
frames by how well they match the WORST-performing failure-stratification
bucket (see ``drivesense.eval.failure_stratification``), so the next mining
pass is driven by measured failure, not manual guessing.

Candidate frames only have ``metadata.jsonl`` fields available (no rendered
image, no projected 2D box yet), so every signal here is a PROXY, honestly
documented as such:

- Size tier -> ``distance_to_ego`` per annotation (pinhole projection means
  closer objects occupy more of the frame; farther objects less — this is a
  heuristic, not a measured pixel size).
- Weather / time_of_day -> keyword search over ``scene_description``, the
  same convention already used elsewhere in this codebase
  (``configs/data.yaml``'s ``adverse_weather_keywords`` and
  ``regenerate_annotations_v2_colab.py``'s ``scene_meta()``).
- ``location`` (road_type, e.g. intersection/urban/parking) is NOT derivable
  from ``metadata.jsonl`` at all — no such field exists there — so it is
  explicitly unsupported and dropped from a target spec with a logged notice,
  rather than silently ignored.

Reuses ``drivesense.data.streaming_miner`` for metadata iteration, the 3-20
hazard band, already-mined exclusion, and the shopping-list output format, and
``drivesense.data.box_sourcing.nuscenes_category_to_hazard`` for identifying
which annotations are hazard-relevant at all — none of that is reimplemented.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Iterable

from drivesense.data.box_sourcing import nuscenes_category_to_hazard
from drivesense.data.streaming_miner import (
    frame_hazard_count,
    in_band,
    iter_metadata,
    local_image_exists,
)

logger = logging.getLogger(__name__)

# Approximate distance -> size-tier bands (metres). Mirrors
# drivesense.eval.failure_stratification.SIZE_TIERS in NAME only — these are a
# heuristic proxy for a candidate frame's likely box size, not measured area.
DISTANCE_SIZE_BANDS: tuple[tuple[str, float, float | None], ...] = (
    ("large", 0.0, 8.0),
    ("medium", 8.0, 20.0),
    ("small", 20.0, 40.0),
    ("tiny", 40.0, None),
)

# Mirrors configs/data.yaml's adverse_weather_keywords + regenerate_annotations_v2_colab.py's
# scene_meta() convention for deriving weather/time_of_day from free-text scene_description.
_WEATHER_KEYWORDS: dict[str, tuple[str, ...]] = {
    "rain": ("rain",),
    "fog": ("fog", "storm"),
}
_NIGHT_KEYWORDS: tuple[str, ...] = ("night",)

# Stratification dimensions this module CAN score candidates on.
SUPPORTED_DIMENSIONS: frozenset[str] = frozenset({"size_tier", "weather", "time_of_day"})
UNSUPPORTED_DIMENSIONS: frozenset[str] = frozenset({"location"})


def proxied_size_tier(distance_to_ego: float) -> str:
    """Map a distance (metres) to an approximate size tier (see module docstring)."""
    for name, lo, hi in DISTANCE_SIZE_BANDS:
        if distance_to_ego >= lo and (hi is None or distance_to_ego < hi):
            return name
    return DISTANCE_SIZE_BANDS[-1][0]


def infer_weather(scene_description: str) -> str:
    """Keyword-match weather from scene_description (mirrors scene_meta())."""
    d = scene_description.lower()
    for weather, keywords in _WEATHER_KEYWORDS.items():
        if any(kw in d for kw in keywords):
            return weather
    return "clear"


def infer_time_of_day(scene_description: str) -> str:
    """Keyword-match time_of_day from scene_description (mirrors scene_meta())."""
    d = scene_description.lower()
    return "night" if any(kw in d for kw in _NIGHT_KEYWORDS) else "day"


def hazard_annotations(record: dict) -> list[dict]:
    """Return only the annotations that map to a box-sourced hazard class."""
    return [
        a for a in record.get("annotations", [])
        if nuscenes_category_to_hazard(a.get("category_name", ""),
                                       int(a.get("visibility_level", 4) or 4)) is not None
    ]


def parse_bucket_key(bucket_key: str) -> dict[str, str]:
    """Parse a failure-stratification bucket key into a target spec.

    Bucket keys look like ``"weather:tiny|rain"`` (dimension:size_tier|value)
    or a marginal row like ``"weather:tiny|ALL"`` / ``"weather:ALL|rain"``.
    ``"ALL"`` on either side means "no constraint on that axis" and is dropped.

    Args:
        bucket_key: A key from a failure-stratification report's cross_tabs
            (as ``"<dimension>:<key>"`` — see how build_report/rank_buckets
            name entries) — or a bare ``"<size_tier>|<value>"`` cross-tab key.

    Returns:
        A target spec dict with a subset of ``{"size_tier", "weather",
        "time_of_day", "location"}`` keys (only non-"ALL" axes present).
    """
    dim, sep, rest = bucket_key.partition(":")
    if not sep:  # bare "tier|value" with no leading "dimension:"
        rest = bucket_key
        dim = ""
    tier, _, value = rest.partition("|")
    spec: dict[str, str] = {}
    if tier and tier != "ALL":
        spec["size_tier"] = tier
    if dim and value and value != "ALL":
        spec[dim] = value
    return spec


def clean_target_spec(spec: dict[str, str]) -> dict[str, str]:
    """Drop unsupported dimensions (e.g. ``location``) from a target spec, logging why."""
    dropped = {k: v for k, v in spec.items() if k in UNSUPPORTED_DIMENSIONS}
    if dropped:
        logger.warning(
            "Target spec dimension(s) %s not derivable from metadata.jsonl (no such "
            "field) — dropped. Scoring on: %s", dropped,
            {k: v for k, v in spec.items() if k in SUPPORTED_DIMENSIONS},
        )
    return {k: v for k, v in spec.items() if k in SUPPORTED_DIMENSIONS}


def worst_bucket_to_target_spec(report: dict) -> dict[str, str]:
    """Pick the top worst_buckets entry from a stratification report and clean it.

    Args:
        report: A ``build_report()``-shaped dict (has ``"worst_buckets"``).

    Returns:
        A supported target spec (see :func:`clean_target_spec`). Empty if the
        report has no worst buckets or none carry a supported dimension.
    """
    worst = report.get("worst_buckets", [])
    if not worst:
        return {}
    return clean_target_spec(parse_bucket_key(worst[0]["bucket"]))


def score_frame(record: dict, target: dict[str, str]) -> float:
    """Score how well one candidate frame matches a target spec, in [0, 1].

    Averages a per-axis match fraction over every axis present in ``target``
    (only :data:`SUPPORTED_DIMENSIONS` are ever scored — call
    :func:`clean_target_spec` first if ``target`` might carry others).

    Args:
        record: One ``metadata.jsonl`` record.
        target: Target spec from :func:`worst_bucket_to_target_spec` (or built
            by hand), e.g. ``{"size_tier": "large", "weather": "rain"}``.

    Returns:
        A score in ``[0, 1]``; ``0.0`` if the frame has no hazard-relevant
        annotations at all, or ``target`` is empty.
    """
    if not target:
        return 0.0
    anns = hazard_annotations(record)
    if not anns:
        return 0.0

    axis_scores: list[float] = []
    if "size_tier" in target:
        matches = sum(
            1 for a in anns
            if proxied_size_tier(float(a.get("distance_to_ego", 999.0))) == target["size_tier"]
        )
        axis_scores.append(matches / len(anns))
    if "weather" in target:
        axis_scores.append(1.0 if infer_weather(record.get("scene_description", ""))
                           == target["weather"] else 0.0)
    if "time_of_day" in target:
        axis_scores.append(1.0 if infer_time_of_day(record.get("scene_description", ""))
                           == target["time_of_day"] else 0.0)

    return sum(axis_scores) / len(axis_scores) if axis_scores else 0.0


def select_targets(
    metadata_path: str | Path,
    target: dict[str, str],
    already_have: set[str] | None = None,
    cam_front_dir: str | Path | None = None,
    band: tuple[int, int] = (3, 20),
    hazard_count_mode: str = "hazard_class",
    target_count: int = 2000,
    min_score: float = 0.0,
    seed: int = 42,
) -> list[dict]:
    """Score and select the top-matching un-mined candidates for a target spec.

    Args:
        metadata_path:      Global ``metadata.jsonl`` (all keyframes).
        target:              Target spec (see :func:`worst_bucket_to_target_spec`).
        already_have:        Basenames already mined (subtracted without needing
                             the physical images — see streaming_miner's
                             ``--have-manifest``).
        cam_front_dir:       If given, also excludes frames whose image
                             physically exists there.
        band:                Inclusive hazard-count band (matches the existing
                             mining criteria — still real, in-scope hazard frames).
        hazard_count_mode:   ``"hazard_class"`` or ``"num_annotations"``.
        target_count:        How many top-scoring frames to select.
        min_score:           Drop candidates scoring below this (0 = keep any
                             frame with a hazard-relevant annotation).
        seed:                Tie-break shuffle seed (deterministic).

    Returns:
        Shopping-list rows: ``{"basename", "sample_token", "scene_token",
        "hazard_count", "cam_front_path", "mining_score"}`` — a strict
        superset of the existing shopping-list schema (extra ``mining_score``
        field for traceability; ``run_streaming_miner.py`` only reads
        ``basename``, so this is a drop-in shopping list).
    """
    already_have = already_have or set()
    scored: list[dict] = []
    for rec in iter_metadata(metadata_path):
        count = frame_hazard_count(rec, hazard_count_mode)
        if not in_band(count, band):
            continue
        basename = Path(rec["cam_front_path"]).name
        if basename in already_have:
            continue
        if cam_front_dir and local_image_exists(basename, cam_front_dir):
            continue
        score = score_frame(rec, target)
        if score < min_score:
            continue
        scored.append({
            "basename": basename,
            "sample_token": rec.get("sample_token", ""),
            "scene_token": rec.get("scene_token", ""),
            "hazard_count": count,
            "cam_front_path": rec["cam_front_path"],
            "mining_score": round(score, 4),
        })

    # Sort by score desc; break ties deterministically (shuffle within equal-
    # score groups so target_count doesn't always favor metadata file order).
    rng = random.Random(seed)
    rng.shuffle(scored)
    scored.sort(key=lambda r: -r["mining_score"])
    return scored[: max(0, target_count)] if target_count else scored


def score_histogram(
    rows: Iterable[dict], edges: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.01),
) -> dict[str, int]:
    """Bucket a selected list's ``mining_score`` values for a quick sanity summary."""
    hist: dict[str, int] = {}
    for r in rows:
        s = r["mining_score"]
        for i in range(len(edges) - 1):
            if edges[i] <= s < edges[i + 1]:
                key = f"[{edges[i]:.2f},{edges[i + 1]:.2f})"
                hist[key] = hist.get(key, 0) + 1
                break
    return dict(sorted(hist.items()))
