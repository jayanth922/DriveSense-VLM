"""Categorical drift detection over DriveSense-VLM stratification metadata.

No live production data required: :class:`DriftMonitor` compares a REFERENCE
categorical distribution (e.g. built from the training set's weather /
time_of_day / location / hazard-class frequencies) against an INCOMING batch's
distribution, using the Population Stability Index (PSI) — a standard,
dependency-free drift metric used across the ML industry for exactly this
(no scipy needed; PSI is just weighted log-ratios over category proportions).

Operates on SFT-format records (``sft_test_enriched.jsonl`` and friends):
top-level ``weather`` / ``time_of_day`` / ``location`` fields (falling back to
``metadata``/``ego_context`` nesting, matching
:func:`drivesense.eval.robustness._extract_stratum_value`), plus a
``hazard_class`` dimension parsed from each record's assistant-turn hazard list.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Iterable

# Avoids log(0) / division-by-zero for categories with zero observed frequency
# in one of the two distributions being compared.
_EPS = 1e-4

# Standard industry PSI interpretation bands (Population Stability Index).
PSI_NO_DRIFT = 0.10
PSI_MODERATE_DRIFT = 0.20

DEFAULT_DIMENSIONS: tuple[str, ...] = ("weather", "time_of_day", "location", "hazard_class")


# ---------------------------------------------------------------------------
# Field extraction (SFT record -> stratification values)
# ---------------------------------------------------------------------------


def _parse_assistant_json(record: dict) -> dict | None:
    """Extract the parsed assistant-turn JSON from an SFT ``messages`` record."""
    for msg in record.get("messages", []):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if isinstance(content, dict):
            return content
        if isinstance(content, str):
            m = re.search(r"\{.*\}", content, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    return None
    return None


def _hazard_labels_of(record: dict) -> list[str]:
    """Return every hazard ``label`` in a record (direct or SFT-messages format)."""
    if "hazards" in record and isinstance(record["hazards"], list):
        hazards = record["hazards"]
    else:
        parsed = _parse_assistant_json(record)
        hazards = parsed.get("hazards", []) if parsed else []
    return [str(h.get("label", "unknown")).lower() for h in hazards if isinstance(h, dict)]


def extract_dimension_values(record: dict, dimension: str) -> list[str]:
    """Return the stratification value(s) for one dimension from an SFT record.

    ``hazard_class`` can yield MULTIPLE values (a frame can have several
    hazards; a frame with none is counted as ``"no_hazard"``). All other
    dimensions yield exactly one value, checked in ``metadata`` -> ``ego_context``
    -> top-level order (mirroring
    :func:`drivesense.eval.robustness._extract_stratum_value`).

    Args:
        record:    One SFT-format record.
        dimension: ``"weather"``, ``"time_of_day"``, ``"location"``, or
                   ``"hazard_class"``.

    Returns:
        A non-empty list of lowercase category strings.
    """
    if dimension == "hazard_class":
        labels = _hazard_labels_of(record)
        return labels or ["no_hazard"]
    val = (
        record.get("metadata", {}).get(dimension)
        or record.get("ego_context", {}).get(dimension)
        or record.get(dimension)
    )
    return [str(val).lower()] if val else ["unknown"]


# ---------------------------------------------------------------------------
# PSI
# ---------------------------------------------------------------------------


def category_distribution(values: Iterable[str]) -> dict[str, float]:
    """Normalise raw category values into a ``{category: proportion}`` distribution."""
    counts = Counter(str(v) for v in values)
    total = sum(counts.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counts.items()}


def category_distribution_from_records(records: list[dict], dimension: str) -> dict[str, float]:
    """Build a category distribution for one dimension across a list of records."""
    values = [v for r in records for v in extract_dimension_values(r, dimension)]
    return category_distribution(values)


def population_stability_index(reference: dict[str, float], incoming: dict[str, float]) -> float:
    """Compute PSI between two categorical distributions (proportions, not counts).

    ``PSI = sum_i (incoming_i - reference_i) * ln(incoming_i / reference_i)`` over
    the union of categories seen in either distribution. A category missing from
    one distribution is treated as :data:`_EPS` (never exactly 0) so the log stays
    defined — this also means a category that appears ONLY in the incoming batch
    (never seen in the reference) correctly contributes a large PSI term, which is
    itself a meaningful drift signal (a genuinely new category showing up).

    Interpretation (standard industry bands — see :data:`PSI_NO_DRIFT` /
    :data:`PSI_MODERATE_DRIFT`): < 0.10 no significant change, 0.10-0.20
    moderate (worth watching), > 0.20 significant distribution shift.

    Args:
        reference: ``{category: proportion}`` for the reference population.
        incoming:  ``{category: proportion}`` for the incoming batch.

    Returns:
        The PSI value (>= 0; 0 means identical distributions).
    """
    categories = set(reference) | set(incoming)
    psi = 0.0
    for cat in categories:
        ref_p = max(reference.get(cat, 0.0), _EPS)
        inc_p = max(incoming.get(cat, 0.0), _EPS)
        psi += (inc_p - ref_p) * math.log(inc_p / ref_p)
    return round(psi, 6)


def psi_severity(psi: float) -> str:
    """Classify a PSI value into a human-readable severity band."""
    if psi < PSI_NO_DRIFT:
        return "none"
    if psi < PSI_MODERATE_DRIFT:
        return "moderate"
    return "significant"


# ---------------------------------------------------------------------------
# DriftMonitor
# ---------------------------------------------------------------------------


class DriftMonitor:
    """Tracks per-dimension categorical drift via PSI against a reference distribution.

    The reference is built ONCE — e.g. from the training set's stratification
    metadata at model-promotion time — and reused for every incoming batch. This
    mirrors a real monitoring pipeline: score rolling production batches against
    a fixed reference without needing live ground-truth labels (PSI only needs
    the categorical metadata, not correctness).
    """

    def __init__(
        self, reference: dict[str, dict[str, float]], threshold: float = PSI_MODERATE_DRIFT,
    ) -> None:
        """
        Args:
            reference: ``{dimension: {category: proportion}}``, e.g.
                ``{"weather": {"clear": 0.9, "rain": 0.1}, ...}``.
            threshold: PSI value at/above which a dimension is flagged as drifted.
        """
        self.reference = reference
        self.threshold = threshold

    @classmethod
    def from_records(
        cls, records: list[dict], dimensions: Iterable[str] = DEFAULT_DIMENSIONS,
        threshold: float = PSI_MODERATE_DRIFT,
    ) -> "DriftMonitor":
        """Build a monitor whose reference distribution is computed FROM ``records``."""
        reference = {
            dim: category_distribution_from_records(records, dim) for dim in dimensions
        }
        return cls(reference, threshold=threshold)

    def check(self, incoming_records: list[dict]) -> dict[str, dict]:
        """Score an incoming batch against the reference; return a per-dimension report.

        Args:
            incoming_records: SFT-format records for the batch being checked.

        Returns:
            ``{dimension: {"psi", "severity", "drifted", "reference_distribution",
            "incoming_distribution"}}``.
        """
        report: dict[str, dict] = {}
        for dim, ref_dist in self.reference.items():
            incoming_dist = category_distribution_from_records(incoming_records, dim)
            psi = population_stability_index(ref_dist, incoming_dist)
            report[dim] = {
                "psi": psi,
                "severity": psi_severity(psi),
                "drifted": psi >= self.threshold,
                "reference_distribution": ref_dist,
                "incoming_distribution": incoming_dist,
            }
        return report

    @staticmethod
    def any_drifted(report: dict[str, dict]) -> bool:
        """True if any dimension in a :meth:`check` report was flagged as drifted."""
        return any(v["drifted"] for v in report.values())
