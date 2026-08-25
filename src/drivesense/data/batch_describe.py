"""Message Batches API describe pass — 50% cheaper than real-time, async is fine.

The describe pass (severity/reasoning/action per already-localized hazard) is
cost-dominated, not latency-sensitive, so it runs through Anthropic's Message
Batches API. This module holds the cost-optimising, testable pieces:

- ``build_request``    — one batch request (image + given-hazards prompt) per frame.
- ``chunk_jobs``       — split frames so each batch stays under the 256 MB / 100k caps.
- ``parse_batch_text`` — extract the JSON object from a model reply.
- ``BatchState``       — persist submitted batch ids so a restart resumes, never
                         resubmits (no double charge).
- ``drain_existing`` / ``submit_new`` — poll + collect results, invoking a caller
                         callback that writes the per-frame resume cache.

The Anthropic client is injected (never imported here) so the orchestration is
unit-testable with a fake client and needs no SDK at import time.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

# Batch caps: 100k requests / 256 MB per batch. Stay well under both. Images
# dominate size, so we also cap by estimated bytes (on-disk size × base64 blowup).
_MAX_REQ_PER_BATCH = 1000
_MAX_BYTES_PER_BATCH = 200 * 1024 * 1024
_B64_BLOWUP = 1.4
_POLL_START_S, _POLL_MAX_S = 15.0, 60.0

# A frame job is (sample_token, image_path, kept_hazards).
Job = tuple[str, str, list[dict]]


def build_request(
    token: str, img_path: str, hazards: list[dict], model: str, system: str,
    max_tokens: int = 4096,
) -> dict:
    """Build one Message Batches request for a frame (image + given-hazards prompt).

    Args:
        token:     Sample token, used as the batch ``custom_id``.
        img_path:  Path to the CAM_FRONT image.
        hazards:   GT-sourced hazards (label + optional bbox_2d) to describe in order.
        model:     Model id (e.g. ``claude-sonnet-5``).
        system:    System prompt.
        max_tokens: Output cap.

    Returns:
        A ``{custom_id, params}`` dict accepted by ``messages.batches.create``.
    """
    with open(img_path, "rb") as f:
        b64 = base64.standard_b64encode(f.read()).decode()
    given = [{"label": h["label"], **({"bbox_2d": h["bbox_2d"]} if "bbox_2d" in h else {})}
             for h in hazards]
    text = ("Given hazards (keep order): " + json.dumps(given)
            + "\nFill severity/reasoning/action for each in the SAME order. JSON only.")
    return {
        "custom_id": token,
        "params": {
            "model": model, "max_tokens": max_tokens, "system": system,
            "messages": [{"role": "user", "content": [
                {"type": "image",
                 "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": text},
            ]}],
        },
    }


def chunk_jobs(
    jobs: list[Job], max_count: int = _MAX_REQ_PER_BATCH, max_bytes: int = _MAX_BYTES_PER_BATCH,
) -> list[list[Job]]:
    """Split frame jobs so each batch stays under the request-count and size caps.

    Bytes are estimated from each image's on-disk size (× base64 blowup) so we
    never build a > 256 MB batch. Images are read lazily per chunk by the caller,
    keeping peak memory to roughly one batch.

    Args:
        jobs:      ``(token, img_path, hazards)`` tuples.
        max_count: Max requests per batch.
        max_bytes: Max estimated bytes per batch.

    Returns:
        A list of job chunks.
    """
    chunks: list[list[Job]] = []
    cur: list[Job] = []
    cur_bytes = 0
    for job in jobs:
        jb = int(Path(job[1]).stat().st_size * _B64_BLOWUP) + 512
        if cur and (len(cur) >= max_count or cur_bytes + jb > max_bytes):
            chunks.append(cur)
            cur, cur_bytes = [], 0
        cur.append(job)
        cur_bytes += jb
    if cur:
        chunks.append(cur)
    return chunks


def parse_batch_text(text: str) -> dict | None:
    """Extract the first JSON object from a model reply, or ``None``."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


class BatchState:
    """Persist submitted batch ids so a restart resumes rather than resubmitting."""

    def __init__(self, path: str | Path) -> None:
        """Load any recorded batch ids from ``path``."""
        self.path = Path(path)
        self.ids: list[str] = []
        if self.path.exists():
            self.ids = json.loads(self.path.read_text()).get("batches", [])

    def add(self, batch_id: str) -> None:
        """Record a submitted batch id and persist immediately (crash-safe)."""
        if batch_id not in self.ids:
            self.ids.append(batch_id)
            self.save()

    def save(self) -> None:
        """Write the batch-id list to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"batches": self.ids}))


def _poll_and_collect(client: object, batch_id: str, on_result: Callable[[str, dict], None]) -> int:
    """Poll one batch to completion, then feed each success to ``on_result``."""
    batch = client.messages.batches.retrieve(batch_id)  # type: ignore[attr-defined]
    poll = _POLL_START_S
    while batch.processing_status != "ended":
        logger.info("batch %s: %s (%s)", batch_id, batch.processing_status,
                    getattr(batch, "request_counts", ""))
        time.sleep(poll)
        poll = min(poll * 1.5, _POLL_MAX_S)
        batch = client.messages.batches.retrieve(batch_id)  # type: ignore[attr-defined]
    n = 0
    for res in client.messages.batches.results(batch_id):  # type: ignore[attr-defined]
        if res.result.type != "succeeded":
            continue
        _text = next((b.text for b in res.result.message.content
                      if getattr(b, "type", None) == "text"), "")
        vlm = parse_batch_text(_text)
        if vlm is not None:
            on_result(res.custom_id, vlm)
            n += 1
    logger.info("batch %s: wrote %d results", batch_id, n)
    return n


def drain_existing(
    client: object, state: BatchState, on_result: Callable[[str, dict], None],
) -> None:
    """Resume: poll every already-submitted batch and collect its results."""
    for bid in list(state.ids):
        try:
            _poll_and_collect(client, bid, on_result)
        except Exception as exc:  # noqa: BLE001 — a stale/expired id must not abort the run
            logger.warning("could not drain batch %s: %s", bid, exc)


def submit_new(
    client: object, jobs: list[Job], state: BatchState, model: str, system: str,
    on_result: Callable[[str, dict], None],
) -> None:
    """Chunk jobs into batches, submit + record each id, then poll and collect.

    The batch id is persisted BEFORE polling so a crash mid-run never resubmits
    the same frames (no double charge). Images are read per chunk, bounding memory.
    """
    for chunk in chunk_jobs(jobs):
        reqs = [build_request(t, img, hz, model, system) for (t, img, hz) in chunk]
        batch = client.messages.batches.create(requests=reqs)  # type: ignore[attr-defined]
        state.add(batch.id)
        logger.info("submitted batch %s (%d requests)", batch.id, len(reqs))
        _poll_and_collect(client, batch.id, on_result)
