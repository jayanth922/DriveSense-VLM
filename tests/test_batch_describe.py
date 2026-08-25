"""Tests for the Message Batches describe pass (CPU-only, no network / no SDK)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from drivesense.data import batch_describe as bd


# ---------------------------------------------------------------------------
# Fake Anthropic client (records created batches, returns canned results)
# ---------------------------------------------------------------------------


def _ok_result(cid: str, text: str) -> SimpleNamespace:
    # Current Sonnet models lead with a thinking block, so the text block is
    # not necessarily content[0] — mirror that here rather than assume it is.
    content = [
        SimpleNamespace(type="thinking", thinking="..."),
        SimpleNamespace(type="text", text=text),
    ]
    return SimpleNamespace(
        custom_id=cid,
        result=SimpleNamespace(type="succeeded",
                               message=SimpleNamespace(content=content)))


def _err_result(cid: str) -> SimpleNamespace:
    return SimpleNamespace(custom_id=cid, result=SimpleNamespace(type="errored"))


class _FakeBatches:
    def __init__(self, reply_text='{"hazards": [{"severity": "high"}]}', fail: set | None = None):
        self.created: list = []
        self._results: dict[str, list] = {}
        self._reply = reply_text
        self._fail = fail or set()

    def create(self, requests):
        bid = f"batch_{len(self.created)}"
        self.created.append((bid, requests))
        self._results[bid] = [
            _err_result(r["custom_id"]) if r["custom_id"] in self._fail
            else _ok_result(r["custom_id"], self._reply)
            for r in requests
        ]
        return SimpleNamespace(id=bid, processing_status="ended", request_counts=None)

    def retrieve(self, bid):
        return SimpleNamespace(id=bid, processing_status="ended", request_counts=None)

    def results(self, bid):
        return iter(self._results[bid])


class _FakeClient:
    def __init__(self, **kw):
        self.messages = SimpleNamespace(batches=_FakeBatches(**kw))


def _img(tmp_path: Path, name: str, size: int = 32) -> str:
    p = tmp_path / name
    p.write_bytes(b"\xff\xd8" + b"x" * size)  # jpeg-ish bytes
    return str(p)


# ---------------------------------------------------------------------------
# Pure pieces
# ---------------------------------------------------------------------------


def test_build_request_shape(tmp_path):
    img = _img(tmp_path, "a.jpg")
    req = bd.build_request("tok1", img, [{"label": "cyclist_proximity", "bbox_2d": [1, 2, 3, 4]}],
                           "claude-sonnet-5", "SYS")
    assert req["custom_id"] == "tok1"
    assert req["params"]["model"] == "claude-sonnet-5"
    content = req["params"]["messages"][0]["content"]
    assert content[0]["type"] == "image" and content[0]["source"]["type"] == "base64"
    assert "cyclist_proximity" in content[1]["text"]


def test_chunk_jobs_respects_count_and_bytes(tmp_path):
    jobs = [(f"t{i}", _img(tmp_path, f"{i}.jpg", size=10), []) for i in range(5)]
    assert len(bd.chunk_jobs(jobs, max_count=2, max_bytes=10 ** 9)) == 3  # 2+2+1 by count
    # tiny byte cap forces one job per chunk
    assert len(bd.chunk_jobs(jobs, max_count=100, max_bytes=1)) == 5


def test_parse_batch_text():
    assert bd.parse_batch_text('prefix {"hazards": []} suffix') == {"hazards": []}
    assert bd.parse_batch_text("no json here") is None
    assert bd.parse_batch_text("{bad json}") is None


def test_batch_state_persists(tmp_path):
    p = tmp_path / "batch_state.json"
    s = bd.BatchState(p)
    s.add("batch_0")
    s.add("batch_0")  # idempotent
    s.add("batch_1")
    assert bd.BatchState(p).ids == ["batch_0", "batch_1"]


# ---------------------------------------------------------------------------
# Orchestration (with the fake client)
# ---------------------------------------------------------------------------


def test_submit_new_collects_results_and_records_ids(tmp_path):
    client = _FakeClient()
    state = bd.BatchState(tmp_path / "state.json")
    jobs = [(f"t{i}", _img(tmp_path, f"{i}.jpg"), []) for i in range(3)]
    got: dict[str, dict] = {}
    bd.submit_new(client, jobs, state, "claude-sonnet-5", "SYS",
                  lambda cid, vlm: got.__setitem__(cid, vlm))
    assert set(got) == {"t0", "t1", "t2"}
    assert got["t0"] == {"hazards": [{"severity": "high"}]}
    assert state.ids and bd.BatchState(tmp_path / "state.json").ids == state.ids  # persisted


def test_errored_result_is_skipped(tmp_path):
    client = _FakeClient(fail={"t1"})
    state = bd.BatchState(tmp_path / "state.json")
    jobs = [(f"t{i}", _img(tmp_path, f"{i}.jpg"), []) for i in range(2)]
    got: dict[str, dict] = {}
    bd.submit_new(client, jobs, state, "m", "SYS", lambda cid, vlm: got.__setitem__(cid, vlm))
    assert set(got) == {"t0"}  # t1 errored -> not collected -> retried next run


def test_drain_existing_polls_recorded_batches(tmp_path):
    client = _FakeClient()
    state = bd.BatchState(tmp_path / "state.json")
    # pretend a batch was submitted in a prior (crashed) run
    client.messages.batches.create([{"custom_id": "t9", "params": {}}])
    state.add("batch_0")
    got: dict[str, dict] = {}
    bd.drain_existing(client, state, lambda cid, vlm: got.__setitem__(cid, vlm))
    assert "t9" in got  # resumed without resubmitting


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
