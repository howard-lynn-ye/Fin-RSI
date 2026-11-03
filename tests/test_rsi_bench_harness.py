"""benchmarks.agent_study.rsi_bench - the Beacon qualification harness, with a fake backend."""
from __future__ import annotations

import json

import pytest

from benchmarks.agent_study import rsi_bench
from tests.test_rsi_methods import LessonModel


class FakeBackend:
    def __init__(self):
        self.inner, self.seed = LessonModel(learns_from_feedback=True), 0
        self.hardware, self.context_limit = {"gpus": ["fake"]}, 32768

    def __call__(self, messages):
        text = self.inner(messages)
        return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": len(text) // 4}, "generation_seconds": 0.0}


def test_qualify_one_records_every_episode_and_never_overwrites(tmp_path, monkeypatch):
    monkeypatch.delenv("FIN_TASKS_CONFINE", raising=False)
    record = rsi_bench.qualify_one(tmp_path, "fake", "org/fake", "rev",
                                   backend_factory=lambda m, r: FakeBackend())
    assert record["loaded"] and len(record["episodes"]) == 8
    assert {e["task_id"] for e in record["episodes"]} == set(rsi_bench.QUALIFY_TASKS)
    s = record["summary"]
    assert s["responses"] == sum(len(e["responses"]) for e in record["episodes"])
    assert s["fenced"] == s["responses"] and s["infrastructure_errors"] == 0
    # The fake writes library calls: accepted with the library, refused without it.
    assert {(e["library"], e["method"]): e["valid"] for e in record["episodes"]
            if e["task_id"] == "momentum_a"} == {
        (False, "no_improve"): False, (False, "check_feedback"): False,
        (True, "no_improve"): False, (True, "check_feedback"): True}
    stored = json.loads((tmp_path / "qualification" / "fake.json").read_text())
    again = rsi_bench.qualify_one(tmp_path, "fake", "org/fake", "rev",
                                  backend_factory=lambda m, r: pytest.fail("reran"))
    assert again == stored


def test_load_failure_is_recorded_not_raised(tmp_path):
    def broken(model, revision):
        raise OSError("weights missing")
    record = rsi_bench.qualify_one(tmp_path, "broken", "org/x", "rev", backend_factory=broken)
    assert record["loaded"] is False and "weights missing" in record["load_error"]


def test_inventory_keeps_cached_entries_only(tmp_path):
    path = tmp_path / "inv.json"
    path.write_text(json.dumps([
        dict(id="a", model="org/a", revision="r1", cached=True, cache_check=dict(cache="/c1")),
        dict(id="b", model="org/b", revision="r2", cached=False)]))
    assert rsi_bench.inventory(path) == {"a": ("org/a", "r1", "/c1")}
