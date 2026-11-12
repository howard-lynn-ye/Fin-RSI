from types import SimpleNamespace

import pytest

import hashlib
import json

from scripts.qualify_learning_fit import (
    CountedBackend, verify_fit_usage, assert_qualified_runtime, accounting_from_events,
)


def test_fit_accounting_matches_only_the_actual_backend_boundary():
    class Backend:
        def generate_bounded(self, messages, *, max_new_tokens, seed):
            assert max_new_tokens == 19 and seed == 11
            return {"usage":dict(prompt_tokens=31, completion_tokens=7, total_tokens=38)}
    backend = CountedBackend(Backend())
    backend.generate_bounded([], max_new_tokens=19, seed=11)
    budget = SimpleNamespace(episodes=6, calls=18, generated_tokens=20000)
    usage = dict(episodes=1, calls=1, generated_tokens=7)
    verify_fit_usage(usage, backend.responses, backend.requests, budget)
    with pytest.raises(ValueError, match="token meter"):
        verify_fit_usage(dict(usage, generated_tokens=14), backend.responses, backend.requests, budget)
    with pytest.raises(ValueError, match="declared budget"):
        verify_fit_usage(dict(usage, episodes=7), backend.responses, backend.requests, budget)


def test_transport_failure_is_an_attempt_with_unknown_response_cost():
    class Backend:
        def generate_bounded(self, *args, **kwargs):
            raise ConnectionError("synthetic provider failure")
    backend = CountedBackend(Backend())
    with pytest.raises(ConnectionError):
        backend.generate_bounded([], max_new_tokens=19, seed=11)
    assert backend.requests == 1 and backend.responses == []
    with pytest.raises(ValueError, match="call meter"):
        verify_fit_usage(dict(episodes=1, calls=1, generated_tokens=0), [], 1,
                         SimpleNamespace(episodes=6, calls=18, generated_tokens=20000))


def test_runtime_binds_entrypoint_and_actual_adapter_tree(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    entry = source / "probe.py"
    adapter = source / "adapter.py"
    entry.write_text("# entry")
    adapter.write_text("# adapter")
    (source / "source-manifest.json").write_text(json.dumps({
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (entry, adapter)}))
    modules = {"benchmarks.agent_study.rsi_backend": SimpleNamespace(__file__=adapter)}
    assert_qualified_runtime(source, entry, modules)
    with pytest.raises(ValueError, match="outside qualified"):
        assert_qualified_runtime(source, entry, {
            "benchmarks.agent_study.rsi_backend": SimpleNamespace(__file__=tmp_path / "other.py")})
    with pytest.raises(ValueError, match="outside qualified"):
        assert_qualified_runtime(source, tmp_path / "other.py", modules)
    adapter.write_text("# changed")
    with pytest.raises(ValueError, match="differs from CPU manifest"):
        assert_qualified_runtime(source, entry, modules)


def test_failure_ledger_counts_actual_boundary_and_retains_unknown_cost(tmp_path):
    ledger = tmp_path / "events.jsonl"
    assert accounting_from_events(ledger)["started_requests"] is None
    rows = [dict(kind="fit_started", method="evolve", library=True, seed=11)]
    for mode in ("bounded", "nested"):
        rows.extend([dict(kind="call_started", backend_mode=mode),
                     dict(kind="call_completed", backend_mode=mode, tokens=7)])
    rows.extend([dict(kind="call_started", backend_mode="bounded"),
                 dict(kind="call_failed", backend_mode="bounded", token_cost=None),
                 dict(kind="call_started", backend_mode="bounded")])
    ledger.write_text("".join(json.dumps(row)+"\n" for row in rows) + '{"kind":')
    counts = accounting_from_events(ledger)
    assert counts["started_requests"] == 3
    assert counts["known_generated_tokens"] == 7
    assert counts["unknown_cost_requests"] == 2
    assert counts["unfinished_requests"] == 1
    assert counts["incomplete_final_event"]
    assert counts["current_fit"] == dict(method="evolve", library=True, seed=11)
