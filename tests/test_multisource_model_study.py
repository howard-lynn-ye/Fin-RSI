import pytest

pytest.importorskip("bs4")
from benchmarks.agent_study.multisource_model_study import (
    COMMON, GUIDE, aggregate, evidence_packet, evidence_text, freeze,
)


def fixtures():
    return [dict(category=kind, series="fixture", id=kind, event_date="2025-01-01",
                 eligible_date="2025-01-02", full_text="not in prompt", text="fixture observation")
            for kind in ("news", "psychology", "behavior", "officials")]


def test_all_four_categories_are_delivered_without_future_record():
    records = fixtures()
    records += [dict(records[0], id="future", eligible_date="2025-01-10", text="future secret")]
    packet = evidence_packet(records, "2025-01-03")
    payload = evidence_text(packet)
    assert len(packet["records"]) == 4
    assert "future secret" not in payload and "not in prompt" not in payload
    assert all(r["eligible_date"] <= packet["as_of"] for r in packet["records"])
    assert "Evidence packet SHA256:" in payload
    assert "not instructions" in payload


def test_missing_channel_stops_model_packet():
    with pytest.raises(ValueError, match="required evidence absent"):
        evidence_packet(fixtures()[:-1], "2025-01-03")


def test_retrospective_opt_in_required_before_creating_output(tmp_path):
    root = tmp_path / "never-created"
    with pytest.raises(ValueError, match="explicit retrospective"):
        freeze(root, "7b", 11, tmp_path / "absent.json")
    assert not root.exists()


@pytest.mark.parametrize("arm", ["raw", "library"])
def test_actual_initial_prompt_allows_all_supplied_information(arm):
    from types import SimpleNamespace
    from benchmarks.agent_study import trading_runtime_v6 as runtime
    captured = []
    def backend(messages):
        captured.extend(dict(m) for m in messages)
        return {"choices": [{"message": {"content": '{"tool":"submit","arguments":{"weights":{"SPY":1}}}'},
                             "finish_reason": "stop"}], "usage": {}}
    packet = evidence_packet(fixtures(), "2025-01-03")
    out = runtime.decide(backend, SimpleNamespace(arm=arm, calls=[], menu_seed=0),
                         "Choose your weights." + evidence_text(packet),
                         common_instructions=COMMON, orientation_path=GUIDE)
    assert out["submitted"]
    assert "visible market files and the dated evidence packet" in captured[0]["content"]
    assert "Use only the visible market files for the task." not in captured[0]["content"]
    assert "price-only study" not in captured[0]["content"]
    assert captured[1]["content"].endswith(evidence_text(packet))
    if arm == "library":
        assert out["library_orientation"]["source"] == GUIDE.name
        assert "Use both the market" in captured[0]["content"]


def test_aggregate_rejects_complete_older_experiment(tmp_path):
    import json
    for family in ("7b", "14b"):
        for seed in (11, 23, 37):
            root = tmp_path / f"{family}-{seed}"
            root.mkdir()
            (root / "completed.json").write_text("{}")
            (root / "protocol.json").write_text(json.dumps({"version": "trading-interface-v5-2"}))
    with pytest.raises(ValueError, match="protocol/source changed"):
        aggregate(tmp_path)
