import pytest

pytest.importorskip("bs4")
from benchmarks.agent_study.multisource_model_study import (
    BATCHES, COMMON, GUIDE, aggregate, evidence_packet, evidence_text, freeze, model_spec,
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


@pytest.mark.parametrize("family", ["32b", "mistral12b"])
def test_expansion_requires_explicit_batch_and_pinned_revision(family):
    with pytest.raises(ValueError, match="outside the declared batch"):
        model_spec(family, 11, "original")
    model = model_spec(family, 11, "expansion")
    assert model[0] == family and len(model[2]) == 40
    assert all(c in "0123456789abcdef" for c in model[2])


def test_expansion_cannot_report_partial_or_original_model_mean(tmp_path):
    for family in BATCHES["original"]:
        for seed in (11, 23, 37):
            root = tmp_path / f"{family}-{seed}"
            root.mkdir()
            (root / "completed.json").write_text("{}")
    result = aggregate(tmp_path, "expansion")
    assert result["status"] == "pending"
    assert set(result["missing"]) == {f"{f}-{s}" for f in BATCHES["expansion"] for s in (11, 23, 37)}


def test_complete_expansion_keeps_batch_identity_and_all_seed_means(tmp_path, monkeypatch):
    from benchmarks.agent_study import multisource_model_study as study
    old = study.old
    for family in BATCHES["expansion"]:
        for seed in (11, 23, 37):
            root = tmp_path / f"{family}-{seed}"
            inputs = dict(arms=["raw", "library"], picks=[1], dates=["fixture"])
            old.write(root / "inputs.json", inputs)
            keys = ("source_sha256", "data_sha256", "raw_download_sha256", "universe",
                    "max_turns", "max_tokens", "temperature", "top_p", "execution", "cost_bps",
                    "cash_interest", "objective", "primary_report", "window", "decisions_per_arm", "treatment")
            protocol = dict.fromkeys(keys, "synthetic common condition")
            protocol.update(version="multisource-model-v1", batch="expansion", seed=seed,
                            model=model_spec(family, seed, "expansion"),
                            input_hashes={n: "fixture" for n in ("evidence.json", "evidence-packets.json")})
            old.write(root / "protocol.json", protocol)
            files = {}
            for arm in inputs["arms"]:
                path = root / "decisions" / arm / "00.json"
                old.write(path, dict(synthetic_fixture=True))
                files[path.relative_to(root).as_posix()] = old.sha(path)
            old.write(root / "inference-receipt.json", dict(
                protocol_sha256=old.sha(root / "protocol.json"), decisions=files))
            row = dict(family=family, seed=seed,
                       paths={a: dict(return_rate_pct=seed + i) for i, a in enumerate(inputs["arms"])},
                       inference_receipt_sha256=old.sha(root / "inference-receipt.json"))
            old.write(root / "scores.json", row)
            old.write(root / "completed.json", dict(scores_sha256=old.sha(root / "scores.json")))
    # Isolate aggregation from machine-specific source/market files, but exercise its
    # real protocol-to-receipt-to-decisions-to-score checks on the synthetic files.
    monkeypatch.setattr(study, "verify", lambda root: old.load(root / "protocol.json"))
    result = aggregate(tmp_path, "expansion")
    assert result["status"] == "complete" and result["batch"] == "expansion"
    assert len(result["seed_results"]) == 6
    assert set(result["return_rate_pct"]) == set(BATCHES["expansion"])
    for metrics in result["return_rate_pct"].values():
        assert metrics["raw"] == pytest.approx((11 + 23 + 37) / 3)
        assert metrics["difference_pp"] == pytest.approx(1.)


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
