"""Regression tests based on the recorded hands-on failures, with real confined execution."""
import json
from pathlib import Path

import pandas as pd
import pytest

from benchmarks.agent_study import market_data as md
from benchmarks.agent_study import trading_runtime_v4 as old
from benchmarks.agent_study import trading_runtime_v5 as current
from benchmarks.agent_study import trading_study_v5 as study
from benchmarks.agent_study.trading_tools_v5 import Tools, text_page, wire, OUTPUT_LIMIT
from benchmarks.agent_study.linux_sandbox import confinement_available


def test_unicode_escaped_pages_roundtrip_without_losing_any_document_characters():
    text = ('Quoted "text"\\\n????\t' * 1000) + "END"
    recovered, offset, pages = "", 0, 0
    while offset is not None:
        page = text_page(text, offset, name="test")
        encoded = wire(page)
        assert len(encoded) <= OUTPUT_LIMIT
        assert json.loads(encoded) == page
        recovered += page["text"]
        offset = page["next_offset"]
        pages += 1
    assert recovered == text and pages > 1


def test_large_numeric_result_is_rejected_not_silently_partially_returned():
    payload = {"ok": True, "result": {"weights": {str(i): 0.1 for i in range(2000)}}}
    result = json.loads(wire(payload))
    assert result["ok"] is False and result["output_omitted"]
    assert "weights" not in result
    small = {"ok": True, "result": {"weights": {"SPY": .2, "IEF": .8}}}
    assert json.loads(wire(small)) == small


@pytest.fixture
def workspace(tmp_path):
    rows = [dict(date=day, ticker=ticker, open=100., high=101., low=99.,
                 close=100., volume=1000)
            for day in ("2024-01-02", "2024-01-03", "2024-01-04")
            for ticker in ("SPY", "IEF")]
    pd.DataFrame(rows).to_csv(tmp_path / "quotes.csv", index=False)
    pd.DataFrame([dict(date="2024-01-03", ticker="SPY", kind="dividend",
                       value=1., note="test")]).to_csv(tmp_path / "corporate_actions.csv",
                                                     index=False)
    (tmp_path / "README.md").write_text("visible files only")
    return tmp_path


def test_market_reader_recent_first_paged_equal_for_both_arms(workspace):
    raw, lib = Tools(workspace, "raw"), Tools(workspace, "library")
    assert raw.read_market() == lib.read_market()
    page = raw.read_market(tickers=["SPY"], limit=2)
    assert page["as_of"] == "2024-01-04" and page["adjusted"] is False
    assert [r["date"] for r in page["rows"]] == ["2024-01-04", "2024-01-03"]
    tail = raw.read_market(tickers=["SPY"], offset=page["next_offset"])
    assert [r["date"] for r in tail["rows"]] == ["2024-01-02"]
    assert tail["next_offset"] is None
    args = dict(file="corporate_actions.csv", start="2024-01-03", end="2024-01-03")
    assert len(lib.read_market(**args)["rows"]) == 1
    assert raw.python_bindings()["read_market"](**args) == raw.call("read_market", args)


@pytest.mark.parametrize("args", [
    {"end": "2024-01-05"}, {"file": "../hidden.csv"}, {"tickers": ["UNKNOWN"]},
    {"start": "2024-01-04", "end": "2024-01-03"}, {"limit": 0}, {"offset": -1}])
def test_market_reader_rejects_invalid_or_future_requests(workspace, args):
    assert Tools(workspace, "raw").call("read_market", args)["ok"] is False


def test_full_skill_and_reference_are_recoverable_and_paths_are_checked(workspace):
    import fin_skills
    tools = Tools(workspace, "library")
    names, offset = [], 0
    while offset is not None:
        page = tools.list_skills(offset=offset)
        assert json.loads(wire(page)) == page
        names.extend(c["name"] for c in page["skills"])
        offset = page["next_offset"]
    assert names == sorted(fin_skills.names())
    name = "portfolio-optimizers"
    chunks, offset = [], 0
    while offset is not None:
        page = tools.read_skill(name, offset=offset)
        assert json.loads(wire(page)) == page
        chunks.append(page["text"])
        offset = page["next_offset"]
    assert "".join(chunks) == fin_skills.load(name)
    assert len(chunks) > 1
    assert not tools.call("read_skill", {"name": "../hidden"})["ok"]
    assert not tools.call("read_skill", {"name": name, "reference": "../../hidden"})["ok"]
    for skill in names:
        refs = fin_skills.references(skill)
        if refs:
            reference, original = next(iter(refs.items()))
            text, offset = "", 0
            while offset is not None:
                page = tools.read_skill(skill, reference=reference, offset=offset)
                text += page["text"]
                offset = page["next_offset"]
            assert text == original
            break
    else:
        pytest.fail("expected at least one real reference document")
    assert not Tools(workspace, "raw").call("list_skills", {})["ok"]


def test_adapter_cards_do_not_ask_for_underlying_data_and_errors_give_signature(workspace):
    tools = Tools(workspace, "library")
    menu = tools.list_algorithms()
    assert json.loads(wire(menu)) == menu
    assert all("inputs" not in c for c in menu["algorithms"])
    for method in ("hrp", "inverse_volatility", "equal_weight", "momentum"):
        card = tools.describe_algorithm(method)
        assert json.loads(wire(card)) == card
        assert "inputs" not in card["card"]
        assert "reply['result']" in card["python_example"]
        assert "do not import" in card["note"]
    error = tools.call("run_algorithm", {"id": "hrp"})
    assert not error["ok"] and "algorithm_id" in error["expected"]


def test_model_history_contains_valid_json_and_records_the_exact_feedback(workspace):
    class Controller:
        arm, menu_seed, calls = "raw", 0, []
        def call(self, name, args):
            return text_page("??\\\n" * 5000, path="README.md")
    class Backend:
        calls = 0
        def __call__(self, history):
            self.calls += 1
            if self.calls == 1:
                content = '{"tool":"read_file","arguments":{"path":"README.md"}}'
            else:
                page = json.loads(history[-1]["content"])
                assert page["next_offset"] is not None
                assert len(history[-1]["content"]) <= OUTPUT_LIMIT
                content = '{"tool":"submit","arguments":{"weights":{"SPY":1}}}'
            return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}
    record = current.decide(Backend(), Controller(), "test")
    assert record["submitted"]
    assert json.loads(record["turns"][0]["model_feedback"]) == record["turns"][0]["result"]


@pytest.mark.skipif(not confinement_available(), reason="requires real Linux confinement")
def test_real_confined_read_describe_algorithm_submit_and_old_failure(tmp_path):
    md.write_dataset(tmp_path / "data")
    md.truncate(tmp_path / "data", tmp_path / "visible", "2025-01-02")
    controller = current.Controller(tmp_path, tmp_path / "visible", "library")
    market = controller.call("read_market", {"tickers": ["SPY"], "limit": 2})
    assert market["ok"] and market["as_of"] == "2025-01-02"
    for method in ("hrp", "inverse_volatility", "equal_weight"):
        card = controller.call("describe_algorithm", {"algorithm_id": method})
        result = controller.call("run_python", {"code": card["python_example"]})
        assert result["ok"] and result["submission"] is not None, result
        assert sum(result["submission"].values()) == pytest.approx(1, abs=1e-5)
        if method == "equal_weight":
            assert all(w == pytest.approx(1 / len(md.TICKERS), abs=1e-6)
                       for w in result["submission"].values())
    broken = controller.call("run_python", {"code":
        "reply=run_algorithm(algorithm_id='hrp')\nsubmit(reply.weights)"})
    assert not broken["ok"] and "AttributeError" in broken["output"]
    guard = controller.call("run_guard", {"name": "data_quality", "ticker": "SPY"})
    assert guard["ok"] and isinstance(guard["passed"], bool)

    bars = pd.read_csv(tmp_path / "visible/quotes.csv")
    bars.loc[bars.ticker == "SPY", "close"] = -1.
    bars.to_csv(tmp_path / "visible/quotes.csv", index=False)
    failed_guard = controller.call("run_guard", {"name": "data_quality", "ticker": "SPY"})
    assert failed_guard["ok"] and failed_guard["passed"] is False
    # Existing security acceptance tests must still execute against the v5 worker.
    qualification = current.qualify(tmp_path / "qualification", tmp_path / "data")
    assert qualification["passed"], qualification


def test_v5_runner_freezes_new_modules_and_preserves_old_versions(tmp_path):
    assert study.runtime is current
    sources = study.sources()
    for suffix in ("trading_tools_v4.py", "trading_tools_v5.py", "trading_worker_v5.py",
                   "trading_runtime_v5.py", "trading_study_v5.py"):
        assert f"benchmarks/agent_study/{suffix}" in sources
    assert "result.weights" in old.LIBRARY
    assert 'reply["result"]["weights"]' in current.LIBRARY
