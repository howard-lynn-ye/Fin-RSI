import json

import numpy as np
import pandas as pd
import pytest

from benchmarks.agent_study.linux_sandbox import confinement_available
from benchmarks.agent_study.trading_tools_v6 import Tools, wire, OUTPUT_LIMIT
from benchmarks.agent_study.trading_worker_v6 import run_python
from benchmarks.agent_study.trading_runtime_v6 import Controller


@pytest.fixture
def workspace(tmp_path):
    dates = pd.bdate_range("2024-01-01", periods=270).strftime("%Y-%m-%d")
    rows = []
    for j, ticker in enumerate(("SPY", "IEF", "GLD")):
        prices = 100 * np.cumprod(1 + .001 + .01 * np.sin(np.arange(270)/(3+j)))
        rows.extend(dict(date=d, ticker=ticker, close=p) for d, p in zip(dates, prices))
    pd.DataFrame(rows).to_csv(tmp_path / "quotes.csv", index=False)
    pd.DataFrame(columns=["date", "ticker", "kind", "value"]).to_csv(
        tmp_path / "corporate_actions.csv", index=False)
    (tmp_path / "README.md").write_text("Synthetic raw prices and complete empty action list.")
    return tmp_path


def test_full_history_stays_in_python_and_has_explicit_provenance(workspace):
    tools = Tools(workspace, "library")
    h = tools.python_bindings()["load_history"](lookback=60, tickers=["GLD", "SPY"])
    assert h["returns"].shape == (60, 2) and h["prices"].shape == (61, 2)
    assert list(h["returns"].columns) == ["GLD", "SPY"]
    assert h["metadata"]["output_adjustment"] == "total_return"
    assert tools.calls[-1]["tool"] == "load_history"
    assert "load_history" not in Tools(workspace, "raw").python_bindings()
    r = tools.call("load_history", {})
    assert not r["ok"] and "Python-only" in r["hint"]


@pytest.mark.parametrize("method,dependency", [
    ("hrp", None), ("pypfopt_hrp", "pypfopt"), ("riskfolio_hrp", "riskfolio")])
def test_advertised_hrp_examples_execute_and_preserve_explicit_linkage_guard(
        workspace, method, dependency):
    if dependency:
        pytest.importorskip(dependency)
    from benchmarks.agent_study.trading_capabilities import Tools as CompleteTools
    tools = CompleteTools(workspace, "library")
    card = tools.call("describe_algorithm", {"algorithm_id": method})
    assert card["ok"], card
    if dependency:
        missing = tools.call("run_algorithm", {"algorithm_id": method})
        assert not missing["ok"] and "linkage" in missing["error"].lower()
        assert "linkage" in card["required_parameters"]
    else:
        assert card["required_parameters"] == {}
    result = run_python(card["python_example"], tools)
    assert result["ok"] and result["submission"] is not None, result
    from benchmarks.agent_study.market_data import TICKERS
    assert set(result["submission"]) == set(TICKERS)
    assert {t for t, w in result["submission"].items() if w} == {"SPY", "IEF", "GLD"}
    assert sum(result["submission"].values()) == pytest.approx(1, abs=1e-5)
    executions = [c for c in tools.calls if c["tool"] == "run_algorithm" and c["ok"]]
    assert executions[-1]["effective_parameters"]["linkage"] == "single"


def test_snapshot_and_history_pages_never_claim_to_be_full_matrices(workspace):
    tools = Tools(workspace, "library")
    snapshot = tools.call("read_snapshot", {})
    assert snapshot["ok"] and snapshot["return_samples"] == 0
    assert len({r["date"] for r in snapshot["rows"]}) == 1
    for method in ("read_history", "read_market"):
        seen, offset = [], 0
        while True:
            page = tools.call(method, {"offset": offset, "limit": 100})
            assert page["ok"] and len(wire(page)) <= OUTPUT_LIMIT
            assert json.loads(wire(page)) == page
            seen.extend(page["rows"])
            if page["next_offset"] is None: break
            assert page["next_offset"] > offset
            offset = page["next_offset"]
        assert len(seen) == (252 * 3 if method == "read_history" else 270 * 3)
    assert not Tools(workspace, "raw").call("read_history", {})["ok"]


def test_methods_use_identical_validated_data_and_keep_ticker_alignment(workspace):
    tools = Tools(workspace, "library")
    h = tools.load_history(["GLD", "SPY"], 60)
    expected = 1 / h["returns"].std(ddof=1)
    expected /= expected.sum()
    r = tools.call("run_algorithm", dict(algorithm_id="inverse_volatility",
                                        tickers=["GLD", "SPY"], lookback=60))
    assert r["ok"], r
    assert r["history"]["return_samples"] == 60
    for t in expected.index:
        assert r["result"]["weights"][t] == pytest.approx(expected[t], abs=1e-6)
    bad = tools.call("run_algorithm", dict(algorithm_id="hrp", lookback=500))
    assert not bad["ok"] and "found 270" in bad["error"]


def test_reproduced_misuse_returns_a_specific_correction(workspace):
    tools = Tools(workspace, "library")
    error = tools.call("read_market", {"lookback": 60})
    assert not error["ok"] and "load_history(lookback=60)" in error["hint"]
    error = run_python("read_market(lookback=60)", tools)
    assert not error["ok"] and "load_history(lookback=...)" in error["error"]
    error = run_python("reply=run_algorithm('hrp')\nsubmit(reply.weights)", tools)
    assert not error["ok"] and "reply['result']['weights']" in error["error"]


@pytest.mark.skipif(not confinement_available(), reason="requires Linux confinement")
def test_real_sandbox_history_submit_and_access_boundaries(workspace, tmp_path):
    lib = Controller(tmp_path / "runtime", workspace, "library")
    code = "h=load_history(lookback=60)\nprint(h['metadata']['return_samples'])\nsubmit({'SPY':1})"
    r = lib.call("run_python", {"code": code})
    assert r["ok"] and r["submission"]["SPY"] == 1, r
    assert "60" in r["output"]
    for method in ("hrp", "inverse_volatility", "equal_weight"):
        card = lib.call("describe_algorithm", {"algorithm_id": method})
        r = lib.call("run_python", {"code": card["python_example"]})
        assert r["ok"] and r["submission"] is not None, r
    raw = Controller(tmp_path / "raw-runtime", workspace, "raw")
    assert raw.call("read_snapshot", {})["ok"]
    assert not raw.call("run_python", {"code": "load_history()"})["ok"]
    assert not raw.call("run_python", {"code": "import fin_skills"})["ok"]
    assert not lib.call("run_python", {"code": "open('quotes.csv','w').write('x')"})["ok"]
    assert not lib.call("run_python", {"code": "import socket; socket.socket()"})["ok"]
