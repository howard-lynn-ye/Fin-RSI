"""Regressions from independent review and the failed 7B development smoke."""
import json

import numpy as np
import pandas as pd
import pytest

from benchmarks.agent_study.trading_tools_v5 import (
    Tools, wire, execution_status, turn_counts, OUTPUT_LIMIT)
from benchmarks.agent_study.trading_worker_v5 import run_python
from benchmarks.agent_study import trading_runtime_v5 as runtime
from benchmarks.agent_study.trading_study import READY_ALGORITHMS


@pytest.fixture
def workspace(tmp_path):
    dates = pd.bdate_range("2024-01-01", periods=96)
    quotes = []
    for j, ticker in enumerate(("SPY", "IEF")):
        price = 100 * np.cumprod(1 + .002 + .02 * np.sin(np.arange(96) / (4 + j)))
        for d, p in zip(dates, price):
            quotes.append(dict(date=d.strftime("%Y-%m-%d"), ticker=ticker,
                open=p, high=p+1, low=p-1, close=p, volume=1000))
    pd.DataFrame(quotes).to_csv(tmp_path / "quotes.csv", index=False)
    pd.DataFrame(columns=["date", "ticker", "kind", "value", "note"]).to_csv(
        tmp_path / "corporate_actions.csv", index=False)
    return tmp_path


def test_execution_success_with_delivery_failure_is_counted():
    original = dict(ok=True, result=dict(weights={str(i): .1 for i in range(2000)}))
    status = execution_status(original)
    assert status["execution_ok"] and not status["delivery_ok"] and not status["turn_ok"]
    row = dict(result=original, **status)
    counts = turn_counts([row])
    assert counts == dict(execution_failed_turns=0, delivery_failed_turns=1, failed_turns=1)
    failed = dict(result=dict(ok=False), execution_ok=False, delivery_ok=True)
    assert turn_counts([row, failed])["failed_turns"] == 2


def test_long_python_output_keeps_error_and_tail_with_valid_envelope(workspace):
    result = run_python("print('x' * 8000)\nraise ValueError('END_OF_TRACE')",
                        Tools(workspace, "raw"))
    assert len(result["output"]) > OUTPUT_LIMIT
    assert result["error"] == "ValueError: END_OF_TRACE"
    model = json.loads(wire(result))
    assert len(wire(result)) <= OUTPUT_LIMIT
    assert "END_OF_TRACE" in model["output"] and model["output_truncated"]
    assert model["error"] == "ValueError: END_OF_TRACE"
    assert not execution_status(result)["delivery_ok"]


def test_verbose_valid_submission_remains_valid_but_delivery_is_flagged(workspace):
    result = run_python("print('x' * 8000)\nsubmit({'SPY':1})", Tools(workspace, "raw"))
    status = execution_status(result)
    assert result["ok"] and result["submission"]["SPY"] == 1
    assert status["execution_ok"] and not status["delivery_ok"]
    assert json.loads(status["model_feedback"])["submission"] == result["submission"]


@pytest.mark.parametrize("lookback", [0, -1, 1.5, True, None, "60"])
def test_history_window_rejects_silent_coercion(workspace, lookback):
    result = Tools(workspace, "library").call("run_algorithm",
        dict(algorithm_id="momentum", lookback=lookback))
    assert not result["ok"] and "positive integer" in result["error"]


@pytest.mark.parametrize("window", [0, -1, 1.5, True, "10"])
def test_method_window_rejects_silent_coercion(workspace, window):
    result = Tools(workspace, "library").call("run_algorithm",
        dict(algorithm_id="momentum", parameters=dict(lookback=window)))
    assert not result["ok"] and "parameters.lookback" in result["error"]


def test_supplied_history_does_not_override_algorithm_window(workspace):
    tools = Tools(workspace, "library")
    default = tools.call("run_algorithm", dict(algorithm_id="momentum", lookback=60))
    shorter = tools.call("run_algorithm", dict(algorithm_id="momentum", lookback=60,
                                              parameters=dict(lookback=10)))
    assert default["ok"] and shorter["ok"], (default, shorter)
    assert default["history"]["requested_returns"] == shorter["history"]["requested_returns"] == 60
    assert default["history"]["price_sessions"] == 61
    assert default["effective_parameters"]["lookback"] == 20
    assert shorter["effective_parameters"]["lookback"] == 10
    assert default["result"]["per_ticker"] != shorter["result"]["per_ticker"]
    assert default["timing"]["output_lag_bars"] == 1
    assert tools.calls[-1]["effective_parameters"]["lookback"] == 10


def test_method_cards_fit_and_report_real_defaults(workspace):
    tools = Tools(workspace, "library")
    for method in READY_ALGORITHMS:
        card = tools.describe_algorithm(method)
        assert json.loads(wire(card)) == card, method
    assert tools.describe_algorithm("momentum")["algorithm_parameter_defaults"]["lookback"] == 20
    assert tools.describe_algorithm("cross_sectional_momentum")["algorithm_parameter_defaults"]["lookback"] == 60
    assert tools.describe_algorithm("rolling_market_state")["timing"]["output_lag_bars"] == 0
    assert tools.describe_algorithm("hrp")["algorithm_parameter_defaults"]["linkage"] == "single"


def test_record_retains_raw_result_and_counts_model_feedback_failure(workspace):
    class Controller:
        arm, menu_seed, calls = "raw", 0, []
        def call(self, name, args):
            return dict(ok=True, result={str(i): "x" * 30 for i in range(200)})
    class Backend:
        calls = 0
        def __call__(self, history):
            self.calls += 1
            if self.calls == 1:
                text = '{"tool":"read_market","arguments":{}}'
            else:
                assert json.loads(history[-1]["content"])["delivery_ok"] is False
                text = '{"tool":"submit","arguments":{"weights":{"SPY":1}}}'
            return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}]}
    record = runtime.decide(Backend(), Controller(), "test")
    assert record["submitted"]
    first = record["turns"][0]
    assert first["result"]["ok"] and not first["delivery_ok"]
    assert turn_counts(record["turns"])["failed_turns"] == 1


def test_market_documentation_names_actual_rows_key(workspace):
    page = Tools(workspace, "raw").read_market(tickers=["SPY"], limit=3)
    assert page["ok"] and len(page["rows"]) == 3 and "data" not in page
    assert 'page["rows"]' in runtime.COMMON
    assert 'pivot(index="date",columns="ticker",values="close")' in runtime.COMMON


def test_normal_document_pagination_is_not_failed_delivery():
    from benchmarks.agent_study.trading_tools_v5 import text_page
    page = text_page("reference text\n" * 2000, limit=500)
    assert page["truncated"] and page["next_offset"] is not None
    status = execution_status(page)
    assert status["execution_ok"] and status["delivery_ok"] and status["turn_ok"]
    assert turn_counts([dict(result=page, **status)])["failed_turns"] == 0


def test_controller_reminds_both_arms_of_call_budget_without_submitting_for_them():
    for arm in ("raw", "library"):
        class Controller:
            menu_seed, calls = 0, []
            def __init__(self):
                self.arm = arm
            def call(self, name, args):
                return dict(ok=True)
        class Backend:
            calls = 0
            def __call__(self, history):
                self.calls += 1
                system = history[0]["content"]
                assert f"Current call {self.calls} of 8" in system
                if self.calls == 8:
                    assert "last call" in system
                return {"choices": [{"message": {"content": "no executable call"},
                                      "finish_reason": "stop"}]}
        record = runtime.decide(Backend(), Controller(), "test")
        assert not record["submitted"] and record["target"] is None
        assert [t["calls_remaining"] for t in record["turns"]] == list(range(8, 0, -1))
        assert "Multiple code blocks" in record["turns"][0]["result"]["error"]

@pytest.mark.parametrize("text", [
    chr(96)*3 + 'python\nsubmit({"SPY":1})\n' + chr(96)*3,
    '{"tool":"submit","arguments":{"weights":{"SPY":1}}}'])
def test_generation_cutoff_never_executes_even_a_parseable_prefix(text):
    class Controller:
        arm, menu_seed, calls = "raw", 0, []
        def call(self, name, args):
            raise AssertionError("truncated generation must not execute")
    class Backend:
        calls = 0
        def __call__(self, history):
            self.calls += 1
            if self.calls == 1:
                value, reason = text, "length"
            else:
                error = json.loads(history[-1]["content"])["error"]
                assert "1024" in error and "NOT executed" in error
                value = '{"tool":"submit","arguments":{"weights":{"IEF":1}}}'
                reason = "stop"
            return {"choices": [{"message": {"content": value}, "finish_reason": reason}]}
    record = runtime.decide(Backend(), Controller(), "test")
    assert len(record["turns"]) == 2
    assert record["turns"][0]["parse"] == "generation-truncated"
    assert record["target"]["IEF"] == 1 and record["target"]["SPY"] == 0
    assert turn_counts(record["turns"])["execution_failed_turns"] == 1
