"""Opt-in history-aware tools; frozen v5 and factorial experiments are unchanged."""
import json

import numpy as np
import pandas as pd

from benchmarks.agent_study.trading_onboarding_v6 import document_retrieval

from benchmarks.agent_study.trading_tools_v5 import (
    Tools as PreviousTools, OUTPUT_LIMIT, rows_page, validate_weights, wire,
    execution_status, positive_integer, method_defaults,
)


class Tools(PreviousTools):
    COMMON = PreviousTools.COMMON + ("read_snapshot",)
    LIBRARY = PreviousTools.LIBRARY + ("read_history",)

    def read_snapshot(self, tickers=None):
        # Uses the common raw reader, so the raw arm does not import library modules.
        quotes = pd.read_csv(self.workspace / "quotes.csv")
        date = str(quotes.date.max())
        page = super().read_market(tickers=tickers, start=date, end=date, limit=100)
        page.update(kind="snapshot", price_sessions=1, return_samples=0,
                    suitable_for_return_estimation=False,
                    note="RAW levels, one session. Not return history. next_offset indicates omitted rows.")
        self._fit_page(page)
        return page

    @staticmethod
    def _fit_page(page):
        while len(json.dumps(page, ensure_ascii=False)) > OUTPUT_LIMIT and page["rows"]:
            page["rows"].pop()
            page["next_offset"] = page["offset"] + len(page["rows"])
        if not page["rows"] and page["total_rows"] > page["offset"]:
            raise ValueError("one row exceeds the response budget")

    def read_market(self, file="quotes.csv", tickers=None, start=None, end=None,
                    offset=0, limit=12):
        page = super().read_market(file, tickers, start, end, offset, limit)
        page.update(kind="raw_page", page_sessions=len({r['date'] for r in page['rows']}),
                    suitable_for_return_estimation=False,
                    note="Inspection page, not an estimation window. Use complete history.")
        self._fit_page(page)
        page["page_sessions"] = len({r['date'] for r in page['rows']})
        return page

    def load_history(self, tickers=None, lookback=252):
        """Python-only full matrices; only the library arm has this binding."""
        if self.arm != "library":
            raise ValueError("load_history unavailable in raw condition")
        from fin_skills.data import prepare_history
        quotes = pd.read_csv(self.workspace / "quotes.csv")
        actions = pd.read_csv(self.workspace / "corporate_actions.csv")
        # The study dataset explicitly supplies all visible actions. No vendor inference.
        return prepare_history(quotes, actions, as_of=str(quotes.date.max()),
                               tickers=tickers, lookback=lookback,
                               input_adjustment="raw", actions_complete=True)

    def read_history(self, tickers=None, lookback=252, offset=0, limit=12):
        history = self.load_history(tickers, lookback)
        frame = history["returns"].rename_axis(index="date", columns="ticker")
        rows = frame.stack().rename("return").reset_index().to_dict("records")
        return rows_page(rows, offset, limit, "rows", metadata=history["metadata"],
                         page_only=True,
                         python_usage="h=load_history(lookback=252); returns=h['returns']")

    def run_algorithm(self, algorithm_id, tickers=None, lookback=252, parameters=None):
        from fin_skills import algorithms
        from benchmarks.agent_study.trading_worker import _round
        if parameters is not None and not isinstance(parameters, dict):
            raise ValueError("parameters must be a dictionary or null")
        cards = {c["id"]: c for c in algorithms.catalog()}
        if algorithm_id not in cards:
            raise ValueError("unknown algorithm_id; call list_algorithms then describe_algorithm")
        card = cards[algorithm_id]
        effective = method_defaults(algorithm_id, card["task"])
        effective.update(parameters or {})
        for name in ("lookback", "fast", "slow", "signal_span", "baseline"):
            if name in effective:
                positive_integer(effective[name], f"parameters.{name}")
        history = self.load_history(tickers, lookback)
        window, returns = history["prices"], history["returns"]
        for name in ("lookback", "fast", "slow", "signal_span", "baseline"):
            if effective.get(name, 0) > len(returns):
                raise ValueError(f"parameters.{name} exceeds {len(returns)} available returns")
        inputs = tuple(card["inputs"])
        if inputs == ("asset_returns",):
            output = np.asarray(algorithms.run(algorithm_id, {"asset_returns": returns},
                                               **effective), dtype=float)
            if output.shape != (len(returns.columns),) or not np.isfinite(output).all():
                raise ValueError("algorithm produced invalid weights or lost ticker alignment")
            value = {"weights": _round(pd.Series(output, index=returns.columns))}
        elif inputs in (("prices",), ("returns",), ("series",)):
            matrix = returns if inputs[0] == "returns" else window
            value = {"per_ticker": {t: _round(algorithms.run(
                algorithm_id, {inputs[0]: matrix[t]}, **effective)) for t in matrix.columns}}
        else:
            raise ValueError(f"unsupported prepared inputs {inputs}; use the direct Python API")
        return {"ok": True, "algorithm_id": algorithm_id, "result": value,
                "tickers": list(returns.columns), "lookback": lookback, "adjusted": True,
                "effective_parameters": effective, "history": history["metadata"],
                "timing": {"output_lag_bars": 1 if card["task"] == "signal" else 0,
                           "execution": "next session close"}}

    def call(self, name, arguments):
        result = super().call(name, arguments)
        if name == "read_skill":
            self.calls[-1].update(document_retrieval(result))
        if not result.get("ok"):
            if name == "read_market":
                result["hint"] = ("read_market pages RAW rows; it has no lookback argument. "
                    "Use read_snapshot for one session. In library Python, use "
                    "h=load_history(lookback=60); returns=h['returns'] for calculations. "
                    "Raw Python must load and adjust complete visible CSVs.")
            elif name == "load_history":
                result["hint"] = ("load_history is a Python-only function returning full DataFrames; "
                                  "use run_python or read_history for JSON pages.")
        return result

    def python_bindings(self):
        bindings = super().python_bindings()
        if self.arm == "library":
            def load(*args, **kwargs):
                try:
                    result = self.load_history(*args, **kwargs)
                except (TypeError, ValueError, KeyError, OSError):
                    self.calls.append(dict(tool="load_history", ok=False))
                    raise
                self.calls.append(dict(tool="load_history", ok=True,
                                       metadata=result["metadata"]))
                return result
            bindings["load_history"] = load
        return bindings
