"""Explicit, paginated study interfaces. Frozen v4 tools remain unchanged."""
import inspect
import json

from benchmarks.agent_study.market_data import VISIBLE
from benchmarks.agent_study.trading_tools_v4 import (
    OUTPUT_LIMIT, Tools as PreviousTools, validate_weights,
)



def wire(result):
    """Budget the final JSON envelope; retain raw execution output in the trace."""
    encode = lambda obj: json.dumps(obj, ensure_ascii=False, allow_nan=False)
    payload = encode(result)
    if len(payload) <= OUTPUT_LIMIT:
        return payload
    if isinstance(result.get("output"), str):
        original = result["output"]
        clipped = dict(result, output_truncated=True, output_chars=len(original),
                       delivery_ok=False, execution_ok=bool(result.get("ok")))
        def candidate(n):
            if n:
                head = (n + 1) // 2
                tail = n // 2
                value = original[:head] + "\n...[output truncated]...\n"
                if tail:
                    value += original[-tail:]
            else:
                value = ""
            return encode(dict(clipped, output=value))
        lo, hi = 0, min(len(original), OUTPUT_LIMIT)
        if len(candidate(0)) <= OUTPUT_LIMIT:
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if len(candidate(mid)) <= OUTPUT_LIMIT:
                    lo = mid
                else:
                    hi = mid - 1
            return candidate(lo)
    return encode(dict(ok=False, execution_ok=bool(result.get("ok")), delivery_ok=False,
        error="Output exceeds the response budget. Use pagination or print fewer values.",
        output_omitted=True))


def execution_status(result):
    feedback = wire(result)
    delivered = json.loads(feedback)
    execution_ok = bool(result.get("ok"))
    delivery_ok = not (delivered.get("output_omitted") or
                       delivered.get("output_truncated"))
    return dict(model_feedback=feedback, execution_ok=execution_ok,
                delivery_ok=delivery_ok, turn_ok=execution_ok and delivery_ok)


def turn_counts(turns):
    execution = [bool(t.get("execution_ok", t["result"].get("ok"))) for t in turns]
    delivery = [bool(t.get("delivery_ok", True)) for t in turns]
    return dict(execution_failed_turns=sum(not x for x in execution),
                delivery_failed_turns=sum(not x for x in delivery),
                failed_turns=sum(not (x and y) for x, y in zip(execution, delivery)))


def positive_integer(value, name):
    from numbers import Integral
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer; no coercion or fallback")
    return int(value)


def method_defaults(algorithm_id, task):
    from fin_skills.algorithms.contracts import defaults
    from fin_skills.algorithms.technical import SIGNAL_DEFAULTS
    from fin_skills.algorithms.strategy import REGIME_DEFAULTS
    if algorithm_id in SIGNAL_DEFAULTS:
        return dict(SIGNAL_DEFAULTS[algorithm_id])
    if algorithm_id == "cross_sectional_momentum":
        return dict(lookback=60, top_k=3)
    if algorithm_id == "rolling_market_state":
        return dict(REGIME_DEFAULTS)
    params = defaults(algorithm_id, task)
    if algorithm_id == "hrp":
        params["linkage"] = "single"
    return params


def text_page(text, offset=0, limit=4000, **metadata):
    offset, limit = int(offset), min(int(limit), OUTPUT_LIMIT)
    if offset < 0 or limit < 1:
        raise ValueError("offset must be >= 0 and limit >= 1")
    def page(n):
        end = min(offset + n, len(text))
        return dict(ok=True, **metadata, offset=offset, total_chars=len(text),
                    text=text[offset:end], next_offset=end if end < len(text) else None,
                    truncated=end < len(text))
    lo, hi = 0, min(limit, max(0, len(text) - offset))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(json.dumps(page(mid), ensure_ascii=False)) <= OUTPUT_LIMIT:
            lo = mid
        else:
            hi = mid - 1
    if not lo and offset < len(text):
        raise ValueError("response metadata exceeds the output budget")
    return page(lo)


def rows_page(rows, offset, limit, key, **metadata):
    offset, limit = int(offset), int(limit)
    if offset < 0 or not 1 <= limit <= 100:
        raise ValueError("offset must be >= 0 and limit between 1 and 100")
    selected = rows[offset:offset + limit]
    def page():
        end = offset + len(selected)
        return dict(ok=True, **metadata, offset=offset, total_rows=len(rows),
                    **{key: selected}, next_offset=end if end < len(rows) else None)
    while selected and len(json.dumps(page(), ensure_ascii=False)) > OUTPUT_LIMIT:
        selected.pop()
    if not selected and offset < len(rows):
        raise ValueError("one row exceeds the output budget")
    return page()


class Tools(PreviousTools):
    COMMON = ("read_file", "read_market")
    LIBRARY = ("list_algorithms", "describe_algorithm", "list_skills", "read_skill",
               "run_algorithm", "run_guard")

    def read_file(self, path="", offset=0, limit=4000):
        if path not in VISIBLE:
            raise ValueError(f"files: {', '.join(VISIBLE)}")
        return text_page((self.workspace / path).read_text(encoding="utf-8"),
                         offset, limit, path=path,
                         note="File-order text; use read_market for recent dated rows.")

    def read_market(self, file="quotes.csv", tickers=None, start=None, end=None,
                    offset=0, limit=12):
        """Recent-first raw rows from the already point-in-time-truncated workspace."""
        import pandas as pd
        if file not in ("quotes.csv", "corporate_actions.csv"):
            raise ValueError("file must be quotes.csv or corporate_actions.csv")
        frame = pd.read_csv(self.workspace / file).fillna("")
        quotes = frame if file == "quotes.csv" else pd.read_csv(self.workspace / "quotes.csv")
        as_of = str(quotes.date.max())
        if tickers is not None:
            if not isinstance(tickers, list) or not tickers:
                raise ValueError("tickers must be a nonempty list, or null for all")
            unknown = set(tickers) - set(quotes.ticker)
            if unknown:
                raise ValueError(f"unknown tickers: {sorted(unknown)}")
            frame = frame[frame.ticker.isin(tickers)]
        bounds = {}
        for name, value in (("start", start), ("end", end)):
            if value is not None:
                parsed = pd.Timestamp(value).strftime("%Y-%m-%d")
                if parsed != value:
                    raise ValueError(f"{name} must be YYYY-MM-DD")
                if value > as_of:
                    raise ValueError(f"{name} exceeds visible as_of={as_of}")
                bounds[name] = value
        if start is not None and end is not None and start > end:
            raise ValueError("start must be <= end")
        if "start" in bounds:
            frame = frame[frame.date >= start]
        if "end" in bounds:
            frame = frame[frame.date <= end]
        order = ["date", "ticker"] + (["kind"] if "kind" in frame else [])
        frame = frame.sort_values(order, ascending=[False] + [True] * (len(order) - 1))
        return rows_page(frame.to_dict("records"), offset, limit, "rows",
            file=file, as_of=as_of, order="date descending, then ticker",
            adjusted=False, note="Raw observations; splits and dividends require adjustment.")

    def list_skills(self, query="", offset=0, limit=8):
        import fin_skills
        cards = [dict(name=c["name"], description=c.get("description", ""))
                 for c in fin_skills.catalog()
                 if str(query).lower() in (c["name"] + " " + c.get("description", "")).lower()]
        return rows_page(cards, offset, limit, "skills", query=query)

    def read_skill(self, name="", offset=0, limit=4000, reference=None):
        import fin_skills
        if name not in fin_skills.names():
            raise ValueError("Unknown skill; use list_skills(query=...)")
        if reference is None:
            return text_page(fin_skills.load(name), offset, limit, name=name,
                             references=list(fin_skills.references(name)))
        refs = fin_skills.references(name)
        if reference not in refs:
            raise ValueError(f"Unknown reference; available: {list(refs)}")
        return text_page(refs[reference], offset, limit, name=name, reference=reference)

    def list_algorithms(self):
        result = super().list_algorithms()
        for card in result["algorithms"]:
            card.pop("inputs", None)
        result["usage"] = ("Choose a method, then describe_algorithm(algorithm_id). "
                           "run_algorithm prepares the method's data internally.")
        return result

    def describe_algorithm(self, algorithm_id=""):
        result = json.loads(json.dumps(super().describe_algorithm(algorithm_id)))
        if not result.get("ok"):
            return result
        card = result["card"]
        prepared = card.pop("inputs")
        portfolio = prepared == ["asset_returns"]
        card["data_prepared_by_adapter"] = prepared
        if algorithm_id == "hrp":
            card["caveat"] = ("Adapter defaults linkage to single. Weights fit past data; "
                              "apply only to subsequent returns.")
        result["call_signature"] = str(inspect.signature(self.run_algorithm))
        result["arguments"] = {
            "algorithm_id": "method ID (not id)",
            "tickers": "list of ticker strings, or null for all visible assets",
            "lookback": "history supplied to method (positive integer, default 252); does NOT set parameters.lookback",
            "parameters": "method parameter mapping, default {}"}
        result["algorithm_parameter_defaults"] = method_defaults(algorithm_id, card["task"])
        result["timing"] = {"output_lag_bars": 1 if card["task"] == "signal" else 0,
                            "execution": "next session close under study protocol"}
        result["python_example"] = (
            f"reply = run_algorithm(algorithm_id={algorithm_id!r}, lookback=252)\n"
            "assert reply['ok'], reply\n" +
            ("weights = reply['result']['weights']\nsubmit(weights)" if portfolio else
             "print(reply['result']['per_ticker'])"))
        result["note"] = ("Functions are already bound in study Python: do not import them. "
                          "All responses are dicts. Do not pass asset_returns to this adapter. "
                          "Direct fin_skills.algorithms.run(algorithm_id, data, **parameters) "
                          "is a separate API.")
        return result


    def run_algorithm(self, algorithm_id, tickers=None, lookback=252, parameters=None):
        import pandas as pd
        import fin_skills.algorithms as algorithms
        lookback = positive_integer(lookback, "lookback (history length)")
        if parameters is not None and not isinstance(parameters, dict):
            raise ValueError("parameters must be a dictionary or null")
        card = next((c for c in algorithms.catalog() if c["id"] == algorithm_id), None)
        if card is None:
            raise ValueError("Unknown algorithm_id; use list_algorithms")
        given = dict(parameters or {})
        for name in ("lookback", "fast", "slow", "signal_span", "baseline"):
            if name in given:
                positive_integer(given[name], f"parameters.{name}")
        effective = method_defaults(algorithm_id, card["task"])
        effective.update(given)
        result = super().run_algorithm(algorithm_id, tickers, lookback, effective)
        if result.get("ok"):
            dates = pd.read_csv(self.workspace / "quotes.csv", usecols=["date"]).date
            sessions = sorted(dates.unique())[-(lookback + 1):]
            result["history"] = dict(requested_returns=lookback, price_sessions=len(sessions),
                                     start=sessions[0], as_of=sessions[-1])
            result["effective_parameters"] = effective
            result["timing"] = dict(output_lag_bars=1 if card["task"] == "signal" else 0,
                                    execution="next session close")
            if "per_ticker" in result["result"]:
                result["result"]["note"] = (
                    "Signal outputs are lagged one bar." if card["task"] == "signal" else
                    "Descriptor/statistic/forecast computed from supplied history; not "
                    "a lagged trading position.")
        return result

    def call(self, name, arguments):
        allowed = self.COMMON + (self.LIBRARY if self.arm == "library" else ())
        args = {} if arguments is None else arguments
        if name not in allowed:
            result = dict(ok=False, error=f"tool {name!r} unavailable in {self.arm}")
        elif not isinstance(args, dict):
            result = dict(ok=False, error="arguments must be an object")
        else:
            try:
                result = getattr(self, name)(**args)
            except (TypeError, ValueError, KeyError, OSError) as exc:
                result = dict(ok=False, error=f"{type(exc).__name__}: {str(exc)[:500]}",
                              expected=f"{name}{inspect.signature(getattr(self, name))}")
        receipt = dict(tool=name, ok=bool(result.get("ok")))
        if name == "run_algorithm" and result.get("ok"):
            receipt.update(algorithm_id=result["algorithm_id"],
                           effective_parameters=result["effective_parameters"],
                           weights=result["result"].get("weights"))
        if name == "run_guard" and result.get("ok"):
            receipt["passed"] = result["passed"]
        self.calls.append(receipt)
        return result

    def python_bindings(self):
        names = self.COMMON + (self.LIBRARY if self.arm == "library" else ())
        def binding(name):
            signature = inspect.signature(getattr(self, name))
            def call(*args, **kwargs):
                try:
                    arguments = dict(signature.bind(*args, **kwargs).arguments)
                except TypeError as exc:
                    raise TypeError(f"{name}{signature}: {exc}") from exc
                return self.call(name, arguments)
            return call
        return {name: binding(name) for name in names}
