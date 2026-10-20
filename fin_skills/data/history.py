"""Explicit snapshot and return-history contracts for caller-supplied daily closes.

No download, imputation, strategy selection, or detection of undeclared adjustment.
Corporate-action coverage and the input convention must be declared by the caller.
"""
from numbers import Integral

import numpy as np
import pandas as pd


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _day(value, name):
    if not isinstance(value, str):
        raise ValueError(f"{name} must be YYYY-MM-DD")
    try:
        stamp = pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be YYYY-MM-DD") from exc
    if pd.isna(stamp) or stamp.strftime("%Y-%m-%d") != value:
        raise ValueError(f"{name} must be YYYY-MM-DD")
    return value


def _visible(quotes, as_of, tickers):
    _day(as_of, "as_of")
    required = {"date", "ticker", "close"}
    if not isinstance(quotes, pd.DataFrame) or not required.issubset(quotes.columns):
        raise ValueError("quotes must be a DataFrame with date,ticker,close columns")
    q = quotes.loc[:, ["date", "ticker", "close"]].copy()
    # Validate dates before comparing, and exclude future rows before inspecting values.
    q["date"] = pd.Series([_day(d, "quotes.date") for d in q.date], index=q.index, dtype=object)
    q = q[q.date <= as_of]
    if tickers is None:
        names = sorted(q.ticker.dropna().unique().tolist())
    else:
        if (not isinstance(tickers, (list, tuple)) or not tickers
                or any(not isinstance(t, str) or not t for t in tickers)
                or len(set(tickers)) != len(tickers)):
            raise ValueError("tickers must be a nonempty list of distinct ticker strings")
        names = list(tickers)
    if not names or any(not isinstance(t, str) or not t for t in names):
        raise ValueError("no valid visible tickers")
    missing = set(names) - set(q.ticker)
    if missing:
        raise ValueError(f"no visible quotes for tickers: {sorted(missing)}")
    q = q[q.ticker.isin(names)]
    if q.duplicated(["date", "ticker"]).any():
        raise ValueError("duplicate date/ticker quotes; reconcile the source first")
    prices = q.pivot(index="date", columns="ticker", values="close").sort_index()
    return prices.reindex(columns=names)


def _validate_prices(prices):
    try:
        result = prices.astype(float)
    except (ValueError, TypeError) as exc:
        raise ValueError("close prices must be finite positive numbers") from exc
    if not np.isfinite(result.to_numpy()).all() or (result <= 0).any().any():
        raise ValueError("missing, nonfinite, or nonpositive close prices; no forward fill is applied")
    return result


def market_snapshot(quotes, *, as_of, tickers=None):
    """One common session of price levels. Never a return-estimation window."""
    prices = _validate_prices(_visible(quotes, as_of, tickers).tail(1))
    date = prices.index[-1]
    return {"ok": True, "kind": "snapshot", "as_of": as_of, "last_session": date,
            "stale": date != as_of, "price_sessions": 1, "return_samples": 0,
            "suitable_for_return_estimation": False,
            "rows": [{"date": date, "ticker": t, "close": float(prices.loc[date, t])}
                     for t in prices.columns],
            "next_step": "Use prepare_history for complete, validated return history."}


def prepare_history(quotes, actions=None, *, as_of, lookback=252, tickers=None,
                    input_adjustment=None, actions_complete=False):
    """Return a dictionary with complete prices/returns DataFrames and metadata.

    input_adjustment='raw' requires explicitly complete visible corporate actions;
    'total_return' accepts caller-declared total-return-adjusted closes without events.
    Actions use date,ticker,kind,value: split ratios are new/old shares and dividends
    are cash per share on the ex-date. Same-day split+dividend events are rejected
    because their per-share basis is ambiguous. Events must be on observed sessions.
    Daily total returns reinvest cash dividends at that session's close. Returned
    adjusted prices are a total-return index anchored to the last raw close.
    as_of is a date cutoff, NOT proof that the source was available historically.
    """
    n = _positive_integer(lookback, "lookback (number of returns)")
    if input_adjustment not in ("raw", "total_return"):
        raise ValueError("declare input_adjustment='raw' or 'total_return'; never infer it")
    prices = _visible(quotes, as_of, tickers).tail(n + 1)
    if len(prices) < n + 1:
        raise ValueError(f"need {n + 1} price sessions for {n} returns; found {len(prices)}")
    prices = _validate_prices(prices)
    if prices.index[-1] != as_of:
        raise ValueError(f"latest visible session is {prices.index[-1]}, not as_of={as_of}")
    event_count = 0
    gross = prices.div(prices.shift(1)).iloc[1:]
    if input_adjustment == "total_return":
        if actions is not None and len(actions):
            raise ValueError("do not apply corporate actions to total_return inputs twice")
    else:
        if actions_complete is not True:
            raise ValueError("raw history requires actions_complete=True after verifying action coverage")
        if not isinstance(actions, pd.DataFrame) or not {"date", "ticker", "kind", "value"}.issubset(actions.columns):
            raise ValueError("provide corporate actions with date,ticker,kind,value, even if empty")
        events = actions.copy()
        events["date"] = pd.Series([_day(d, "actions.date") for d in events.date],
                                   index=events.index, dtype=object)
        events = events[(events.date > prices.index[0]) & (events.date <= as_of)
                        & events.ticker.isin(prices.columns)]
        if events.duplicated(["date", "ticker", "kind"]).any():
            raise ValueError("duplicate corporate actions; reconcile before computing returns")
        for (date, ticker), group in events.groupby(["date", "ticker"]):
            if date not in gross.index:
                raise ValueError(f"action on unobserved session {date}: {ticker}")
            if len(group) > 1:
                raise ValueError("same-session action types require an explicit per-share convention")
            event = group.iloc[0]
            value = float(event.value)
            if not np.isfinite(value) or value < 0 or (event.kind == "split" and value == 0):
                raise ValueError("corporate action values must be finite and valid")
            if event.kind == "split":
                gross.loc[date, ticker] *= value
            elif event.kind == "dividend":
                previous = prices[ticker].shift(1).loc[date]
                gross.loc[date, ticker] += value / previous
            else:
                raise ValueError(f"unsupported action kind: {event.kind}")
            event_count += 1
    returns = gross - 1
    if not np.isfinite(returns.to_numpy()).all() or (gross <= 0).any().any():
        raise ValueError("invalid adjusted returns")
    if input_adjustment == "raw":
        index = pd.concat([pd.DataFrame(1., index=prices.index[:1], columns=prices.columns),
                           gross.cumprod()])
        adjusted = index.mul(prices.iloc[-1] / index.iloc[-1], axis=1)
    else:
        adjusted = prices.copy()
    metadata = {"kind": "history", "as_of": as_of, "start": prices.index[0],
                "end": prices.index[-1], "price_sessions": len(prices),
                "return_samples": len(returns), "requested_returns": n,
                "tickers": list(prices.columns), "input_adjustment": input_adjustment,
                "output_adjustment": "total_return", "actions_applied": event_count,
                "adjustment_method": "cash dividends reinvested at ex-date close",
                "coverage": "caller-declared; missing corporate actions cannot be inferred",
                "point_in_time": "date-filtered; historical source availability is not verified",
                "complete": True, "suitable_for_return_estimation": True}
    return {"ok": True, "prices": adjusted, "returns": returns, "metadata": metadata}
