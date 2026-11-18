"""Date x asset position tables built from the catalogued algorithms, prefix data only.

The algorithms in this package return ONE allocation (portfolio task) or ONE lagged exposure
series for ONE asset (signal task). Research code needs a table of positions with a row per
date and a column per asset. Assembling that table by hand is where generated code went
wrong in our audits: split factors applied twice, positions computed on the close they trade
over, warm-up rows left as NaN, index or column order lost. These two functions do the
assembly once, with every one of those choices explicit.

    from fin_skills.algorithms import adjust_prices, positions, signal_positions
    w = positions("inverse_volatility", raw, split_factors, window=60)   # adjusts once, inside
    w = positions("ma_crossover", raw, split_factors, fast=5, slow=20)  # = parameters={...}
    w = signal_positions(my_states)    # your own rule: states decided at each close, unlagged

    close = adjust_prices(raw, split_factors)    # only when you need the adjusted closes
    w = positions("inverse_volatility", close, window=60)   # already adjusted: no factors

Give the split factors exactly once. adjust_prices() marks its output as adjusted, and
positions() raises ValueError if it receives that output together with split factors,
instead of applying them a second time.

Convention: the position on row t is held over the return from row t-1 to row t, and with
lag=1 it is computed from adjusted closes through row t-1 only.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

FACTOR_KINDS = ("cumulative", "ratio")
WARMUP = ("zero", "nan")
ADJUSTED = "split_adjusted"      # attrs key set by adjust_prices() on its output


def _mark_adjusted(x):
    x.attrs[ADJUSTED] = True
    return x


def _frame(x: Any, name: str) -> pd.DataFrame:
    if isinstance(x, pd.Series):
        x = x.to_frame(name=x.name if x.name is not None else "asset")
    if not isinstance(x, pd.DataFrame) or x.empty:
        raise TypeError(f"{name} must be a nonempty pandas DataFrame or Series")
    if x.index.has_duplicates or not x.index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be unique and increasing")
    if x.columns.has_duplicates:
        raise ValueError(f"{name} columns must be unique")
    values = x.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains missing or nonfinite values")
    if (values <= 0).any():
        hint = (" (closing prices, not returns: positions() computes returns itself)"
                if name == "prices" else "")
        raise ValueError(f"{name} must be positive{hint}")
    return x.astype(float)


def adjust_prices(prices: pd.DataFrame | pd.Series, split_factors=None, *,
                  factors: str = "cumulative") -> pd.DataFrame | pd.Series:
    """Continuous split-adjusted closes from raw quotes.

    factors="cumulative" (default): split_factors[t] is already the product of every split
        ratio effective on or before row t, known on its event date. The adjusted close is
        prices * split_factors. Do NOT take a cumulative product of these again.
    factors="ratio": split_factors[t] is the ratio of a split effective on row t and 1.0 on
        every other row; its running product is formed here, once.

    split_factors=None returns the prices unchanged (already adjusted). Index and columns of
    prices and split_factors must match exactly; nothing is aligned silently.

    The result carries attrs["split_adjusted"] = True. Pass it to positions() WITHOUT split
    factors; adjust_prices() and positions() both refuse to apply factors to it again.
    """
    if factors not in FACTOR_KINDS:
        raise ValueError(f"factors must be one of {FACTOR_KINDS}, got {factors!r}")
    series = isinstance(prices, pd.Series)
    if split_factors is not None:
        _refuse_readjusting(prices)
    p = _frame(prices, "prices")
    if split_factors is None:
        return p.iloc[:, 0] if series else p
    f = _frame(split_factors, "split_factors")
    if not (p.index.equals(f.index) and list(p.columns) == list(f.columns)):
        raise ValueError("prices and split_factors must have identical index and columns")
    if factors == "ratio":
        f = f.cumprod()
    out = p * f
    return _mark_adjusted(out.iloc[:, 0].copy()) if series else _mark_adjusted(out)


def _refuse_readjusting(prices) -> None:
    if getattr(prices, "attrs", {}).get(ADJUSTED):
        raise ValueError(
            "these prices were already split-adjusted by adjust_prices(); passing split_factors "
            "again would apply the splits twice. Either pass the raw prices with split_factors, "
            "or pass the adjusted prices with split_factors=None.")


def positions_ids(registry=None) -> list[str]:
    """Ids of the catalogued algorithms positions() can run (portfolio and signal tasks)."""
    from . import _DEFAULT
    reg = registry if registry is not None else _DEFAULT
    return sorted(a.id for a in reg.algorithms() if a.task in ("portfolio", "signal")
                  and reg.status(a.id) == "ready")


def _spec_for_positions(reg, algorithm_id):
    """The catalogue entry, or an error that says how to call positions() instead."""
    ids = positions_ids(reg)
    if callable(algorithm_id):
        raise TypeError(
            "positions() takes the id of a catalogued algorithm, not a function. If the rule is "
            "one of these, pass its id: " + ", ".join(ids) + ". For your own rule, compute a "
            "date x asset table of states decided at each close (row t uses closes through t; do "
            "not shift it) and pass it to signal_positions(states), which lags, fills the warm-up "
            "and scales it the same way positions() does.")
    if not isinstance(algorithm_id, str):
        raise TypeError(f"algorithm_id must be a catalogue id string, got "
                        f"{type(algorithm_id).__name__}. Call positions(algorithm_id, prices, "
                        f"split_factors, ...); ids: {', '.join(ids)}")
    try:
        return reg.get(algorithm_id)
    except KeyError:
        from .core import suggest
        close = suggest(algorithm_id, ids)
        hint = f" Did you mean {' or '.join(repr(c) for c in close)}?" if close else ""
        raise KeyError(f"unknown algorithm: {algorithm_id!r}.{hint} positions() accepts: "
                       f"{', '.join(ids)}. For your own rule use signal_positions().") from None


def _merge_parameters(parameters, extra) -> dict:
    params = dict(parameters or {})
    both = sorted(set(params) & set(extra))
    if both:
        raise TypeError(f"{both} given both in parameters= and as keyword arguments")
    params.update(extra)
    return params


def _refuse_unused_arguments(spec, algorithm_id, window, rebalance_every, scale):
    """An argument that does not apply to this kind of algorithm would otherwise be ignored and
    the result would look right: window=15 for a signal rule keeps its default lookback."""
    if spec.task == "signal":
        unused = [f"{k}={v!r}" for k, v in (("window", window), ("rebalance_every", rebalance_every))
                  if v is not None and not (k == "rebalance_every" and v == 1)]
        if unused:
            verb = "apply" if len(unused) > 1 else "applies"
            raise TypeError(f"{' and '.join(unused)} {verb} to portfolio algorithms only and would "
                            f"be ignored: {algorithm_id} is a signal algorithm. Pass the rule's own "
                            f"parameters as keywords instead, e.g. positions({algorithm_id!r}, "
                            f"prices, split_factors, lookback=15) for its rolling window; an "
                            f"unknown keyword raises and lists the accepted ones.")
    elif spec.task == "portfolio" and not (isinstance(scale, str) and scale == "equal"):
        raise TypeError(f"scale={scale!r} applies to signal algorithms only and would be ignored: "
                        f"{algorithm_id} is a portfolio algorithm, whose weights are returned as "
                        "computed")


def _needed(spec, window: int | None, parameters: Mapping[str, Any]) -> int:
    lookback = parameters.get("lookback")
    need = max(int(spec.min_observations or 1), int(window or 0))
    if isinstance(lookback, (int, np.integer)) and not isinstance(lookback, bool):
        need = max(need, int(lookback))
    return need


def positions(algorithm_id: str, prices: pd.DataFrame | pd.Series, split_factors=None, *,
              factors: str = "cumulative", window: int | None = None,
              rebalance_every: int = 1, lag: int = 1, warmup: str = "zero",
              scale: float | str = "equal", parameters: Mapping[str, Any] | None = None,
              registry=None, **algorithm_parameters) -> pd.DataFrame:
    """Final positions: one row per date, one column per asset, decided lag bars ahead.

    The result is ready to use: already lagged, warm-up rows filled and, for signal
    algorithms, already multiplied by scale (1 / number of assets by default). Do not shift
    it or divide it by the number of assets again.

    algorithm_id   : the id of a catalogued portfolio algorithm (input asset_returns) or
                     signal algorithm (input prices or returns), e.g. "rsi_reversion"; the
                     list is below and in catalog(). Not a function: for your own rule use
                     signal_positions().
    prices         : raw closes, one column per asset (a Series is one asset).
    split_factors  : see adjust_prices(); None means prices are already adjusted. The output
                     of adjust_prices() must come with split_factors=None, or this raises.
    window         : portfolio algorithms only - how many trailing simple returns each
                     decision sees. None passes the whole available history. Rows without
                     enough history are warm-up rows.
    rebalance_every: portfolio algorithms only - recompute every n rows from the first
                     decision row and hold the target weights in between.
    lag            : bars between the last close a position may read and the row it is on.
    warmup         : "zero" (default) fills rows before the first decision with 0.0;
                     "nan" leaves them missing so they stay visibly unavailable.
    scale          : signal algorithms only - per-asset exposures in [-1, 1] are multiplied
                     by this number, or by 1 / number of assets for "equal" (default).
    parameters     : the algorithm's parameters, e.g. {"lookback": 20}. They may also be given
                     as keyword arguments: positions("rsi_reversion", prices, lookback=10).

    The returned frame has exactly the input index and columns. Its attrs record the
    algorithm, lag, warm-up rows and parameters. Errors from the algorithm are raised, not
    turned into zeros. So is a result with no decision on any row (a window shorter than
    the algorithm needs, or a panel shorter than its warm-up), which would otherwise be all
    zeros.
    """
    from . import _DEFAULT
    reg = registry if registry is not None else _DEFAULT
    params = _merge_parameters(parameters, algorithm_parameters)
    for key, value in (("lag", lag), ("rebalance_every", rebalance_every)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{key} must be an integer >= 1, got {value!r}")
    if window is not None and (isinstance(window, bool) or not isinstance(window, (int, np.integer))
                               or window < 2):
        raise ValueError(f"window must be None or an integer >= 2, got {window!r}")
    if warmup not in WARMUP:
        raise ValueError(f"warmup must be one of {WARMUP}, got {warmup!r}")
    spec = _spec_for_positions(reg, algorithm_id)
    _refuse_unused_arguments(spec, algorithm_id, window, rebalance_every, scale)
    if split_factors is not None:
        _refuse_readjusting(prices)
    close = adjust_prices(_frame(prices, "prices"), split_factors, factors=factors)
    cols, index = close.columns, close.index
    out = pd.DataFrame(np.nan, index=index, columns=cols)

    if spec.task == "portfolio":
        if "asset_returns" not in spec.inputs:
            raise ValueError(f"{algorithm_id} needs inputs {spec.inputs}; positions() supplies "
                             "asset_returns only")
        returns = close.pct_change(fill_method=None).iloc[1:]
        need = _needed(spec, window, params)
        if window is not None and window < need:
            # Every decision would see fewer returns than the algorithm needs, so no row could
            # ever be decided and the result would be warm-up (zeros) throughout.
            raise ValueError(f"window={window} is shorter than the {need} returns "
                             f"{algorithm_id} needs (its minimum or lookback); pass "
                             f"window >= {need} or window=None")
        first = None
        for t in range(len(index)):
            end = t - lag                      # last close the decision may read
            if end < 1:
                continue
            hist = returns.iloc[:end]          # returns ending at close `end`
            if window is not None:
                hist = hist.iloc[-window:]
            if len(hist) < need:
                continue
            if first is None:
                first = t
            if (t - first) % rebalance_every:
                out.iloc[t] = out.iloc[t - 1]
                continue
            w = reg.run(algorithm_id, {"asset_returns": hist}, **params)
            w = pd.Series(w, index=cols) if not isinstance(w, pd.Series) else w.reindex(cols)
            if w.isna().any():
                raise ValueError(f"{algorithm_id} returned weights that do not cover every asset")
            out.iloc[t] = w.to_numpy(dtype=float)
    elif spec.task == "signal":
        if isinstance(scale, str):
            if scale != "equal":
                raise ValueError("scale must be a number or 'equal'")
            factor = 1.0 / len(cols)
        else:
            factor = float(scale)
            if not np.isfinite(factor):
                raise ValueError("scale must be finite")
        for c in cols:
            if "prices" in spec.inputs:
                raw = reg.run(algorithm_id, {"prices": close[c]}, **params)
            elif "returns" in spec.inputs:
                r = close[c].pct_change(fill_method=None).iloc[1:]
                raw = reg.run(algorithm_id, {"returns": r}, **params)
            else:
                raise ValueError(f"{algorithm_id} needs inputs {spec.inputs}")
            # Read the built-in lag before reindexing: a new Series does not keep attrs. An
            # unlabelled result is treated as unlagged, so it is shifted the full `lag`.
            built_in = int(getattr(raw, "attrs", {}).get("lag_bars", 0))
            s = pd.Series(raw, dtype=float).reindex(index)
            if built_in > lag:
                raise ValueError(f"{algorithm_id} is already lagged {built_in} bars, more than lag")
            out[c] = s.shift(lag - built_in) * factor
    else:
        raise ValueError(f"positions() supports portfolio and signal algorithms, not "
                         f"{spec.task!r} ({algorithm_id})")

    leading = out.notna().any(axis=1).to_numpy()
    if not leading.any():
        # All warm-up: with warmup="zero" this would be a frame of zeros that looks like a
        # valid flat book.
        raise ValueError(f"{algorithm_id} made no decision on any of the {len(out)} rows: "
                         f"the panel is shorter than its warm-up plus lag={lag}")
    warm_rows = int(np.argmax(leading))
    started = out.notna().cummax()               # per column: has a decision been made yet
    if (out.isna() & started).to_numpy().any():
        raise ValueError(f"{algorithm_id} produced missing positions after its warm-up rows")
    if warmup == "zero":
        out = out.fillna(0.0)
    out.attrs.update(algorithm=algorithm_id, task=spec.task, lag_bars=int(lag),
                     warmup_rows=warm_rows, warmup=warmup, factors=factors,
                     window=window, rebalance_every=int(rebalance_every), parameters=params)
    return out


def signal_positions(states: pd.DataFrame | pd.Series, *, lag: int = 1, warmup: str = "zero",
                     scale: float | str = "equal") -> pd.DataFrame:
    """Final positions from YOUR OWN rule's states, with the same timing as positions().

    states : date x asset table (a Series is one asset) of the state your rule decides at
             each close, e.g. +1 long, -1 short, 0 flat; row t may use closes through row t.
             Do NOT shift it: this function applies the lag. Rows before your rule is
             defined are NaN (warm-up); a NaN after that is an error.
    lag    : the position on row t is the state decided at close t - lag.
    warmup : "zero" (default) fills rows before the first decision with 0.0; "nan" keeps them.
    scale  : states are multiplied by this number, or by 1 / number of assets for "equal"
             (default). Do not divide by the number of assets again.

    Compute your states from split-adjusted closes: adjust_prices(prices, split_factors).
    """
    frame = states.to_frame() if isinstance(states, pd.Series) else states
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise TypeError("states must be a nonempty pandas DataFrame or Series")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("states index must be unique and increasing")
    if isinstance(lag, bool) or not isinstance(lag, (int, np.integer)) or lag < 1:
        raise ValueError(f"lag must be an integer >= 1, got {lag!r}")
    if warmup not in WARMUP:
        raise ValueError(f"warmup must be one of {WARMUP}, got {warmup!r}")
    values = frame.astype(float)
    if np.isinf(values.to_numpy()).any():
        raise ValueError("states contain infinite values")
    started = values.notna().cummax()
    if (values.isna() & started).to_numpy().any():
        raise ValueError("states have missing values after a column's first decision; only "
                         "warm-up rows before it may be NaN")
    if isinstance(scale, str):
        if scale != "equal":
            raise ValueError("scale must be a number or 'equal'")
        factor = 1.0 / frame.shape[1]
    else:
        factor = float(scale)
        if not np.isfinite(factor):
            raise ValueError("scale must be finite")
    out = values.shift(lag) * factor
    leading = out.notna().any(axis=1).to_numpy()
    if not leading.any():
        raise ValueError(f"no decision on any of the {len(out)} rows: the states are all "
                         f"warm-up, or shorter than lag={lag}")
    warm_rows = int(np.argmax(leading))
    if warmup == "zero":
        out = out.fillna(0.0)
    out.attrs.update(algorithm="signal_positions", task="signal", lag_bars=int(lag),
                     warmup_rows=warm_rows, warmup=warmup)
    return out


ROLLING_TASKS = {"volatility": "volatility", "risk": "var", "forecast": None}


def rolling(algorithm_id: str, prices: pd.DataFrame | pd.Series, split_factors=None, *,
            window: int, output: str | None = None, factors: str = "cumulative", lag: int = 1,
            warmup: str = "zero", parameters: Mapping[str, Any] | None = None,
            registry=None) -> pd.DataFrame:
    """One value per date and asset from a volatility, risk or forecast algorithm.

    The companion of positions() for algorithms that return a number rather than a weight.
    Row t holds the algorithm's result on the `window` most recent observations ending lag
    bars before t, so with lag=1 it is known at close t-1 and reads nothing from row t on.

    algorithm_id : a catalogued volatility or risk algorithm (they read simple returns of the
                   adjusted closes) or forecast algorithm (it reads the adjusted closes and its
                   one-step forecast is used); see catalog().
    prices       : raw closes, one column per asset (a Series is one asset).
    split_factors: as in positions(); None means prices are already adjusted. The output of
                   adjust_prices() must come with split_factors=None, or this raises.
    window       : observations per value: returns for volatility and risk, closes for forecast.
    output       : the result field to keep. Defaults: "volatility" for volatility algorithms,
                   "var" for risk algorithms ("expected_shortfall" is the other risk field).
    warmup       : "zero" (default) fills rows without a full window with 0.0; "nan" leaves them.
    parameters   : passed to the algorithm unchanged (e.g. periods_per_year, confidence).

    The returned frame has exactly the input index and columns; its attrs record the choices.
    """
    from . import _DEFAULT
    reg = registry if registry is not None else _DEFAULT
    params = dict(parameters or {})
    if isinstance(lag, bool) or not isinstance(lag, (int, np.integer)) or lag < 1:
        raise ValueError(f"lag must be an integer >= 1, got {lag!r}")
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)) or window < 2:
        raise ValueError(f"window must be an integer >= 2, got {window!r}")
    if warmup not in WARMUP:
        raise ValueError(f"warmup must be one of {WARMUP}, got {warmup!r}")
    spec = reg.get(algorithm_id)
    if spec.task not in ROLLING_TASKS:
        raise ValueError(f"rolling() supports {sorted(ROLLING_TASKS)} algorithms, not "
                         f"{spec.task!r} ({algorithm_id}); use positions() for weights")
    if window < int(spec.min_observations or 1):
        raise ValueError(f"{algorithm_id} needs at least {spec.min_observations} observations")
    field = output if output is not None else ROLLING_TASKS[spec.task]
    if split_factors is not None:
        _refuse_readjusting(prices)
    close = adjust_prices(_frame(prices, "prices"), split_factors, factors=factors)
    out = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    for c in close.columns:
        if spec.task == "forecast":
            if set(spec.inputs) != {"series"}:
                raise ValueError(f"{algorithm_id} needs inputs {spec.inputs}; rolling() "
                                 "supplies series only")
            source, first = close[c].to_numpy(), window + lag - 1
        else:
            source = close[c].pct_change(fill_method=None).to_numpy()   # NaN on row 0
            first = window + lag
        for t in range(first, len(close)):
            end = t - lag + 1                         # one past the last observation used
            chunk = source[end - window:end]
            key = "series" if spec.task == "forecast" else "returns"
            result = reg.run(algorithm_id, {key: chunk}, **params)
            if spec.task == "forecast":
                value = float(np.asarray(result, dtype=float).ravel()[0])
            else:
                if field not in result:
                    raise ValueError(f"{algorithm_id} has no output {field!r}; it returns "
                                     f"{sorted(result)}")
                value = float(result[field])
            out.iloc[t, out.columns.get_loc(c)] = value
    if not out.notna().any(axis=None):
        raise ValueError(f"{algorithm_id} produced no value on any of the {len(out)} rows: "
                         f"the panel is shorter than window={window} plus lag={lag}")
    warm_rows = int(np.argmax(out.notna().any(axis=1).to_numpy()))
    if warmup == "zero":
        out = out.fillna(0.0)
    out.attrs.update(algorithm=algorithm_id, task=spec.task, output=field, window=int(window),
                     lag_bars=int(lag), warmup_rows=warm_rows, warmup=warmup, factors=factors,
                     parameters=params)
    return out
