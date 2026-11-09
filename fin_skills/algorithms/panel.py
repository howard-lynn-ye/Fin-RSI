"""Date x asset position tables built from the catalogued algorithms, prefix data only.

The algorithms in this package return ONE allocation (portfolio task) or ONE lagged exposure
series for ONE asset (signal task). Research code needs a table of positions with a row per
date and a column per asset. Assembling that table by hand is where generated code went
wrong in our audits: split factors applied twice, positions computed on the close they trade
over, warm-up rows left as NaN, index or column order lost. These two functions do the
assembly once, with every one of those choices explicit.

    from fin_skills.algorithms import adjust_prices, positions
    w = positions("inverse_volatility", raw, split_factors, window=60)   # adjusts once, inside
    w = positions("ma_crossover", raw, split_factors, parameters={"fast": 5, "slow": 20})

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
        raise ValueError(f"{name} must be positive")
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
              registry=None) -> pd.DataFrame:
    """Positions with one row per date and one column per asset, decided lag bars ahead.

    algorithm_id   : a catalogued portfolio algorithm (input asset_returns) or signal
                     algorithm (input prices or returns); see catalog().
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
    parameters     : passed to the algorithm unchanged.

    The returned frame has exactly the input index and columns. Its attrs record the
    algorithm, lag, warm-up rows and parameters. Errors from the algorithm are raised, not
    turned into zeros. So is a result with no decision on any row (a window shorter than
    the algorithm needs, or a panel shorter than its warm-up), which would otherwise be all
    zeros.
    """
    from . import _DEFAULT
    reg = registry if registry is not None else _DEFAULT
    params = dict(parameters or {})
    for key, value in (("lag", lag), ("rebalance_every", rebalance_every)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{key} must be an integer >= 1, got {value!r}")
    if window is not None and (isinstance(window, bool) or not isinstance(window, (int, np.integer))
                               or window < 2):
        raise ValueError(f"window must be None or an integer >= 2, got {window!r}")
    if warmup not in WARMUP:
        raise ValueError(f"warmup must be one of {WARMUP}, got {warmup!r}")
    spec = reg.get(algorithm_id)
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
