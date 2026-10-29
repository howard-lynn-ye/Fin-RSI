"""fin_skills.algorithms.panel - adjust_prices() and positions()."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fin_skills.algorithms import adjust_prices, positions, run
from fin_skills.core.assert_causal import assert_lagged


def _data(seed: int = 11, n: int = 96):
    rng = np.random.default_rng(seed)
    r = rng.normal([.001, -.0004, .0005, .0002], [.013, .009, .017, .011], (n, 4))
    adjusted = 100 * np.cumprod(1 + r, axis=0)
    factors = np.ones_like(adjusted)
    factors[45:, 0] = 2          # cumulative factors: 2-for-1 on row 45, held afterwards
    factors[70:, 2] = 3
    idx = pd.bdate_range("2024-01-02", periods=n)
    cols = list("ABCD")
    return (pd.DataFrame(adjusted / factors, index=idx, columns=cols),
            pd.DataFrame(factors, index=idx, columns=cols),
            pd.DataFrame(adjusted, index=idx, columns=cols))


def _reference(task: str, adjusted: pd.DataFrame) -> np.ndarray:
    """Explicit-window equations written independently of the library."""
    p = adjusted.to_numpy()
    ret = p[1:] / p[:-1] - 1
    w = np.zeros_like(p)
    for t in range(1, len(p)):
        if task in ("ivol", "xsmom") and t >= 61:
            hist = ret[t - 61:t - 1]
            if task == "ivol":
                inv = 1 / hist.std(axis=0, ddof=1)
                w[t] = inv / inv.sum()
            else:
                m = np.prod(1 + hist, axis=0) - 1
                chosen = np.argsort(-m, kind="stable")[:2]
                w[t, chosen[m[chosen] > 0]] = .5
        elif task == "ma" and t >= 20:
            w[t] = np.sign(p[t - 5:t].mean(axis=0) - p[t - 20:t].mean(axis=0)) / 4
        elif task == "vt" and t >= 21:
            vol = ret[t - 21:t - 1].std(axis=0, ddof=1) * np.sqrt(252)
            w[t] = np.sign(p[t - 1] / p[t - 21] - 1) * np.minimum(1, .10 / vol) / 4
    return w


CASES = {
    "ivol": ("inverse_volatility", dict(window=60)),
    "xsmom": ("cross_sectional_momentum", dict(parameters={"lookback": 60, "top_k": 2})),
    "ma": ("ma_crossover", dict(parameters={"fast": 5, "slow": 20})),
    "vt": ("vol_target_momentum", dict()),
}


@pytest.mark.parametrize("seed", [11, 23, 37])
@pytest.mark.parametrize("task", sorted(CASES))
def test_positions_match_independent_reference(task, seed):
    raw, factors, adjusted = _data(seed)
    alg, kw = CASES[task]
    w = positions(alg, raw, factors, **kw)
    assert list(w.columns) == list(raw.columns) and w.index.equals(raw.index)
    assert np.isfinite(w.to_numpy()).all()
    np.testing.assert_allclose(w.to_numpy(), _reference(task, adjusted), atol=1e-12, rtol=0)


@pytest.mark.parametrize("task", sorted(CASES))
def test_positions_are_lagged_and_split_invariant(task):
    raw, factors, adjusted = _data()
    alg, kw = CASES[task]
    fn = lambda p: positions(alg, p, factors, **kw)
    for k in (32, 64):
        assert assert_lagged(fn, raw, k=k) is None
    same = positions(alg, adjusted, None, **kw)      # already adjusted, unit factors
    np.testing.assert_allclose(same.to_numpy(), fn(raw).to_numpy(), atol=1e-12, rtol=0)


def test_adjust_prices_applies_cumulative_factors_once():
    raw, factors, adjusted = _data()
    np.testing.assert_allclose(adjust_prices(raw, factors).to_numpy(), adjusted.to_numpy())
    # re-cumulating is the defect this function exists to prevent: it doubles the 2-for-1
    wrong = raw * factors.cumprod()
    assert not np.allclose(wrong.to_numpy(), adjusted.to_numpy())


def test_adjust_prices_accepts_event_ratios():
    raw, factors, adjusted = _data()
    ratios = factors / factors.shift(1).fillna(1.0)          # 2.0 on row 45, 3.0 on row 70
    np.testing.assert_allclose(adjust_prices(raw, ratios, factors="ratio").to_numpy(),
                               adjusted.to_numpy())


def test_adjust_prices_refuses_misaligned_or_bad_input():
    raw, factors, _ = _data()
    with pytest.raises(ValueError, match="identical index and columns"):
        adjust_prices(raw, factors[["B", "A", "C", "D"]])
    with pytest.raises(ValueError, match="positive"):
        adjust_prices(raw, -factors)
    bad = raw.copy()
    bad.iloc[3, 1] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        adjust_prices(bad, factors)
    with pytest.raises(ValueError, match="factors must be one of"):
        adjust_prices(raw, factors, factors="events")


def test_series_input_and_series_output():
    raw, factors, adjusted = _data()
    out = adjust_prices(raw["A"], factors["A"])
    assert isinstance(out, pd.Series)
    np.testing.assert_allclose(out.to_numpy(), adjusted["A"].to_numpy())
    w = positions("ma_crossover", raw["A"], factors["A"], parameters={"fast": 5, "slow": 20},
                  scale=1.0)
    assert w.shape == (len(raw), 1)


def test_lag_two_and_nan_warmup():
    raw, factors, _ = _data()
    one = positions("ma_crossover", raw, factors, parameters={"fast": 5, "slow": 20})
    two = positions("ma_crossover", raw, factors, parameters={"fast": 5, "slow": 20}, lag=2)
    np.testing.assert_array_equal(two.to_numpy()[1:], one.to_numpy()[:-1])
    assert assert_lagged(lambda p: positions("ma_crossover", p, factors, lag=2,
                                             parameters={"fast": 5, "slow": 20}),
                         raw, k=40, lag=2) is None
    nan = positions("inverse_volatility", raw, factors, window=60, warmup="nan")
    assert nan.iloc[:61].isna().all().all() and nan.iloc[61:].notna().all().all()
    assert nan.attrs["warmup_rows"] == 61


def test_rebalance_every_holds_targets_between_rebalances():
    raw, factors, _ = _data()
    w = positions("inverse_volatility", raw, factors, window=60, rebalance_every=5)
    daily = positions("inverse_volatility", raw, factors, window=60)
    for t in range(61, len(raw)):
        anchor = 61 + ((t - 61) // 5) * 5
        np.testing.assert_allclose(w.iloc[t].to_numpy(), daily.iloc[anchor].to_numpy())


def test_bad_arguments_raise():
    raw, factors, _ = _data()
    with pytest.raises(ValueError, match="lag"):
        positions("ma_crossover", raw, factors, lag=0)
    with pytest.raises(ValueError, match="window"):
        positions("inverse_volatility", raw, factors, window=1)
    with pytest.raises(ValueError, match="warmup"):
        positions("inverse_volatility", raw, factors, warmup="drop")
    with pytest.raises(ValueError, match="supports portfolio and signal"):
        positions("historical_var_es", raw, factors)
    with pytest.raises(KeyError):
        positions("no_such_algorithm", raw, factors)


def test_run_error_messages_point_to_the_panel_and_the_nan_row():
    raw, factors, _ = _data()
    with pytest.raises(ValueError, match="positions"):
        run("ma_crossover", {"prices": raw})
    with pytest.raises(ValueError, match="pct_change"):
        run("inverse_volatility", {"asset_returns": raw.pct_change()})
