"""fin_skills.algorithms.panel - adjust_prices() and positions()."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fin_skills.algorithms import adjust_prices, positions, rolling, run
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


@pytest.mark.parametrize("task", sorted(CASES))
def test_positions_event_ratios_match_cumulative_factors(task):
    """Both recipes offered by the rule card must implement the independent equations."""
    raw, cumulative, adjusted = _data()
    ratios = cumulative / cumulative.shift(1).fillna(1.0)
    algorithm, options = CASES[task]
    actual = positions(algorithm, raw, ratios, factors="ratio", **options)
    np.testing.assert_allclose(actual.to_numpy(), _reference(task, adjusted), atol=1e-12, rtol=0)


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


@pytest.mark.parametrize("task", sorted(CASES))
def test_adjusted_prices_with_split_factors_are_refused(task):
    """codegen-enforced-v2: 8 of 12 14B library programs called adjust_prices() and then passed
    its output to positions() together with the split factors, applying the splits twice. The
    result ran without error and was wrong. It must now fail loudly and say how to fix it."""
    raw, factors, _ = _data()
    alg, kw = CASES[task]
    adjusted = adjust_prices(raw, factors)
    assert adjusted.attrs["split_adjusted"] is True
    with pytest.raises(ValueError, match="twice"):
        positions(alg, adjusted, factors, **kw)
    with pytest.raises(ValueError, match="split_factors=None"):
        adjust_prices(adjusted, factors)
    # The two correct spellings agree exactly.
    np.testing.assert_array_equal(positions(alg, adjusted, None, **kw).to_numpy(),
                                  positions(alg, raw, factors, **kw).to_numpy())


def test_adjusted_series_is_marked_and_refused():
    raw, factors, _ = _data()
    adjusted = adjust_prices(raw["A"], factors["A"])
    assert adjusted.attrs["split_adjusted"] is True
    with pytest.raises(ValueError, match="twice"):
        positions("ma_crossover", adjusted, factors["A"], parameters={"fast": 5, "slow": 20})
    # Unmarked inputs are untouched: raw prices are not marked, and neither is a pass-through.
    assert "split_adjusted" not in raw.attrs
    assert "split_adjusted" not in adjust_prices(raw).attrs


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


def test_no_decision_on_any_row_raises_instead_of_returning_zeros():
    raw, factors, _ = _data()
    # cross_sectional_momentum needs 60 returns; a 20-return window can never decide.
    with pytest.raises(ValueError, match="window=20 is shorter than the 60 returns"):
        positions("cross_sectional_momentum", raw, factors, window=20)
    with pytest.raises(ValueError, match="made no decision on any of the 40 rows"):
        positions("inverse_volatility", raw.iloc[:40], factors.iloc[:40], window=60)
    with pytest.raises(ValueError, match="no value on any of the 20 rows"):
        rolling("historical_volatility", raw.iloc[:20], factors.iloc[:20], window=20)
    decided = positions("cross_sectional_momentum", raw, factors, window=60)
    assert decided.abs().to_numpy().sum() > 0


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


ROLLING = [("historical_volatility", dict(window=20, parameters={"periods_per_year": 252})),
           ("ewma_volatility", dict(window=40)),
           ("historical_var_es", dict(window=40, output="expected_shortfall")),
           ("normal_var_es", dict(window=40)),
           ("drift", dict(window=30)), ("mean", dict(window=20))]


@pytest.mark.parametrize("algorithm,kw", ROLLING, ids=[a for a, _ in ROLLING])
def test_rolling_is_lagged_split_invariant_and_shaped(algorithm, kw):
    raw, factors, adjusted = _data()
    out = rolling(algorithm, raw, factors, **kw)
    assert out.index.equals(raw.index) and list(out.columns) == list(raw.columns)
    assert np.isfinite(out.to_numpy()).all() and out.attrs["lag_bars"] == 1
    assert assert_lagged(lambda p: rolling(algorithm, p, factors, **kw), raw, k=64) is None
    np.testing.assert_allclose(rolling(algorithm, adjusted, None, **kw).to_numpy(),
                               out.to_numpy(), atol=1e-12, rtol=0)


def test_rolling_volatility_matches_its_equation():
    raw, factors, adjusted = _data()
    out = rolling("historical_volatility", raw, factors, window=20,
                  parameters={"periods_per_year": 252})
    r = adjusted.pct_change()
    want = (r.rolling(20).std().shift(1) * np.sqrt(252)).fillna(0.0)
    np.testing.assert_allclose(out.to_numpy(), want.to_numpy(), atol=1e-12, rtol=0)
    assert out.attrs["warmup_rows"] == 21


def test_rolling_refuses_misuse():
    raw, factors, _ = _data()
    with pytest.raises(ValueError, match="twice"):
        rolling("drift", adjust_prices(raw, factors), factors, window=30)
    with pytest.raises(ValueError, match="positions"):
        rolling("ma_crossover", raw, factors, window=20)
    with pytest.raises(ValueError, match="no output"):
        rolling("historical_var_es", raw, factors, window=40, output="volatility")
    with pytest.raises(ValueError, match="at least"):
        rolling("historical_var_es", raw, factors, window=10)
    nan = rolling("mean", raw, factors, window=20, warmup="nan")
    assert nan.iloc[:20].isna().all().all() and nan.iloc[20:].notna().all().all()
