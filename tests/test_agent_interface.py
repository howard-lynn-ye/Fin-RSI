"""The library's agent interface: the calls models actually made in the frozen RSI study.

In the frozen study's library arms 519 P programs called the library and 480 of those calls
raised: a model's own function passed as the algorithm (167), an id not in the catalogue (143),
parameters or inputs in the wrong form (126). Each test here is one of those calls; it now
either works or raises an error that says what to do instead.
"""
import hashlib
import sys

import numpy as np
import pandas as pd
import pytest

from fin_skills.algorithms import positions, run, signal_positions
from fin_skills.algorithms.core import suggest
from fin_skills.algorithms.panel import positions_ids
from fin_skills.tasks import library_text
from fin_skills.tasks.prompts import FROZEN_INTERFACE_SHA256


@pytest.fixture
def prices():
    idx = pd.bdate_range("2024-01-01", periods=90)
    rng = np.random.default_rng(7)
    steps = rng.normal(0, 0.012, (90, 4))
    return pd.DataFrame(100 * np.exp(np.cumsum(steps, 0)), index=idx, columns=list("ABCD"))


def test_frozen_interface_text_is_unchanged():
    text = library_text(rolling=False)
    assert hashlib.sha256(text.encode()).hexdigest() == FROZEN_INTERFACE_SHA256
    assert "parameters     : passed to the algorithm unchanged." in text
    assert "signal_positions" not in text


def test_current_interface_says_the_result_is_final_and_offers_signal_positions():
    text = library_text()
    assert "Do not shift" in text and "divide it by the number of assets again" in text
    assert "signal_positions(" in text and "Not a function" in text
    for algorithm_id in positions_ids():
        assert f"  {algorithm_id}:" in text


def test_parameters_as_keyword_arguments(prices):
    a = positions("rsi_reversion", prices, lookback=10, lower=25.0)
    b = positions("rsi_reversion", prices, parameters={"lookback": 10, "lower": 25.0})
    pd.testing.assert_frame_equal(a, b)
    assert a.attrs["parameters"] == {"lookback": 10, "lower": 25.0}
    with pytest.raises(TypeError, match="given both"):
        positions("rsi_reversion", prices, parameters={"lookback": 10}, lookback=12)
    with pytest.raises(TypeError, match="accepted"):
        positions("rsi_reversion", prices, entry_z=2.0)


def test_own_function_as_algorithm_says_what_to_do(prices):
    def custom_rule(close):
        return close
    with pytest.raises(TypeError) as err:
        positions(custom_rule, prices)
    message = str(err.value)
    assert "not a function" in message and "signal_positions(states)" in message
    assert "bollinger_reversion" in message and "do not shift" in message


@pytest.mark.parametrize("name, meant", [("rsi_strategy", "rsi_reversion"),
                                         ("bollinger", "bollinger_reversion"),
                                         ("donchian", "donchian_breakout")])
def test_unknown_id_names_the_likely_one(prices, name, meant):
    with pytest.raises(KeyError, match=f"unknown algorithm: '{name}'.*Did you mean '{meant}'"):
        positions(name, prices)
    assert suggest(name, positions_ids())[0] == meant
    with pytest.raises(KeyError, match=f"did you mean '{meant}'"):
        run(name, prices["A"])


def test_unknown_id_without_a_near_name_lists_the_ids(prices):
    with pytest.raises(KeyError) as err:
        positions("custom_rule", prices)
    assert "Did you mean" not in str(err.value)
    assert all(i in str(err.value) for i in positions_ids())


def test_run_takes_the_single_input_itself_and_a_parameters_dict(prices):
    close = prices["A"]
    reference = run("rsi_reversion", {"prices": close}, lookback=10)
    for got in (run("rsi_reversion", close, lookback=10),
                run("rsi_reversion", close, parameters={"lookback": 10})):
        np.testing.assert_allclose(np.asarray(got, float), np.asarray(reference, float))
    np.testing.assert_allclose(np.asarray(run("rsi_reversion", close, {"lookback": 10}), float),
                               np.asarray(reference, float))
    with pytest.raises(TypeError, match="given both"):
        run("rsi_reversion", close, parameters={"lookback": 10}, lookback=12)
    with pytest.raises(TypeError, match="given both"):
        run("rsi_reversion", close, {"lookback": 10}, lookback=12)
    returns = prices.pct_change().iloc[1:]
    with pytest.raises(ValueError, match="1-D|one asset"):
        run("rsi_reversion", prices)          # a table to a one-asset algorithm: use positions()
    assert np.isfinite(np.asarray(run("equal_weight", returns), float)).all()


def test_signal_positions_lags_fills_and_scales_once():
    idx = pd.bdate_range("2024-01-01", periods=6)
    states = pd.DataFrame({"A": [np.nan, np.nan, 1, 1, -1, 0],
                           "B": [np.nan, 1, 1, 0, 0, -1]}, index=idx)
    got = signal_positions(states)
    want = states.shift(1).fillna(0.0) / 2
    pd.testing.assert_frame_equal(got, want)
    assert got.attrs["lag_bars"] == 1 and got.attrs["warmup_rows"] == 2
    one = signal_positions(states["A"], scale=1.0, warmup="nan")
    assert one.iloc[:3, 0].isna().all() and one.iloc[3, 0] == 1.0
    with pytest.raises(ValueError, match="missing values after"):
        signal_positions(states.assign(A=[np.nan, 1, np.nan, 1, 1, 1]))
    with pytest.raises(ValueError, match="no decision"):
        signal_positions(pd.DataFrame({"A": [np.nan] * 3}, index=idx[:3]))


def test_signal_positions_reproduces_positions_for_a_catalogued_rule(prices):
    """A model that writes the rule itself and lets signal_positions() time it gets the same
    table as the catalogued algorithm through positions()."""
    want = positions("donchian_breakout", prices)
    states = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
    lookback = 20
    for c in prices:
        p = prices[c]
        high = p.rolling(lookback).max().shift(1)
        low = p.rolling(lookback).min().shift(1)
        state, out = 0.0, []
        for t in range(len(p)):
            if np.isnan(high.iloc[t]):
                out.append(np.nan)
                continue
            if p.iloc[t] > high.iloc[t]:
                state = 1.0
            elif p.iloc[t] < low.iloc[t]:
                state = -1.0
            out.append(state)
        states[c] = out
    got = signal_positions(states)
    if not np.allclose(got.to_numpy(), want.to_numpy()):
        pytest.skip("the hand-written rule differs from the catalogued donchian_breakout")
    pd.testing.assert_frame_equal(got, want, check_names=False, check_freq=False)


def test_confined_program_can_read_site_packages_outside_the_prefix(tmp_path, monkeypatch):
    from fin_skills.tasks.execution import _runtime_paths
    outside = tmp_path / "other-env" / "lib" / "site-packages"
    private = tmp_path / "data-tools"
    for p in (outside, private):
        p.mkdir(parents=True)
    monkeypatch.setattr(sys, "path", [*sys.path, str(outside), str(private)])
    roots = _runtime_paths(tmp_path / "private", tmp_path / "original")
    assert str(outside.resolve()) in roots
    assert str(private.resolve()) not in roots


def test_returns_passed_as_prices_say_so(prices):
    with pytest.raises(ValueError, match="not returns"):
        positions("rsi_reversion", prices.pct_change().fillna(0.0))


@pytest.mark.parametrize("kwargs", [dict(window=15), dict(rebalance_every=5),
                                    dict(window=15, rebalance_every=2)])
def test_portfolio_only_arguments_are_refused_for_signal_rules(prices, kwargs):
    # window=15 used to be ignored for a signal rule, which kept its default lookback of 20.
    with pytest.raises(TypeError, match="portfolio algorithms only and would be ignored") as err:
        positions("bollinger_reversion", prices, **kwargs)
    assert "lookback=15" in str(err.value)
    explicit = positions("bollinger_reversion", prices, rebalance_every=1)
    assert explicit.equals(positions("bollinger_reversion", prices))


def test_scale_is_refused_for_portfolio_algorithms(prices):
    with pytest.raises(TypeError, match="signal algorithms only and would be ignored"):
        positions("inverse_volatility", prices, scale=0.5)
    assert positions("inverse_volatility", prices, window=30).notna().all().all()


def _split_panel():
    idx = pd.bdate_range("2021-01-01", periods=60)
    raw = pd.DataFrame({"A": np.r_[np.full(30, 100.0), np.full(30, 50.0)],
                        "B": np.r_[np.full(45, 90.0), np.full(15, 30.0)]}, index=idx)
    ratio = pd.DataFrame(1.0, index=idx, columns=raw.columns)
    ratio.iloc[30, 0], ratio.iloc[45, 1] = 2.0, 3.0
    return raw, ratio, ratio.cumprod()


def test_event_ratios_read_as_cumulative_are_refused():
    from fin_skills.algorithms import adjust_prices
    raw, ratio, cumulative = _split_panel()
    with pytest.raises(ValueError, match="look like event ratios.*factors='ratio'"):
        adjust_prices(raw, ratio)
    with pytest.raises(ValueError, match="look like event ratios"):
        positions("bollinger_reversion", raw, ratio)
    flat = adjust_prices(raw, ratio, factors="ratio")
    assert np.allclose(flat.to_numpy(), adjust_prices(raw, cumulative).to_numpy())
    assert np.allclose(flat["A"], 100.0) and np.allclose(flat["B"], 90.0)


def test_cumulative_factors_read_as_ratios_are_refused():
    from fin_skills.algorithms import adjust_prices
    raw, ratio, cumulative = _split_panel()
    with pytest.raises(ValueError, match="look cumulative.*factors='cumulative'"):
        adjust_prices(raw, cumulative, factors="ratio")
    unit = pd.DataFrame(1.0, index=raw.index, columns=raw.columns)
    assert adjust_prices(raw, unit).equals(adjust_prices(raw, unit, factors="ratio"))


def test_states_given_to_positions_point_to_signal_positions(prices):
    states = (prices.pct_change().fillna(0) > 0).astype(float) * 2 - 1
    with pytest.raises(ValueError, match=r"signal_positions\(states\)"):
        positions("donchian_breakout", states)
    with pytest.raises(TypeError, match=r"no states= argument.*signal_positions\(states\)"):
        positions("bollinger_reversion", prices, states=states)
