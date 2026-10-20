import pandas as pd
import pytest

from benchmarks.agent_study.audit_manual_multisource import replay


def test_first_fee_and_hold_growth_are_kept():
    prices = pd.DataFrame({"SPY": [100., 110., 121.]}, index=["a", "b", "c"])
    nav, detail = replay(prices, {"a": {"SPY": 1.}, "b": None})
    assert nav.iloc[-1] == pytest.approx((1 - .0005) * 1.21)
    assert detail["rebalances"] == 1
    assert detail["fee_drag_initial_capital_pp"] == pytest.approx(.05)


def test_trade_occurs_after_old_position_earns_daily_return():
    prices = pd.DataFrame({"SPY": [100., 110., 220.]}, index=["a", "b", "c"])
    nav, detail = replay(prices, {"a": {"SPY": .5}, "b": {"SPY": 0.}})
    before_sale = (1 - .00025) * 1.05
    after_sale = before_sale - (1 - .00025) * .55 * .0005
    assert nav.iloc[-1] == pytest.approx(after_sale)
    assert nav.loc["b"] == nav.loc["c"]
    assert detail["rebalances"] == 2
