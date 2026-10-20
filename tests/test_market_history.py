import numpy as np
import pandas as pd
import pytest

from fin_skills.data import market_snapshot, prepare_history


def data():
    quotes = pd.DataFrame([
        ("2025-01-01", "A", 100.), ("2025-01-02", "A", 50.),
        ("2025-01-03", "A", 49.), ("2025-01-04", "A", 50.),
        ("2025-01-01", "B", 20.), ("2025-01-02", "B", 21.),
        ("2025-01-03", "B", 22.), ("2025-01-04", "B", 23.),
    ], columns=["date", "ticker", "close"])
    actions = pd.DataFrame([("2025-01-02", "A", "split", 2.),
                            ("2025-01-03", "A", "dividend", 1.)],
                           columns=["date", "ticker", "kind", "value"])
    return quotes, actions


def history(q=None, a=None, **kwargs):
    quotes, actions = data()
    return prepare_history(quotes if q is None else q, actions if a is None else a,
                           as_of="2025-01-03", lookback=2,
                           input_adjustment="raw", actions_complete=True, **kwargs)


def test_split_and_cash_dividend_do_not_create_fake_losses():
    h = history(tickers=["B", "A"])
    assert list(h["returns"].columns) == ["B", "A"]
    np.testing.assert_allclose(h["returns"].A, [0., 0.], atol=1e-12)
    np.testing.assert_allclose(h["returns"].B, [.05, 1/21], atol=1e-12)
    np.testing.assert_allclose(h["prices"].pct_change().iloc[1:], h["returns"], atol=1e-12)
    assert h["metadata"]["price_sessions"] == 3
    assert h["metadata"]["return_samples"] == 2
    assert h["metadata"]["actions_applied"] == 2


def test_future_values_and_events_cannot_change_visible_history():
    q, a = data()
    expected = history()["returns"]
    q.loc[q.date > "2025-01-03", "close"] = -1000
    a.loc[len(a)] = ["2025-01-04", "A", "split", 10000]
    pd.testing.assert_frame_equal(history(q, a)["returns"], expected)


def test_snapshot_is_explicitly_not_history():
    q, _ = data()
    r = market_snapshot(q, as_of="2025-01-03")
    assert r["return_samples"] == 0 and r["price_sessions"] == 1
    assert not r["suitable_for_return_estimation"]
    assert {x["date"] for x in r["rows"]} == {"2025-01-03"}


def test_declared_empty_actions_keep_dates_comparable():
    q, a = data()
    h = history(q[q.ticker == "B"], a.iloc[:0])
    np.testing.assert_allclose(h["returns"].B, [.05, 1/21], atol=1e-12)
    assert h["metadata"]["actions_applied"] == 0


@pytest.mark.parametrize("defect", ["missing", "duplicate", "nan", "zero", "negative", "stale"])
def test_fail_closed_instead_of_filling_or_shortening(defect):
    q, a = data()
    if defect == "missing": q = q.drop(5)
    elif defect == "duplicate": q = pd.concat([q, q.iloc[[0]]])
    elif defect == "stale": q = q[q.date < "2025-01-03"]
    else: q.loc[1, "close"] = {"nan": np.nan, "zero": 0, "negative": -1}[defect]
    with pytest.raises(ValueError): history(q, a)


def test_declarations_and_duplicate_adjustment_are_not_inferred():
    q, a = data()
    with pytest.raises(ValueError, match="declare input_adjustment"):
        prepare_history(q, a, as_of="2025-01-03", lookback=2)
    with pytest.raises(ValueError, match="actions_complete"):
        prepare_history(q, a, as_of="2025-01-03", lookback=2, input_adjustment="raw")
    with pytest.raises(ValueError, match="twice"):
        prepare_history(q, a, as_of="2025-01-03", lookback=2, input_adjustment="total_return")
    h = prepare_history(q, as_of="2025-01-03", lookback=2, input_adjustment="total_return")
    assert h["metadata"]["input_adjustment"] == "total_return"


@pytest.mark.parametrize("defect", ["duplicate", "ambiguous", "bad_value", "unknown"])
def test_invalid_actions_do_not_silently_change_returns(defect):
    q, a = data()
    if defect == "duplicate": a = pd.concat([a, a.iloc[[0]]])
    elif defect == "ambiguous": a.loc[len(a)] = ["2025-01-02", "A", "dividend", 1.]
    elif defect == "bad_value": a.loc[0, "value"] = 0.
    else: a.loc[0, "kind"] = "spinoff"
    with pytest.raises(ValueError): history(q, a)


@pytest.mark.parametrize("n", [True, 0, -1, 1.5, "2", 20])
def test_invalid_or_unavailable_lookback_is_rejected(n):
    q, a = data()
    with pytest.raises(ValueError):
        prepare_history(q, a, as_of="2025-01-03", lookback=n,
                        input_adjustment="raw", actions_complete=True)
