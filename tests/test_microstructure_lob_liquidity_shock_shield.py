"""Independent numerical and temporal checks for the daily liquidity proxy."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fin_skills.microstructure.lob_liquidity_shock_shield import (
    audit_lob_liquidity_causality, compute_lob_liquidity_shock_shield,
)


def make_panel():
    rng = np.random.default_rng(29)
    return pd.DataFrame([
        {"date": d, "symbol": s, "garman_klass_volatility": rng.uniform(.01, .05),
         "ret_1d": rng.normal(0, .02), "volume": rng.uniform(1e6, 5e7),
         "margin_balance_z30": rng.normal(), "amihud_illiquidity": rng.uniform(.001, .004)}
        for d in pd.date_range("2025-01-01", periods=12) for s in ("A", "B", "C")
    ])


def test_matches_small_independent_history_calculation():
    panel = make_panel()
    expected = []
    for date in panel.date.unique():
        rows = []
        for symbol in ("A", "B", "C"):
            history = panel[(panel.symbol == symbol) & (panel.date <= date)].tail(5)
            if len(history) < 3:
                rows.append([np.nan] * 6)
            else:
                rows.append([np.std(history.ret_1d, ddof=1),
                             np.std(history.garman_klass_volatility, ddof=1),
                             np.mean(history.garman_klass_volatility),
                             np.std(np.log1p(history.volume), ddof=1),
                             np.std(history.amihud_illiquidity, ddof=1),
                             history.margin_balance_z30.iloc[-1]])
        if np.isnan(rows).all():
            expected.extend([np.nan] * 3)
        else:
            from scipy.stats import rankdata
            ranks = rankdata(rows, axis=0) / 3 - .5
            raw = ranks @ np.array([-.32, -.24, -.20, -.10, -.06, -.08])
            scale = np.std(raw, ddof=1) or 1.
            expected.extend(np.clip((raw - raw.mean()) / (scale + 1e-8), -3.5, 3.5))
    actual = compute_lob_liquidity_shock_shield(panel, window=5)
    np.testing.assert_allclose(actual, expected, atol=1e-12, equal_nan=True)


def test_every_prefix_matches_full_panel_and_input_is_unchanged():
    panel = make_panel()
    saved = panel.copy(deep=True)
    full = compute_lob_liquidity_shock_shield(panel)
    for date in panel.date.unique():
        prefix = panel[panel.date <= date]
        pd.testing.assert_series_equal(compute_lob_liquidity_shock_shield(prefix),
                                       full.loc[prefix.index])
    pd.testing.assert_frame_equal(panel, saved)


def test_shuffled_rows_categorical_symbols_and_duplicate_index_preserve_alignment():
    panel = make_panel()
    full = compute_lob_liquidity_shock_shield(panel)
    shuffled = panel.sample(frac=1, random_state=7)
    expected = full.loc[shuffled.index].to_numpy()
    shuffled.index = [1] * len(shuffled)
    shuffled.symbol = shuffled.symbol.astype("category")
    actual = compute_lob_liquidity_shock_shield(shuffled)
    np.testing.assert_allclose(actual, expected, atol=0, rtol=0, equal_nan=True)
    assert actual.index.tolist() == shuffled.index.tolist()


def test_future_contaminated_precomputed_signal_is_rejected():
    panel = make_panel()
    panel["score"] = compute_lob_liquidity_shock_shield(panel)
    assert audit_lob_liquidity_causality(panel, "score")["passed"]
    panel["score"] += .01 * panel.groupby("symbol").ret_1d.shift(-1).fillna(0)
    audit = audit_lob_liquidity_causality(panel, "score")
    assert not audit["passed"] and not audit["caller_matches_causal"]


@pytest.mark.parametrize("defect", ["missing_feature", "nan", "inf", "negative_volume",
                                    "missing_symbol", "bad_date", "duplicate_pair"])
def test_bad_observations_rejected(defect):
    panel = make_panel()
    if defect == "missing_feature": panel = panel.drop(columns="volume")
    elif defect == "nan": panel.loc[0, "ret_1d"] = np.nan
    elif defect == "inf": panel.loc[0, "margin_balance_z30"] = np.inf
    elif defect == "negative_volume": panel.loc[0, "volume"] = -1
    elif defect == "missing_symbol": panel.loc[0, "symbol"] = None
    elif defect == "bad_date": panel["date"] = "bad"
    else: panel = pd.concat([panel, panel.iloc[:1]])
    with pytest.raises(ValueError): compute_lob_liquidity_shock_shield(panel)


def test_custom_columns_windows_and_warmup_are_consistent():
    panel = make_panel().rename(columns={"date": "when", "symbol": "asset"})
    panel["score"] = compute_lob_liquidity_shock_shield(panel, 5, 2, "when", "asset")
    assert panel.score.isna().sum() == 3
    assert audit_lob_liquidity_causality(panel, "score", "when", "asset", 5, 2)["passed"]


def test_insufficient_history_and_constant_features_do_not_pass_audit():
    panel = make_panel()
    assert not audit_lob_liquidity_causality(panel.iloc[:3])["passed"]
    for col in panel.columns[2:]: panel[col] = 1.
    assert not audit_lob_liquidity_causality(panel)["passed"]
    values = compute_lob_liquidity_shock_shield(panel)
    assert values.dropna().eq(0).all()


@pytest.mark.parametrize("window,minimum", [(1, 1), (2, 3), (3., 2), (3, True)])
def test_bad_window_rejected(window, minimum):
    with pytest.raises(ValueError):
        compute_lob_liquidity_shock_shield(make_panel(), window, minimum)
