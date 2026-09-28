"""Tests for fin_skills.core.panel_balance (panel balance guard & rebalancer)."""

from __future__ import annotations

import pandas as pd

from fin_skills.core.panel_balance import (
    audit_panel_balance,
    compute_gini,
    rebalance_company_year_panel,
)


def test_compute_gini_uniform_and_skewed():
    # Uniform counts -> Gini = 0.0
    assert abs(compute_gini([10, 10, 10, 10])) < 1e-9
    # Heavily concentrated counts -> high Gini > 0.5
    assert compute_gini([1, 1, 1, 1, 100]) > 0.5


def test_audit_panel_balance_passes_on_balanced_panel():
    rows = []
    for t in [f"SYM{i:02d}" for i in range(10)]:
        for y in [2022, 2023, 2024]:
            for d in range(1, 6):
                rows.append({"ticker": t, "timestamp": f"{y}-06-{d:02d}"})
    df = pd.DataFrame(rows)
    res = audit_panel_balance(df)
    assert res["passed"] is True
    assert res["n_companies"] == 10
    assert res["n_years"] == 3
    assert res["company_gini"] < 0.15


def test_audit_panel_balance_fails_on_skewed_panel():
    # 1 ticker has 90 rows in 1 year; 2 tickers have 1 row
    rows = [{"ticker": "MEGA", "timestamp": "2024-06-01"} for _ in range(90)]
    rows.append({"ticker": "TINY1", "timestamp": "2020-01-01"})
    rows.append({"ticker": "TINY2", "timestamp": "2021-01-01"})
    df = pd.DataFrame(rows)
    res = audit_panel_balance(df, min_companies=5)
    assert res["passed"] is False
    assert len(res["violations"]) >= 2


def test_rebalance_company_year_panel_caps_cells_and_drops_sparse():
    rows = []
    # 5 companies with 2 years and 20 rows per cell
    for i in range(5):
        for y in [2023, 2024]:
            for d in range(20):
                rows.append({"ticker": f"GOOD{i}", "timestamp": f"{y}-03-{(d % 28) + 1:02d}", "val": d})
    # 1 sparse company with only 1 year
    rows.append({"ticker": "SPARSE", "timestamp": "2024-01-02", "val": 1})
    df = pd.DataFrame(rows)

    rebalanced, report = rebalance_company_year_panel(
        df,
        max_per_company_year=5,
        min_years_per_company=2,
        min_obs_per_company=4,
        min_companies_per_year=3,
    )
    assert "SPARSE" not in set(rebalanced["ticker"])
    assert len(rebalanced) == 5 * 2 * 5
    assert report["after"]["passed"] is True
