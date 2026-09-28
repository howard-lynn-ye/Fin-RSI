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
                rows.append({"symbol": t, "date": f"{y}-06-{d:02d}"})
    df = pd.DataFrame(rows)
    res = audit_panel_balance(df)
    assert res.passed is True
    assert res.n_entities == 10
    assert res.n_years == 3
    assert res.company_gini < 0.15


def test_audit_panel_balance_fails_on_skewed_panel():
    # 1 ticker has 90 rows in 1 year; 2 tickers have 1 row
    rows = [{"symbol": "MEGA", "date": "2024-06-01"} for _ in range(90)]
    rows.append({"symbol": "TINY1", "date": "2020-01-01"})
    rows.append({"symbol": "TINY2", "date": "2021-01-01"})
    df = pd.DataFrame(rows)
    res = audit_panel_balance(df)
    assert res.passed is False
    assert "IMBALANCED PANEL" in res.verdict
    assert len(res.notes) >= 1


def test_rebalance_company_year_panel_caps_cells_and_attaches_weights():
    rows = []
    for i in range(5):
        for y in [2023, 2024]:
            for d in range(20):
                rows.append({"symbol": f"GOOD{i}", "date": f"{y}-03-{(d % 28) + 1:02d}", "val": d})
    df = pd.DataFrame(rows)

    rebalanced = rebalance_company_year_panel(
        df,
        entity_col="symbol",
        time_col="date",
        max_per_cell=5,
    )
    assert len(rebalanced) == 5 * 2 * 5
    assert "normalized_year_balanced_weight" in rebalanced.columns
    res = audit_panel_balance(rebalanced)
    assert res.passed is True
