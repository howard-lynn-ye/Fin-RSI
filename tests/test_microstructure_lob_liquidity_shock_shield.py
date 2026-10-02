"""Unit tests for fin_skills.microstructure.lob_liquidity_shock_shield."""

from __future__ import annotations

import numpy as np

from fin_skills.microstructure.lob_liquidity_shock_shield import (
    audit_lob_liquidity_causality,
    compute_lob_liquidity_shock_shield,
    estimate_intraday_lob_execution_cost,
)
from fin_skills.skill_rsi import make_synthetic_ashare_panel


def test_lob_liquidity_shock_shield_and_causality() -> None:
    panel = make_synthetic_ashare_panel(n_days=20, n_stocks=12, seed=42)
    shield = compute_lob_liquidity_shock_shield(panel)
    assert len(shield) == len(panel)
    assert np.all(np.isfinite(shield.to_numpy(dtype=np.float64)))

    audit = audit_lob_liquidity_causality(panel)
    assert audit["passed"] is True
    assert audit["max_future_leak_diff"] == 0.0

    cost = estimate_intraday_lob_execution_cost(
        symbol="SH600000",
        order_notional_rmb=150_000.0,
        adv_rmb=50_000_000.0,
        gk_volatility=0.025,
        amihud_illiquidity=0.0015,
        limit_proximity_ratio=0.25,
        queue_depth_fraction=0.45,
        is_short_leg=False,
    )
    assert cost["total_all_in_cost_bps"] > 0.0
