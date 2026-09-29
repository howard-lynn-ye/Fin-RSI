"""Unit tests for fin_skills.china.cross_board_supply_chain_rsi."""

from __future__ import annotations

import numpy as np

from fin_skills.china.cross_board_supply_chain_rsi import (
    audit_cross_board_spillover_causality,
    compute_cross_board_limit_bifurcation,
    compute_cross_board_supply_chain_features,
    compute_supply_chain_leader_spillover,
)
from fin_skills.skill_rsi import make_synthetic_ashare_panel


def test_cross_board_supply_chain_rsi_operators_and_causality() -> None:
    panel = make_synthetic_ashare_panel(n_days=20, n_stocks=12, seed=42)
    enr = compute_cross_board_supply_chain_features(panel)
    bif = compute_cross_board_limit_bifurcation(panel)
    sp = compute_supply_chain_leader_spillover(panel)
    assert len(bif) == len(panel)
    assert len(sp) == len(panel)
    assert "intraday_candle_asymmetry" in enr.columns
    assert np.all(np.isfinite(bif.to_numpy(dtype=np.float64)))
    assert np.all(np.isfinite(sp.to_numpy(dtype=np.float64)))
    assert np.all(np.isfinite(enr["intraday_candle_asymmetry"].to_numpy(dtype=np.float64)))

    audit = audit_cross_board_spillover_causality(panel)
    assert audit["passed"] is True
    assert audit["zero_self_leakage"] is True
