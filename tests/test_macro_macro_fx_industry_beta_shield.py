"""Unit tests for fin_skills.macro.macro_fx_industry_beta_shield."""

from __future__ import annotations

import numpy as np

from fin_skills.macro.macro_fx_industry_beta_shield import (
    audit_macro_fx_beta_causality,
    compute_macro_fx_beta_shield,
)
from fin_skills.skill_rsi import make_synthetic_ashare_panel


def test_macro_fx_industry_beta_shield_and_causality() -> None:
    panel = make_synthetic_ashare_panel(n_days=20, n_stocks=12, seed=42)
    shield = compute_macro_fx_beta_shield(panel)
    assert len(shield) == len(panel)
    assert np.all(np.isfinite(shield.to_numpy(dtype=np.float64)))

    audit = audit_macro_fx_beta_causality(panel)
    assert audit["passed"] is True
    assert audit["max_future_leak_diff"] == 0.0
