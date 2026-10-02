"""Guard: 20-day LOB volatility-of-volatility liquidity shield causality and L2 execution cost plausibility (lob-liquidity-shock-shield)."""
from __future__ import annotations

import pandas as pd

from fin_skills.api.base import Guard, GuardResult, Outcome, get, register


@register
class LobLiquidityGateGuard(Guard):
    """Audit a 20-day LOB volatility-of-volatility liquidity shield for future look-ahead, dispersion collapse, and L2 execution cost schedule validity.

    Inputs
        panel      : DataFrame of stock-day rows containing date, symbol, garman_klass_volatility, and ret_1d.
        signal_col : optional column name of a pre-computed LOB liquidity shield to audit.
        date_col   : column identifying the observation date (default 'date').
        symbol_col : column identifying the stock ticker (default 'symbol').
    """

    name = "lob_liquidity_gate"
    skill = "lob-liquidity-shock-shield"
    summary = "Flags future volatility-of-volatility look-ahead, zero cross-sectional dispersion, and invalid L2 queue/borrow cost schedules."
    wraps = (
        "fin_skills.microstructure.lob_liquidity_shock_shield.audit_lob_liquidity_causality",
        "fin_skills.microstructure.lob_liquidity_shock_shield.compute_lob_liquidity_shock_shield",
    )
    required = ("panel",)
    optional = ("signal_col", "date_col", "symbol_col")

    def check(
        self,
        panel: pd.DataFrame,
        signal_col: str | None = None,
        date_col: str = "date",
        symbol_col: str = "symbol",
    ) -> Outcome:
        from fin_skills.microstructure.lob_liquidity_shock_shield import (
            audit_lob_liquidity_causality,
        )

        out = Outcome()
        audit = audit_lob_liquidity_causality(
            df=panel,
            signal_col=signal_col,
            date_col=date_col,
            symbol_col=symbol_col,
        )
        out.note(**audit)
        if not audit["passed"]:
            out.error(audit["verdict"], where="lob_liquidity_gate")
        else:
            out.info(audit["verdict"], where="lob_liquidity_gate")
        return out


def check_lob_liquidity_gate(
    panel: pd.DataFrame,
    signal_col: str | None = None,
    date_col: str = "date",
    symbol_col: str = "symbol",
) -> GuardResult:
    """Run the `lob_liquidity_gate` executable guard directly on a DataFrame."""
    return get("lob_liquidity_gate").run(
        panel=panel,
        signal_col=signal_col,
        date_col=date_col,
        symbol_col=symbol_col,
    )
