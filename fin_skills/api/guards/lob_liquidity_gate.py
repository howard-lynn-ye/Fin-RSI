"""Guard for the built-in daily liquidity proxy and matching supplied scores."""
from __future__ import annotations

import pandas as pd

from fin_skills.api.base import Guard, GuardResult, Outcome, get, register


@register
class LobLiquidityGateGuard(Guard):
    """Probe the built-in proxy; compare supplied scores without a correlation shortcut.

    Inputs
        panel : observed features keyed by date and symbol.
        signal_col : optional column containing this formula's scores, including warmup NaNs.
        date_col : timestamp column; features must already be available at that time.
        symbol_col : asset identifier column.
        window : rolling observation count.
        min_periods : observations required before emitting a score.
    """
    name = "lob_liquidity_gate"
    skill = "lob-liquidity-shock-shield"
    summary = "Checks the built-in liquidity proxy and rejects mismatching supplied scores."
    wraps = ("fin_skills.microstructure.lob_liquidity_shock_shield.audit_lob_liquidity_causality",)
    required = ("panel",)
    optional = ("signal_col", "date_col", "symbol_col", "window", "min_periods")

    def check(self, panel: pd.DataFrame, signal_col: str | None = None,
              date_col: str = "date", symbol_col: str = "symbol", window: int = 20,
              min_periods: int = 3) -> Outcome:
        from fin_skills.microstructure.lob_liquidity_shock_shield import audit_lob_liquidity_causality
        audit = audit_lob_liquidity_causality(
            panel, signal_col, date_col, symbol_col, window, min_periods,
        )
        out = Outcome()
        out.note(**audit)
        (out.info if audit["passed"] else out.error)(audit["verdict"], where=self.name)
        return out


def check_lob_liquidity_gate(panel: pd.DataFrame, signal_col: str | None = None,
                            date_col: str = "date", symbol_col: str = "symbol",
                            window: int = 20, min_periods: int = 3) -> GuardResult:
    return get("lob_liquidity_gate").run(
        panel=panel, signal_col=signal_col, date_col=date_col, symbol_col=symbol_col,
        window=window, min_periods=min_periods,
    )
