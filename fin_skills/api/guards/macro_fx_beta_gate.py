"""Guard: Industry-beta-gated macro & FX transmission shield vs uniform broadcast and full-sample beta look-ahead (macro-fx-industry-beta-shield)."""
from __future__ import annotations

import pandas as pd

from fin_skills.api.base import Guard, GuardResult, Outcome, get, register
from fin_skills.macro.macro_fx_industry_beta_shield import (
    audit_macro_fx_beta_causality,
    compute_macro_fx_beta_shield,
)


@register
class MacroFxBetaGateGuard(Guard):
    """Audit a macro/FX stock signal for uniform 1-D broadcast (zero cross-sectional dispersion) and full-sample beta look-ahead.

    Inputs
        panel        : DataFrame of stock-day rows containing date, symbol, ret_1d, and macro_fx_shock.
        signal_col   : optional column name of a pre-computed macro/FX stock signal to audit.
        date_col     : column identifying the observation date (default 'date').
        symbol_col   : column identifying the stock ticker (default 'symbol').
        industry_col : column identifying the industry group (default 'industry').
        ret_col      : column identifying the 1-day stock return (default 'ret_1d').
        shock_col    : column identifying the macro/FX shock series (default 'macro_fx_shock').
    """

    name = "macro_fx_beta_gate"
    skill = "macro-fx-industry-beta-shield"
    summary = "Flags uniform 1-D macro broadcasts (zero cross-sectional dispersion) and full-sample macro sensitivity beta look-ahead."
    wraps = (
        "fin_skills.macro.macro_fx_industry_beta_shield.audit_macro_fx_beta_causality",
        "fin_skills.macro.macro_fx_industry_beta_shield.compute_macro_fx_beta_shield",
    )
    required = ("panel",)
    optional = ("signal_col", "date_col", "symbol_col", "industry_col", "ret_col", "shock_col")

    def check(
        self,
        panel: pd.DataFrame,
        signal_col: str | None = None,
        date_col: str = "date",
        symbol_col: str = "symbol",
        industry_col: str = "industry",
        ret_col: str = "ret_1d",
        shock_col: str = "macro_fx_shock",
    ) -> Outcome:
        out = Outcome()
        audit = audit_macro_fx_beta_causality(
            df=panel,
            signal_col=signal_col,
            date_col=date_col,
            symbol_col=symbol_col,
            industry_col=industry_col,
            ret_col=ret_col,
            shock_col=shock_col,
        )
        out.note(**audit)
        if not audit["passed"]:
            out.error(audit["verdict"], where="macro_fx_beta_gate")
        else:
            out.info(audit["verdict"], where="macro_fx_beta_gate")
        return out


def check_macro_fx_beta_gate(
    panel: pd.DataFrame,
    signal_col: str | None = None,
    date_col: str = "date",
    symbol_col: str = "symbol",
    industry_col: str = "industry",
    ret_col: str = "ret_1d",
    shock_col: str = "macro_fx_shock",
) -> GuardResult:
    """Run the `macro_fx_beta_gate` executable guard directly on a DataFrame."""
    return get("macro_fx_beta_gate").run(
        panel=panel,
        signal_col=signal_col,
        date_col=date_col,
        symbol_col=symbol_col,
        industry_col=industry_col,
        ret_col=ret_col,
        shock_col=shock_col,
    )
