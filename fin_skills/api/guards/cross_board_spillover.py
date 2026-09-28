"""Guard: A-share 10%/20% cross-board sign bifurcation and leave-one-out peer spillover (cross-board-supply-chain-rsi)."""
from __future__ import annotations

import pandas as pd

from fin_skills.api.base import Guard, GuardResult, Outcome, get, register
from fin_skills.china.cross_board_supply_chain_rsi import (
    audit_cross_board_spillover_causality,
    compute_cross_board_limit_bifurcation,
    compute_supply_chain_leader_spillover,
)


@register
class CrossBoardSpilloverGuard(Guard):
    """Audit an A-share panel for leave-one-out (j != i) supply-chain spillover causality and 10%/20% board sign bifurcation.

    Inputs
        panel                : DataFrame of A-share stock-day rows with symbol and date columns.
        symbol_col           : column identifying the stock ticker (default 'symbol').
        date_col             : column identifying the trading date (default 'date').
        include_self_in_peer : if True, flags self-inclusion (A_ii != 0) in peer spillover aggregation.
    """

    name = "cross_board_spillover"
    skill = "cross-board-supply-chain-rsi"
    summary = "Audits leave-one-out (j != i) supply-chain peer spillover causality and 10%-Main vs 20%-STAR/ChiNext board sign bifurcation."
    wraps = (
        "fin_skills.china.cross_board_supply_chain_rsi.audit_cross_board_spillover_causality",
        "fin_skills.china.cross_board_supply_chain_rsi.compute_cross_board_limit_bifurcation",
        "fin_skills.china.cross_board_supply_chain_rsi.compute_supply_chain_leader_spillover",
    )
    required = ("panel",)
    optional = ("symbol_col", "date_col", "include_self_in_peer")

    def check(
        self,
        panel: pd.DataFrame,
        symbol_col: str = "symbol",
        date_col: str = "date",
        include_self_in_peer: bool = False,
    ) -> Outcome:
        out = Outcome()
        audit = audit_cross_board_spillover_causality(
            df=panel,
            symbol_col=symbol_col,
            date_col=date_col,
            include_self_in_peer=include_self_in_peer,
        )
        out.note(**audit)
        if not audit["passed"]:
            out.error(audit["verdict"], where="cross_board_spillover")
        else:
            out.info(audit["verdict"], where="cross_board_spillover")
        return out


def check_cross_board_spillover(
    panel: pd.DataFrame,
    symbol_col: str = "symbol",
    date_col: str = "date",
    include_self_in_peer: bool = False,
) -> GuardResult:
    """Run the `cross_board_spillover` executable guard directly on a DataFrame."""
    return get("cross_board_spillover").run(
        panel=panel,
        symbol_col=symbol_col,
        date_col=date_col,
        include_self_in_peer=include_self_in_peer,
    )
