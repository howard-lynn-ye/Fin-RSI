"""Guard wrapper: A-share cross-board board/sign bookkeeping (cross-board-supply-chain-rsi); NOT a causality check."""
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
    """Report A-share board counts and fixed board-sign / finite-spillover flags; NOT a causality check.

    Not a causality check: zero_self_leakage is True whenever include_self_in_peer is False and the spillover is finite (leave-one-out construction is not tested), and the board-sign check compares the fixed symbol-prefix sign mapping with itself unless the caller supplies micro_reversal_sign. A-share symbols only; other symbols fall into one HK_US bucket.

    Inputs
        panel                : DataFrame of A-share stock-day rows with symbol and date columns.
        symbol_col           : column identifying the stock ticker (default 'symbol').
        date_col             : column identifying the trading date (default 'date').
        include_self_in_peer : if True, flags self-inclusion (A_ii != 0) in peer spillover aggregation.
    """

    name = "cross_board_spillover"
    skill = "cross-board-supply-chain-rsi"
    summary = ("Not a causality check: reports board counts and fixed sign/finite-spillover flags for an "
               "A-share panel; the leakage verdict follows include_self_in_peer, not the data.")
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
    """Run the `cross_board_spillover` guard on a DataFrame. Not a causality check (see the guard docstring)."""
    return get("cross_board_spillover").run(
        panel=panel,
        symbol_col=symbol_col,
        date_col=date_col,
        include_self_in_peer=include_self_in_peer,
    )
