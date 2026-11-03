"""Guard: one-bar execution lag of a position function (signal-construction / assert_causal.py)."""
from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from fin_skills.api.base import Guard, Outcome, register
from fin_skills.api.guards._common import require_frame, require_int
from fin_skills.core.assert_causal import assert_lagged


@register
class ExecutionLagGuard(Guard):
    """Perturb only rows >= k and assert that no position up to row k + lag - 1 moved.

    Inputs
        fn   : callable df -> POSITIONS (Series/DataFrame indexed like df). The position
               on row t is held over the return from row t-1 to row t.
        df   : the input frame (numeric columns are the ones perturbed).
        k    : first perturbed row; must sit past any warm-up. Default len(df) // 2.
        lag  : bars between the last row a position may read and the row it trades on.
               Default 1 (decide on the previous close, earn the next return).
        tol  : absolute tolerance on a position moving. Default 1e-9.
        name : label for fn in messages.

    Fails when a position at or before row k + lag - 1 changed when only rows >= k were
    perturbed. This is a different property from assert_causal, which lets a value at
    row t use row t: a rule that trades on the same close it reads passes assert_causal
    and fails here. A function that raises on the perturbed frame is reported as a
    warning, not as a lag violation.
    """

    name = "execution_lag"
    skill = "signal-construction"
    summary = ("Detects same-bar execution: perturb rows >= k, assert positions through "
               "row k + lag - 1 are unchanged.")
    wraps = ("fin_skills.core.assert_causal.assert_lagged",)
    required = ("fn", "df")
    optional = ("k", "lag", "tol", "name")

    def check(self, fn: Callable[[pd.DataFrame], Any], df: pd.DataFrame,
              k: int | None = None, lag: int = 1, tol: float = 1e-9,
              name: str = "") -> Outcome:
        out = Outcome()
        df = require_frame(df, "df")
        if not callable(fn):
            raise TypeError("fn must be callable")
        if len(df) < 4:
            raise TypeError("df needs at least 4 rows to split into before/after k")
        k = len(df) // 2 if k is None else require_int(k, "k", minimum=1)
        if k >= len(df):
            raise TypeError(f"k={k} must be < len(df)={len(df)}")
        lag = require_int(lag, "lag", minimum=1)
        label = name or getattr(fn, "__name__", "fn")
        out.note(k=k, lag=lag, n_rows=len(df), tol=tol, checked_through_row=min(len(df), k + lag) - 1)
        try:
            assert_lagged(fn, df, k=k, lag=lag, tol=tol, name=label)
        except AssertionError as exc:
            out.error(str(exc), where=label)
        except (TypeError, ValueError, KeyError, IndexError, ArithmeticError) as exc:
            out.warning(f"position function raised, not a lag result: "
                        f"{type(exc).__name__}: {exc}", where=label)
        else:
            out.info(f"lagged: no position through row {min(len(df), k + lag) - 1} depends on "
                     f"rows >= {k}", where=label)
        return out
