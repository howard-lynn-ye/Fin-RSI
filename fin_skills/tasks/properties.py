"""Properties a position task's text declares about its output, checked without the reference.

The three original checks (finite output, split representation, execution lag) accept any
program that is finite, split-invariant and non-anticipating. In the P-fix records that included
105 final programs that returned zeros everywhere and 55 whose positions were scaled by
1 / (number of assets) twice. Each check here tests one sentence of the task text on the
program's own output:

  takes_positions   the output is not zero on every row (every task panel makes its rule trade)
  position_scale    "the weight is that value divided by the number of assets": a long/short/flat
                    rule's weights are -1/N, 0 or +1/N; a sized rule stays within its cap; portfolio
                    weights are non-negative and sum to at most one
  warmup_rows_zero  "rows where it is not yet defined are 0": no position before the first row
                    the rule's own lookback allows

They are opt-in (Judge(properties=True)); the frozen study ran without them.
"""
from __future__ import annotations

import numpy as np

CHECKS = ("takes_positions", "position_scale", "warmup_rows_zero")
# What a failure usually means. Check feedback shows the names only; the rule cards and
# fin_skills.api.rules.CAUSES (the frozen study's explanations) are unchanged.
CAUSES = {
    "takes_positions": "the output is zero on every row, so the rule never took a position: "
                       "its condition is never met, or its result is lost before it is returned",
    "position_scale": "a weight is not a value the rule allows after scaling (for a long, short "
                      "or flat rule: -1/N, 0 or +1/N), often a result that was divided by the "
                      "number of assets twice or not at all",
    "warmup_rows_zero": "a position is set before the rule's own lookback is complete; those "
                        "rows must be 0",
}
STATE_RULES = frozenset({"ma_crossover", "momentum", "macd", "bollinger_reversion",
                         "rsi_reversion", "donchian_breakout"})   # value at a close is -1, 0 or +1
TOLERANCE = 1e-6


def first_position_row(task) -> int | None:
    """The first row that may hold a position: the task's lookback read as its text states it,
    plus the one-bar lag. None when the task declares no single lookback (no warm-up check)."""
    p = task.parameters
    rows = {
        "bollinger_reversion": lambda: p["lookback"],            # z from row lookback-1
        "rsi_reversion": lambda: p["lookback"] + 1,              # lookback changes: row lookback
        "donchian_breakout": lambda: p["lookback"] + 1,          # lookback closes before row t
        "ma_crossover": lambda: p["slow"],                       # slow closes: row slow-1
        "momentum": lambda: p["lookback"] + 1,                   # lookback returns: row lookback
        "vol_target_momentum": lambda: p["lookback"] + 1,
        "macd": lambda: p["slow"] + p["signal_span"] - 1,        # line from slow-1, then its EMA
    }.get(task.algorithm)
    try:
        return int(rows()) if rows else None
    except KeyError:
        return None


def checks(task, values) -> dict:
    """The declared-property checks for one program output (rows x assets)."""
    if task.kind != "positions":
        return {}
    v = np.nan_to_num(np.asarray(values, dtype=float))
    n = v.shape[1]
    out = {"takes_positions": bool((np.abs(v) > TOLERANCE / n).any())}
    if task.family == "signal":
        x = np.abs(v) * n
        if task.algorithm in STATE_RULES:
            out["position_scale"] = bool(((x < TOLERANCE) | (np.abs(x - 1) < TOLERANCE)).all())
        else:
            cap = float(task.parameters.get("max_exposure", 1.0))
            out["position_scale"] = bool((x <= cap + TOLERANCE).all())
    else:
        out["position_scale"] = bool((v >= -TOLERANCE).all()
                                     and (v.sum(axis=1) <= 1 + TOLERANCE).all())
    first = first_position_row(task)
    if first is not None:
        out["warmup_rows_zero"] = bool((np.abs(v[:first]) < TOLERANCE / n).all())
    return out
