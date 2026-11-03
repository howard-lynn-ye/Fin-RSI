"""Synthetic raw-price panels with known splits, for graded position tasks.

Four assets with seeded Gaussian returns. Raw quotes are the continuous adjusted closes divided
by cumulative split factors, so a program that forgets or repeats the adjustment sees jumps.
With rows=96 and the default column order this reproduces codegen_utility.dataset(seed)
exactly, so the E4 tasks keep their original inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ASSETS = ("A", "B", "C", "D")
FACTOR_KINDS = ("cumulative", "ratio")
START = "2024-01-02"


def _cumulative(rows: int) -> np.ndarray:
    f = np.ones((rows, len(ASSETS)))
    f[45:, 0] = 2.0              # A: 2-for-1 on row 45
    f[70:, 2] = 3.0              # C: 3-for-1 on row 70
    if rows > 100:
        f[100:, 1] = 1.5         # B: 3-for-2 on row 100
    if rows > 110:
        f[110:, 3] = 0.5         # D: 1-for-2 reverse split on row 110
    return f


def panel(seed: int, *, rows: int = 128, factors: str = "cumulative",
          order: tuple[str, ...] = ASSETS) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(raw prices, split factors as given to the program, adjusted closes), date-indexed.

    factors="cumulative": factor[t] is the product of every split effective on or before t.
    factors="ratio": factor[t] is the ratio of a split effective on row t, 1.0 elsewhere.
    order permutes the columns; the asset names stay A-D.
    """
    if factors not in FACTOR_KINDS:
        raise ValueError(f"factors must be one of {FACTOR_KINDS}")
    if sorted(order) != sorted(ASSETS):
        raise ValueError(f"order must be a permutation of {ASSETS}")
    if rows < 60:
        raise ValueError("rows must be at least 60")
    rng = np.random.default_rng(seed)
    r = rng.normal([.001, -.0004, .0005, .0002], [.013, .009, .017, .011], (rows, 4))
    adjusted = 100 * np.cumprod(1 + r, axis=0)
    cumulative = _cumulative(rows)
    given = cumulative.copy()
    if factors == "ratio":
        given[1:] = cumulative[1:] / cumulative[:-1]
    idx = pd.bdate_range(START, periods=rows)
    frame = lambda x: pd.DataFrame(x, index=idx, columns=list(ASSETS))[list(order)]
    return frame(adjusted / cumulative), frame(given), frame(adjusted)


def sample_text(raw: pd.DataFrame, factors: pd.DataFrame, rows: int = 3) -> str:
    """A short, model-facing view of the input layout (not the data the program is graded on)."""
    return ("prices (first rows):\n" + raw.head(rows).round(4).to_string()
            + "\n\nsplit_factors (first rows):\n" + factors.head(rows).to_string())
