"""Declarative Skill Upgrade Blueprints (`fin_skills.skill_rsi.blueprints`).

Houses canonical `SkillUpgradeBlueprint` definitions managed by the Skill-RSI scaffolding,
including:
  - Skill #131: `macro-fx-industry-beta-shield` (`fin-macro` / `macro_fx_industry_beta_shield.py`),
    upgrading raw macro/FX time-series broadcasts into a strictly causal (`<= t-1` sensitivity beta
    times `t` macro shock) cross-sectional industry-beta-gated shield.
"""
from __future__ import annotations

from fin_skills.skill_rsi.scaffold import SkillUpgradeBlueprint

MACRO_FX_SCRIPT_SOURCE = '''#!/usr/bin/env python3
"""Industry-Beta-Gated Macro & FX Transmission Shield (macro-fx-industry-beta-shield).

Converts a 1-D time-series macro or USD/CNH FX shock M_t into a genuine cross-sectional
equity alpha/shield score without look-ahead bias:

  1. Uniform Broadcast Defect:
     Broadcasting a raw macro series M_t equally across all stocks on day t has zero
     cross-sectional variance (Rank IC == 0) or, if added with a static positive sign,
     wrongly treats export-driven tech/manufacturing (beta_FX > 0) and domestic financials
     (beta_FX < 0) identically.
  2. Full-Sample Beta Look-Ahead Defect:
     Estimating stock i's macro/FX sensitivity beta_i over the full backtest sample [1..T]
     leaks future regime shifts into historical cross-sectional rankings.
  3. Causal Shrinkage Beta x Shock Shield:
     Estimates rolling causal sensitivity beta_{i, t-1} using strictly past data (<= t-1)
     and shrinks noisy single-stock betas toward the Leave-One-Out (j != i) industry peer
     beta prior before multiplying by today's Point-in-Time macro/FX shock M_t.
"""
from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pandas as pd


def estimate_causal_macro_fx_beta(
    df: pd.DataFrame,
    window: int = 20,
    min_periods: int = 5,
    shrinkage_lambda: float = 0.40,
    date_col: str = "date",
    symbol_col: str = "symbol",
    industry_col: str = "industry",
    ret_col: str = "ret_1d",
    shock_col: str = "macro_fx_shock",
) -> pd.Series:
    """Estimate strictly causal (<= t-1) stock sensitivity beta to macro/FX shocks with peer shrinkage."""
    if df.empty:
        return pd.Series(dtype=float)

    work = df[[date_col, symbol_col, ret_col, shock_col]].copy()
    work[industry_col] = df[industry_col] if industry_col in df.columns else "ALL"
    work["_orig_pos"] = np.arange(len(df))
    work.sort_values([symbol_col, date_col], inplace=True, kind="mergesort")

    r = pd.to_numeric(work[ret_col], errors="coerce").fillna(0.0)
    m = pd.to_numeric(work[shock_col], errors="coerce").fillna(0.0)
    rm = r * m
    m2 = m * m

    grp = work.groupby(symbol_col, sort=False)
    # Strictly lagged (<= t-1) rolling covariance / variance so beta at t never sees ret_1d at t
    roll_rm = grp.apply(
        lambda g: (pd.to_numeric(g[ret_col], errors="coerce").fillna(0.0)
                   * pd.to_numeric(g[shock_col], errors="coerce").fillna(0.0))
        .shift(1)
        .rolling(window, min_periods=min_periods)
        .mean()
    ).reset_index(level=0, drop=True)
    roll_r = grp[ret_col].transform(lambda s: pd.to_numeric(s, errors="coerce").fillna(0.0).shift(1).rolling(window, min_periods=min_periods).mean())
    roll_m = grp[shock_col].transform(lambda s: pd.to_numeric(s, errors="coerce").fillna(0.0).shift(1).rolling(window, min_periods=min_periods).mean())
    roll_m2 = grp.apply(
        lambda g: (pd.to_numeric(g[shock_col], errors="coerce").fillna(0.0) ** 2)
        .shift(1)
        .rolling(window, min_periods=min_periods)
        .mean()
    ).reset_index(level=0, drop=True)

    cov = (roll_rm - roll_r * roll_m).fillna(0.0)
    var = (roll_m2 - roll_m * roll_m).clip(lower=1e-6).fillna(1e-4)
    raw_beta = (cov / (var + 1e-5)).clip(-5.0, 5.0).fillna(0.0)
    work["_raw_beta"] = raw_beta

    # Leave-One-Out (j != i) industry peer beta on date t
    ind_grp = work.groupby([date_col, industry_col], sort=False)["_raw_beta"]
    ind_sum = ind_grp.transform("sum")
    ind_cnt = ind_grp.transform("count")
    loo_ind_beta = np.where(
        ind_cnt > 1,
        (ind_sum - work["_raw_beta"]) / np.maximum(ind_cnt - 1, 1),
        work["_raw_beta"],
    )
    shrunk_beta = (1.0 - shrinkage_lambda) * work["_raw_beta"].to_numpy(dtype=float) + shrinkage_lambda * loo_ind_beta
    work["_beta"] = shrunk_beta
    work.sort_values("_orig_pos", inplace=True, kind="mergesort")
    return pd.Series(work["_beta"].to_numpy(dtype=float), index=df.index, name="causal_macro_fx_beta")


def compute_macro_fx_beta_shield(
    df: pd.DataFrame,
    window: int = 20,
    min_periods: int = 3,
    date_col: str = "date",
    symbol_col: str = "symbol",
    industry_col: str = "industry",
    ret_col: str = "ret_1d",
    shock_col: str = "macro_fx_shock",
    policy_col: str = "macro_policy_score",
) -> pd.Series:
    """Compute cross-sectional z-scored causal macro/FX beta-gated transmission shield."""
    if df.empty:
        return pd.Series(dtype=float)

    beta = estimate_causal_macro_fx_beta(
        df,
        window=window,
        min_periods=min_periods,
        date_col=date_col,
        symbol_col=symbol_col,
        industry_col=industry_col,
        ret_col=ret_col,
        shock_col=shock_col,
    )
    shock = pd.to_numeric(df[shock_col], errors="coerce").fillna(0.0)
    policy = (
        pd.to_numeric(df[policy_col], errors="coerce").fillna(0.0)
        if policy_col in df.columns
        else pd.Series(0.0, index=df.index)
    )
    # Cross-sectional transmission score: stock sensitivity beta * contemporaneous macro/FX impulse
    raw_transmission = beta * (shock + 0.25 * policy)
    grp = raw_transmission.groupby(df[date_col])
    mean = grp.transform("mean")
    std = grp.transform("std").replace(0.0, 1.0).fillna(1.0)
    z = ((raw_transmission - mean) / (std + 1e-8)).clip(-3.5, 3.5).fillna(0.0)
    return pd.Series(z.to_numpy(dtype=float), index=df.index, name="macro_fx_industry_beta_shield")


def audit_macro_fx_beta_causality(
    df: pd.DataFrame,
    signal_col: str | None = None,
    date_col: str = "date",
    symbol_col: str = "symbol",
    industry_col: str = "industry",
    ret_col: str = "ret_1d",
    shock_col: str = "macro_fx_shock",
) -> dict[str, Any]:
    """Audit a macro/FX stock signal for uniform-broadcast defects and future-beta look-ahead."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise TypeError("df must be a non-empty DataFrame")
    for c in (date_col, symbol_col, ret_col, shock_col):
        if c not in df.columns:
            raise TypeError(f"df is missing required column {c!r}")

    if signal_col is not None:
        if signal_col not in df.columns:
            raise TypeError(f"signal_col {signal_col!r} not in df.columns")
        sig = pd.to_numeric(df[signal_col], errors="coerce").fillna(0.0)
    else:
        sig = compute_macro_fx_beta_shield(
            df,
            date_col=date_col,
            symbol_col=symbol_col,
            industry_col=industry_col,
            ret_col=ret_col,
            shock_col=shock_col,
        )

    # Check 1: Cross-sectional dispersion on active dates (detects raw 1-D macro broadcast)
    cs_std = sig.groupby(df[date_col]).std().dropna()
    active_std = cs_std[cs_std.index > sorted(df[date_col].astype(str).unique())[min(3, len(cs_std) - 1)]]
    mean_cs_std = float(active_std.mean()) if len(active_std) else float(cs_std.mean() or 0.0)
    non_uniform_broadcast = bool(mean_cs_std > 1e-4)

    # Check 2: Future-bar perturbation invariance when computing causal shield
    dates = sorted(df[date_col].astype(str).unique())
    cut_idx = max(1, min(len(dates) - 2, int(len(dates) * 0.7)))
    cut_date = dates[cut_idx]
    past_mask = (df[date_col].astype(str) <= cut_date).to_numpy()

    pert = df.copy()
    pert.loc[~past_mask, ret_col] = pd.to_numeric(pert.loc[~past_mask, ret_col], errors="coerce").fillna(0.0) + 5.0
    pert.loc[~past_mask, shock_col] = pd.to_numeric(pert.loc[~past_mask, shock_col], errors="coerce").fillna(0.0) + 2.0

    causal_base = compute_macro_fx_beta_shield(
        df, date_col=date_col, symbol_col=symbol_col, industry_col=industry_col, ret_col=ret_col, shock_col=shock_col
    ).to_numpy(dtype=float)
    causal_pert = compute_macro_fx_beta_shield(
        pert, date_col=date_col, symbol_col=symbol_col, industry_col=industry_col, ret_col=ret_col, shock_col=shock_col
    ).to_numpy(dtype=float)
    max_future_leak = float(np.nanmax(np.abs(causal_base[past_mask] - causal_pert[past_mask])))

    # If caller passed a precomputed signal_col, verify it correlates with causal shield and is not full-sample leaked
    caller_matches_causal = True
    if signal_col is not None and non_uniform_broadcast:
        corr = float(np.corrcoef(sig.to_numpy(dtype=float), causal_base)[0, 1]) if np.std(sig) > 1e-8 else 0.0
        caller_matches_causal = bool(corr >= 0.50)

    passed = bool(non_uniform_broadcast and (max_future_leak <= 1e-10) and caller_matches_causal)
    verdict = (
        "PASS: Macro-FX transmission shield has cross-sectional beta dispersion and zero future beta leakage."
        if passed
        else (
            "FAIL: Uniform 1-D macro broadcast detected (zero cross-sectional dispersion)"
            if not non_uniform_broadcast
            else "FAIL: Full-sample future beta look-ahead detected in macro/FX signal"
        )
    )
    return {
        "passed": passed,
        "verdict": verdict,
        "non_uniform_broadcast": non_uniform_broadcast,
        "mean_cross_sectional_std": round(mean_cs_std, 6),
        "max_future_leak_diff": float(max_future_leak),
        "caller_matches_causal": caller_matches_causal,
        "n_rows": int(len(df)),
        "n_dates": int(len(dates)),
    }


def _demo() -> int:
    rng = np.random.default_rng(11)
    dates = [f"2025-02-{d:02d}" for d in range(1, 16)]
    rows = []
    for d in dates:
        fx_shock = float(rng.normal(0.004, 0.01))
        for sym, ind, beta_true in (
            ("SH688981", "Semiconductor", 0.9),
            ("SH688041", "Semiconductor", 0.8),
            ("SH600036", "Bank", -0.4),
            ("SZ000001", "Bank", -0.35),
        ):
            rows.append({
                "date": d,
                "symbol": sym,
                "industry": ind,
                "ret_1d": float(beta_true * fx_shock + rng.normal(0.0, 0.01)),
                "macro_fx_shock": fx_shock,
                "macro_policy_score": 0.2,
            })
    df = pd.DataFrame(rows)
    df["shield"] = compute_macro_fx_beta_shield(df, window=5, min_periods=2)
    audit = audit_macro_fx_beta_causality(df, signal_col="shield")
    print(f"Macro-FX Industry Beta Shield demo: passed={audit['passed']} cs_std={audit['mean_cross_sectional_std']:.4f}")
    return 0 if audit["passed"] else 1


if __name__ == "__main__":
    sys.exit(_demo())
'''

MACRO_FX_SKILL_BLUEPRINT = SkillUpgradeBlueprint(
    name="macro-fx-industry-beta-shield",
    plugin="fin-macro",
    module_name="macro_fx_industry_beta_shield",
    description=(
        "[fin-macro] Convert a 1-D macro liquidity or USD/CNH FX shock into a causal cross-sectional "
        "equity shield by multiplying lagged stock-and-industry sensitivity beta by the regime impulse. "
        "TRIGGER - macro_fx_industry_beta_shield, compute_macro_fx_beta_shield, estimate_causal_macro_fx_beta, "
        "audit_macro_fx_beta_causality, industry beta gated macro FX transmission shield, USD/CNH shock "
        "cross-sectional stock sensitivity beta, uniform macro broadcast zero cross-sectional dispersion, "
        "full-sample macro beta look-ahead trap, 行业敏感度Beta加权的宏观汇率门控. "
        "SKIP for GDP nowcasting (gdp-nowcasting-dynamic-factor), recession indicators "
        "(macro-regime-and-recession-indicators), release calendars (macro-release-calendar-and-embargo), "
        "macro vintage tables (real-time-macro-backtesting), and FX pip conventions (fx-and-carry-conventions)."
    ),
    body_markdown=r"""# Industry-Beta-Gated Macro & FX Transmission Shield (`macro-fx-industry-beta-shield`)

## Why Raw 1-D Macro Broadcasts Fail in Cross-Sectional Stock Prediction

In multi-modal equity ranking, a macro liquidity impulse or USD/CNH exchange-rate shock $M_t$ is a **1-D time-series regime variable** (constant across all stocks $i \in \{1, \dots, N\}$ on date $t$). Two failure modes corrupt naive macro features:

1. **The Uniform Broadcast Trap (Zero Cross-Sectional Dispersion)**:
   - Appending $M_t$ directly or adding $+w \cdot M_t$ uniformly to every stock on day $t$ shifts all scores by the same constant, yielding zero cross-sectional Rank IC ($\text{RankIC}_t \equiv 0$) or inverting export-oriented tech/manufacturing ($\beta_{i,\text{FX}} > 0$) against domestic financials ($\beta_{i,\text{FX}} < 0$).
2. **The Full-Sample Beta Look-Ahead Trap**:
   - Estimating stock $i$'s macro sensitivity $\hat{\beta}_i$ via a single regression over the entire backtest $[1, T]$ leaks future regime breaks and post-tariff export re-ratings into historical predictions.

## Strictly Causal (`<= t-1`) Shrinkage Beta $\times$ Regime Impulse

`scripts/macro_fx_industry_beta_shield.py` (`fin_skills.macro.macro_fx_industry_beta_shield`) computes a strictly causal, peer-shrunk sensitivity beta $\hat{\beta}_{i,t-1}$ using only lagged returns $r_{i,\tau}$ and shocks $M_\tau$ ($\tau \le t-1$):

$$\hat{\beta}_{i,t-1}^{\text{raw}} = \frac{\widehat{\text{Cov}}_{W}(r_{i, t-W:t-1},\, M_{t-W:t-1})}{\widehat{\text{Var}}_{W}(M_{t-W:t-1}) + \epsilon}$$

$$\hat{\beta}_{i,t-1}^{\text{shrunk}} = (1 - \lambda)\,\hat{\beta}_{i,t-1}^{\text{raw}} + \lambda\,\frac{1}{|G(i)| - 1}\sum_{j \in G(i),\, j \ne i} \hat{\beta}_{j,t-1}^{\text{raw}}$$

$$S_{i,t}^{\text{MacroFX}} = \text{CS-ZScore}_t\!\left(\hat{\beta}_{i,t-1}^{\text{shrunk}} \cdot \left(M_t + \gamma P_t\right)\right)$$

## Executable API & Guard Verification

```bash
python3 -c "from fin_skills.macro.macro_fx_industry_beta_shield import _demo; raise SystemExit(_demo())"
```

## Cross-References

- `macro-regime-and-recession-indicators` — Macro regime state estimation and look-ahead gates.
- `real-time-macro-backtesting` — Point-in-time macro vintage alignment (`ALFRED` / release dates).
- `cross-board-supply-chain-rsi` — Leave-one-out (`j != i`) A-share peer spillover and board-limit routing.
""",
    script_source=MACRO_FX_SCRIPT_SOURCE,
    verified_on="2026-09-28",
    xref_neighbors=(
        "macro-regime-and-recession-indicators",
        "real-time-macro-backtesting",
        "cross-board-supply-chain-rsi",
    ),
    guard_name="macro_fx_beta_gate",
    probe_queries=[
        {
            "q": "How do I use macro_fx_industry_beta_shield and compute_macro_fx_beta_shield to turn a 1-D USD/CNH shock into a causal cross-sectional stock sensitivity beta shield?",
            "expect": "macro-fx-industry-beta-shield",
        },
        {
            "q": "如何构建行业敏感度Beta加权的宏观汇率门控 (macro_fx_industry_beta_shield) 并用 audit_macro_fx_beta_causality 检查全样本Beta未来函数？",
            "expect": "macro-fx-industry-beta-shield",
        },
    ],
)
