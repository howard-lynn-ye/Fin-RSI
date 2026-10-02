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


def _resolve_ret_series(df: pd.DataFrame, ret_col: str) -> pd.Series:
    if ret_col in df.columns:
        return pd.to_numeric(df[ret_col], errors="coerce").fillna(0.0)
    for alt in ("stock_excess_return_1d", "return_1d", "ret_1d"):
        if alt in df.columns:
            return pd.to_numeric(df[alt], errors="coerce").fillna(0.0)
    raise TypeError(f"df is missing return column {ret_col!r}")


def _resolve_shock_series(df: pd.DataFrame, shock_col: str) -> pd.Series:
    if shock_col in df.columns:
        return pd.to_numeric(df[shock_col], errors="coerce").fillna(0.0)
    if "dxy_return_1d" in df.columns:
        dxy = pd.to_numeric(df["dxy_return_1d"], errors="coerce").fillna(0.0)
        vix_r = (
            pd.to_numeric(df["vix_return_1d"], errors="coerce").fillna(0.0)
            if "vix_return_1d" in df.columns
            else 0.0
        )
        us10y = (
            pd.to_numeric(df["us10y_change_1d"], errors="coerce").fillna(0.0)
            if "us10y_change_1d" in df.columns
            else 0.0
        )
        impulse = -np.tanh(dxy * 40.0) - 0.50 * np.tanh(vix_r * 10.0) - 0.50 * np.tanh(us10y * 10.0)
        return pd.Series(impulse, index=df.index, dtype=float)
    raise TypeError(f"df is missing macro/FX shock column {shock_col!r}")


def _resolve_industry_col(df: pd.DataFrame, industry_col: str) -> pd.Series:
    if industry_col in df.columns:
        return df[industry_col]
    for alt in ("supply_chain_cluster", "sector", "board_name"):
        if alt in df.columns:
            return df[alt]
    return pd.Series("ALL", index=df.index)


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

    work = pd.DataFrame(
        {
            date_col: df[date_col],
            symbol_col: df[symbol_col],
            "_ind": _resolve_industry_col(df, industry_col),
            "_r": _resolve_ret_series(df, ret_col).to_numpy(dtype=np.float64),
            "_m": _resolve_shock_series(df, shock_col).to_numpy(dtype=np.float64),
            "_orig_pos": np.arange(len(df)),
        },
        index=df.index,
    )
    work.sort_values([symbol_col, date_col], inplace=True, kind="mergesort")

    grp_sym = work.groupby(symbol_col, sort=False)
    r_lag = grp_sym["_r"].shift(1)
    m_lag = grp_sym["_m"].shift(1)
    work["_r_lag"] = r_lag
    work["_m_lag"] = m_lag
    work["_rm_lag"] = r_lag * m_lag
    work["_m2_lag"] = m_lag * m_lag

    # Vectorized Cython rolling means over strictly lagged (<= t-1) quantities
    grp_roll = work.groupby(symbol_col, sort=False)
    roll_rm = grp_roll["_rm_lag"].rolling(window, min_periods=min_periods).mean().to_numpy(dtype=np.float64)
    roll_r = grp_roll["_r_lag"].rolling(window, min_periods=min_periods).mean().to_numpy(dtype=np.float64)
    roll_m = grp_roll["_m_lag"].rolling(window, min_periods=min_periods).mean().to_numpy(dtype=np.float64)
    roll_m2 = grp_roll["_m2_lag"].rolling(window, min_periods=min_periods).mean().to_numpy(dtype=np.float64)

    cov = np.nan_to_num(roll_rm - roll_r * roll_m, nan=0.0)
    var = np.clip(np.nan_to_num(roll_m2 - roll_m * roll_m, nan=1e-4), 1e-6, None)
    raw_beta = np.clip(cov / (var + 1e-5), -5.0, 5.0)
    work["_raw_beta"] = raw_beta

    # Leave-One-Out (j != i) industry peer beta on date t
    ind_grp = work.groupby([date_col, "_ind"], sort=False)["_raw_beta"]
    ind_sum = ind_grp.transform("sum").to_numpy(dtype=np.float64)
    ind_cnt = ind_grp.transform("count").to_numpy(dtype=np.float64)
    loo_ind_beta = np.where(
        ind_cnt > 1,
        (ind_sum - raw_beta) / np.maximum(ind_cnt - 1.0, 1.0),
        raw_beta,
    )
    shrunk_beta = (1.0 - shrinkage_lambda) * raw_beta + shrinkage_lambda * loo_ind_beta
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
    shock = _resolve_shock_series(df, shock_col)
    policy = (
        pd.to_numeric(df[policy_col], errors="coerce").fillna(0.0)
        if policy_col in df.columns
        else pd.Series(0.0, index=df.index)
    )
    raw_transmission = beta * (shock + 0.25 * policy)

    # When multimodal defensive quality & stress columns exist, combine beta*impulse with stress-gated resilience
    if "garp_valuation_quality" in df.columns and "vix_z30" in df.columns:
        vix_z = pd.to_numeric(df["vix_z30"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        dxy_abs = (
            np.abs(np.tanh(pd.to_numeric(df["dxy_return_1d"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64) * 40.0))
            if "dxy_return_1d" in df.columns
            else np.abs(shock.to_numpy(dtype=np.float64))
        )
        macro_stress = np.clip(0.50 * np.clip(vix_z, -1.5, 3.0) + 0.50 * dxy_abs + 0.50, 0.20, 2.50)
        garp = pd.to_numeric(df["garp_valuation_quality"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        smart = (
            pd.to_numeric(df["smart_vs_retail_divergence"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
            if "smart_vs_retail_divergence" in df.columns
            else np.zeros(len(df), dtype=np.float64)
        )
        beta_abs = pd.Series(np.abs(beta.to_numpy(dtype=np.float64)), index=df.index)
        b_grp = beta_abs.groupby(df[date_col])
        beta_abs_z = ((beta_abs - b_grp.transform("mean")) / (b_grp.transform("std").replace(0.0, 1.0).fillna(1.0) + 1e-8)).fillna(0.0).to_numpy(dtype=np.float64)
        raw_transmission = pd.Series(
            0.30 * raw_transmission.to_numpy(dtype=np.float64)
            + macro_stress * (0.45 * garp + 0.30 * smart - 0.25 * beta_abs_z),
            index=df.index,
        )

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
    for c in (date_col, symbol_col):
        if c not in df.columns:
            raise TypeError(f"df is missing required column {c!r}")
    _ = _resolve_ret_series(df, ret_col)
    _ = _resolve_shock_series(df, shock_col)

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
    r_col_actual = ret_col if ret_col in pert.columns else ("stock_excess_return_1d" if "stock_excess_return_1d" in pert.columns else "return_1d")
    s_col_actual = shock_col if shock_col in pert.columns else "dxy_return_1d"
    pert.loc[~past_mask, r_col_actual] = pd.to_numeric(pert.loc[~past_mask, r_col_actual], errors="coerce").fillna(0.0) + 5.0
    pert.loc[~past_mask, s_col_actual] = pd.to_numeric(pert.loc[~past_mask, s_col_actual], errors="coerce").fillna(0.0) + 2.0

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
            })
    df = pd.DataFrame(rows)
    df["macro_policy_score"] = np.cos(np.arange(len(df), dtype=float) * 0.25)
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

LOB_LIQUIDITY_SCRIPT_SOURCE = '''#!/usr/bin/env python3
"""20-Day LOB Volatility-of-Volatility Liquidity Shield & L2 Queue Execution Cost Engine.

Implements the 132nd FinSkill (`lob-liquidity-shock-shield` in `fin_skills.microstructure.lob_liquidity_shock_shield`):
  1. `compute_lob_liquidity_shock_shield(df, ...)`:
     Synthesizes a strictly causal (`<= t`) 20-day rolling order-book volatility-of-volatility
     (`std_20d(GK_t)`), return dispersion (`std_20d(r_t)`), log-volume shock dispersion
     (`std_20d(log(1+V_t))`), and Amihud illiquidity instability (`std_20d(Amihud_t)`) cross-sectional
     resilience shield (9th Point-in-Time channel `Ch8` in `TriAxisStockRSINet` Model 8).
  2. `estimate_intraday_lob_execution_cost(...)`:
     Computes L2 Limit-Order-Book (`LOB`) passive queue-position fill probability, adverse selection
     queue impact, square-root participation slippage, board price-limit lockout penalty (`+-10%` vs
     `+-20%`), and A-share short-selling (`融券`) tiered borrow-fee schedules.
  3. `audit_lob_liquidity_causality(df, ...)`:
     Executable guard verifying non-degenerate cross-sectional dispersion, bit-for-bit future-bar
     perturbation invariance (`max_future_leak_diff <= 1e-10`), and monotonic L2 LOB execution costs.
"""
from __future__ import annotations

import math
import sys
from typing import Any

import numpy as np
import pandas as pd


def _resolve_symbol_col(df: pd.DataFrame, symbol_col: str) -> str:
    if symbol_col in df.columns:
        return symbol_col
    for alt in ("symbol", "stock_id", "ticker"):
        if alt in df.columns:
            return alt
    raise TypeError(f"df is missing symbol column {symbol_col!r}")


def _extract_numeric(df: pd.DataFrame, cols: tuple[str, ...], default: float = 0.0) -> np.ndarray:
    for col in cols:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(default).to_numpy(dtype=np.float64)
    return np.full(len(df), default, dtype=np.float64)


def compute_lob_liquidity_shock_shield(
    df: pd.DataFrame,
    window: int = 20,
    min_periods: int = 3,
    date_col: str = "date",
    symbol_col: str = "symbol",
) -> pd.Series:
    """Compute strictly causal (<= t) 20-day LOB volatility-of-volatility & liquidity absorption shield."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if df.empty:
        return pd.Series(dtype=float)
    if date_col not in df.columns:
        raise TypeError(f"df is missing required date column {date_col!r}")
    sym_col = _resolve_symbol_col(df, symbol_col)

    work = pd.DataFrame(index=df.index)
    work[date_col] = df[date_col].astype(str)
    work[sym_col] = df[sym_col].astype(str)
    work["_orig_pos"] = np.arange(len(df))

    gk = _extract_numeric(df, ("garman_klass_volatility", "parkinson_volatility", "idio_vol_20d"), 0.02)
    ret_1d = _extract_numeric(df, ("stock_excess_return_1d", "ret_1d", "return_1d"), 0.0)
    vol = _extract_numeric(df, ("volume", "turnover_z"), 1.0)
    mgn_z = _extract_numeric(df, ("margin_balance_z30", "margin_chg_1d"), 0.0)
    amihud = _extract_numeric(df, ("amihud_illiquidity",), 0.001)

    work["_gk"] = gk
    work["_ret_1d"] = ret_1d
    work["_log_vol"] = np.log1p(np.maximum(vol, 0.0))
    work["_mgn_z"] = mgn_z
    work["_amihud"] = amihud

    work.sort_values([sym_col, date_col], inplace=True, kind="mergesort")
    grp_sym = work.groupby(sym_col, sort=False)

    gk_mean20 = grp_sym["_gk"].rolling(window, min_periods=min_periods).mean().to_numpy(dtype=np.float64)
    gk_std20 = grp_sym["_gk"].rolling(window, min_periods=min_periods).std().fillna(0.0).to_numpy(dtype=np.float64)
    ret_std20 = grp_sym["_ret_1d"].rolling(window, min_periods=min_periods).std().fillna(0.0).to_numpy(dtype=np.float64)
    vol_std20 = grp_sym["_log_vol"].rolling(window, min_periods=min_periods).std().fillna(0.0).to_numpy(dtype=np.float64)
    amihud_std20 = grp_sym["_amihud"].rolling(window, min_periods=min_periods).std().fillna(0.0).to_numpy(dtype=np.float64)

    work["_gk_mean20"] = np.nan_to_num(gk_mean20, nan=0.02)
    work["_gk_std20"] = np.nan_to_num(gk_std20, nan=0.0)
    work["_ret_std20"] = np.nan_to_num(ret_std20, nan=0.0)
    work["_vol_std20"] = np.nan_to_num(vol_std20, nan=0.0)
    work["_amihud_std20"] = np.nan_to_num(amihud_std20, nan=0.0)
    work.sort_values("_orig_pos", inplace=True, kind="mergesort")

    d_key = work[date_col]

    def _cs_rank(arr: np.ndarray) -> np.ndarray:
        s = pd.Series(arr, index=work.index)
        if s.nunique() <= 1:
            return np.zeros(len(s), dtype=np.float64)
        rk = s.groupby(d_key).rank(pct=True, method="average").to_numpy(dtype=np.float64)
        return np.nan_to_num(rk - 0.5, nan=0.0)

    r_ret_s20 = _cs_rank(work["_ret_std20"].to_numpy(dtype=np.float64))
    r_gk_s20 = _cs_rank(work["_gk_std20"].to_numpy(dtype=np.float64))
    r_gk_m20 = _cs_rank(work["_gk_mean20"].to_numpy(dtype=np.float64))
    r_vol_s20 = _cs_rank(work["_vol_std20"].to_numpy(dtype=np.float64))
    r_ami_s20 = _cs_rank(work["_amihud_std20"].to_numpy(dtype=np.float64))
    r_mgn_z = _cs_rank(work["_mgn_z"].to_numpy(dtype=np.float64))

    raw_shield = (
        -0.32 * r_ret_s20
        - 0.24 * r_gk_s20
        - 0.20 * r_gk_m20
        - 0.10 * r_vol_s20
        - 0.06 * r_ami_s20
        - 0.08 * r_mgn_z
    )
    s_raw = pd.Series(raw_shield, index=df.index)
    g_d = s_raw.groupby(df[date_col])
    mean = g_d.transform("mean")
    std = g_d.transform("std").replace(0.0, 1.0).fillna(1.0)
    z = ((s_raw - mean) / (std + 1e-8)).clip(-3.5, 3.5).fillna(0.0)
    return pd.Series(z.to_numpy(dtype=float), index=df.index, name="lob_liquidity_shock_shield")


def estimate_intraday_lob_execution_cost(
    symbol: str,
    order_notional_rmb: float = 150_000.0,
    adv_rmb: float = 50_000_000.0,
    gk_volatility: float = 0.022,
    amihud_illiquidity: float = 0.0015,
    limit_proximity_ratio: float = 0.25,
    queue_depth_fraction: float = 0.45,
    margin_crowding_z: float = 0.0,
    holding_days: int = 5,
    is_short_leg: bool = False,
) -> dict[str, Any]:
    """Estimate L2 LOB queue-position impact, square-root participation slippage, and short-borrow fee."""
    raw = str(symbol).strip().upper()
    digits = "".join(c for c in raw if c.isdigit())
    norm = f"SH{digits}" if (len(digits) == 6 and digits.startswith(("5", "6"))) else (
        f"SZ{digits}" if len(digits) == 6 else raw
    )

    if digits.startswith(("51", "56", "58", "15")):
        board_regime = "Broad_Index_ETF"
        half_spread_bps = 2.5
        impact_eta = 0.18
        base_borrow_annual_pct = 2.50
    elif norm.startswith(("SH688", "SZ300")):
        board_regime = "STAR_ChiNext_20pct"
        half_spread_bps = 7.5
        impact_eta = 0.42
        base_borrow_annual_pct = 6.50
    else:
        board_regime = "Main_Board_10pct"
        half_spread_bps = 4.8
        impact_eta = 0.32
        base_borrow_annual_pct = 4.50

    safe_adv = max(float(adv_rmb), 100_000.0)
    participation_rate = max(0.0, float(order_notional_rmb)) / safe_adv
    q_pos = float(np.clip(queue_depth_fraction, 0.05, 0.99))
    lim_prox = float(np.clip(limit_proximity_ratio, 0.0, 1.0))
    sigma_gk = max(float(gk_volatility), 0.005)
    crowding_surcharge = max(0.0, float(margin_crowding_z))

    # Cont-Stoikov-inspired passive fill probability given queue depth q_pos and participation rate
    passive_fill_prob = float(np.clip(math.exp(-1.25 * q_pos * (1.0 + 4.0 * participation_rate) - 0.85 * max(0.0, lim_prox - 0.65)), 0.02, 0.98))

    # L2 Queue-Position & Adverse Selection Cost (bps)
    queue_impact_bps = float(
        half_spread_bps
        * (0.42 * passive_fill_prob + 1.55 * (1.0 - passive_fill_prob))
        * (1.0 + 0.18 * min(crowding_surcharge, 3.0))
    )

    # Square-root participation market impact + Amihud depth term (bps)
    sqrt_impact_bps = float(
        10_000.0 * impact_eta * sigma_gk * math.sqrt(participation_rate)
        + min(12.0, float(amihud_illiquidity) * 1200.0)
    )

    # Price-limit lockout penalty when stock approaches +-10% / +-20% limit cap
    limit_lockout_penalty_bps = float(
        3200.0 * (max(0.0, lim_prox - 0.72) ** 2)
    )

    one_way_lob_slippage_bps = round(queue_impact_bps + sqrt_impact_bps + limit_lockout_penalty_bps, 3)

    # Statutory A-share stamp duty (5 bps sell-side only for stocks, 0 for ETFs) + broker commission (2.5 bps/side)
    statutory_round_trip_bps = 5.0 if board_regime == "Broad_Index_ETF" else 10.0
    round_trip_lob_cost_bps = round(2.0 * one_way_lob_slippage_bps + statutory_round_trip_bps, 3)

    # Tiered A-share short-selling (融券) borrow fee schedule
    short_borrow_annual_pct = round(
        base_borrow_annual_pct + (2.00 if crowding_surcharge > 1.0 and board_regime != "Broad_Index_ETF" else 0.0),
        2,
    )
    shadow_short_holding_bps = round(short_borrow_annual_pct * 100.0 * (max(1, int(holding_days)) / 252.0), 3)
    applied_borrow_cost_bps = shadow_short_holding_bps if is_short_leg else 0.0

    total_all_in_cost_bps = round(round_trip_lob_cost_bps + applied_borrow_cost_bps, 3)
    execution_feasible = bool(lim_prox < 0.96 and participation_rate <= 0.05 and one_way_lob_slippage_bps <= 45.0)

    return {
        "symbol": norm,
        "board_regime": board_regime,
        "order_notional_rmb": round(float(order_notional_rmb), 2),
        "adv_rmb": round(safe_adv, 2),
        "participation_rate_pct": round(participation_rate * 100.0, 4),
        "passive_fill_probability": round(passive_fill_prob, 4),
        "queue_impact_bps": round(queue_impact_bps, 3),
        "sqrt_participation_impact_bps": round(sqrt_impact_bps, 3),
        "limit_lockout_penalty_bps": round(limit_lockout_penalty_bps, 3),
        "one_way_lob_slippage_bps": one_way_lob_slippage_bps,
        "statutory_round_trip_bps": statutory_round_trip_bps,
        "round_trip_lob_cost_bps": round_trip_lob_cost_bps,
        "short_borrow_fee_annual_pct": short_borrow_annual_pct,
        "shadow_short_holding_cost_bps": shadow_short_holding_bps,
        "applied_borrow_cost_bps": applied_borrow_cost_bps,
        "total_all_in_cost_bps": total_all_in_cost_bps,
        "execution_feasible": execution_feasible,
    }


def audit_lob_liquidity_causality(
    df: pd.DataFrame,
    signal_col: str | None = None,
    date_col: str = "date",
    symbol_col: str = "symbol",
) -> dict[str, Any]:
    """Executable guard verifying LOB liquidity shield causality, dispersion, and execution cost monotonicity."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise TypeError("df must be a non-empty DataFrame")
    if date_col not in df.columns:
        raise TypeError(f"df is missing required column {date_col!r}")
    sym_col = _resolve_symbol_col(df, symbol_col)

    causal_base_s = compute_lob_liquidity_shock_shield(df, date_col=date_col, symbol_col=sym_col)
    if signal_col is not None:
        if signal_col not in df.columns:
            raise TypeError(f"signal_col {signal_col!r} not in df.columns")
        sig = pd.to_numeric(df[signal_col], errors="coerce").fillna(0.0)
    else:
        sig = causal_base_s

    cs_std = sig.groupby(df[date_col]).std().dropna()
    unique_dates = sorted(df[date_col].astype(str).unique())
    active_std = cs_std[cs_std.index > unique_dates[min(3, len(unique_dates) - 1)]] if len(unique_dates) > 3 else cs_std
    mean_cs_std = float(active_std.mean()) if len(active_std) else float(cs_std.mean() or 0.0)
    non_degenerate_dispersion = bool(mean_cs_std > 0.05)

    # Future-bar perturbation test
    cut_idx = max(1, min(len(unique_dates) - 2, int(len(unique_dates) * 0.70)))
    cut_date = unique_dates[cut_idx]
    past_mask = (df[date_col].astype(str) <= cut_date).to_numpy()

    pert = df.copy()
    for c in ("garman_klass_volatility", "parkinson_volatility", "idio_vol_20d", "ret_1d", "stock_excess_return_1d", "amihud_illiquidity"):
        if c in pert.columns:
            pert.loc[~past_mask, c] = pd.to_numeric(pert.loc[~past_mask, c], errors="coerce").fillna(0.0) + 5.0

    causal_base = causal_base_s.to_numpy(dtype=float)
    causal_pert = compute_lob_liquidity_shock_shield(pert, date_col=date_col, symbol_col=sym_col).to_numpy(dtype=float)
    max_future_leak = float(np.nanmax(np.abs(causal_base[past_mask] - causal_pert[past_mask])))

    caller_matches_causal = True
    if signal_col is not None and non_degenerate_dispersion:
        corr = float(np.corrcoef(sig.to_numpy(dtype=float), causal_base)[0, 1]) if np.std(sig) > 1e-8 else 0.0
        caller_matches_causal = bool(corr >= 0.50)

    # Verify L2 LOB execution cost monotonicity & borrow tier schedule
    low_part = estimate_intraday_lob_execution_cost("SH600036", order_notional_rmb=50_000.0, adv_rmb=80_000_000.0)
    high_part = estimate_intraday_lob_execution_cost("SH688981", order_notional_rmb=800_000.0, adv_rmb=20_000_000.0, is_short_leg=True)
    lob_cost_schedule_valid = bool(
        high_part["one_way_lob_slippage_bps"] > low_part["one_way_lob_slippage_bps"]
        and high_part["short_borrow_fee_annual_pct"] > low_part["short_borrow_fee_annual_pct"]
        and high_part["passive_fill_probability"] < low_part["passive_fill_probability"]
    )

    passed = bool(
        non_degenerate_dispersion
        and (max_future_leak <= 1e-10)
        and caller_matches_causal
        and lob_cost_schedule_valid
    )
    verdict = (
        "PASS: 20-day LOB volatility-of-volatility liquidity shield is strictly causal (max_leak=0.0) and L2 queue/borrow cost schedule is valid."
        if passed
        else (
            "FAIL: Zero or degenerate cross-sectional dispersion in LOB liquidity shield"
            if not non_degenerate_dispersion
            else "FAIL: Future look-ahead leakage or invalid LOB cost schedule detected"
        )
    )
    return {
        "passed": passed,
        "verdict": verdict,
        "non_degenerate_dispersion": non_degenerate_dispersion,
        "mean_cross_sectional_std": round(mean_cs_std, 6),
        "max_future_leak_diff": float(max_future_leak),
        "caller_matches_causal": caller_matches_causal,
        "lob_cost_schedule_valid": lob_cost_schedule_valid,
        "sample_main_board_lob_cost_bps": low_part["round_trip_lob_cost_bps"],
        "sample_star_short_all_in_cost_bps": high_part["total_all_in_cost_bps"],
        "n_rows": int(len(df)),
        "n_dates": int(len(unique_dates)),
    }


def _demo() -> int:
    rng = np.random.default_rng(19)
    dates = [f"2025-03-{d:02d}" for d in range(1, 22)]
    rows = []
    for d in dates:
        for sym in ("SH600036", "SH601318", "SZ000001", "SH688981", "SZ300750"):
            rows.append({
                "date": d,
                "symbol": sym,
                "garman_klass_volatility": float(rng.uniform(0.010, 0.045)),
                "ret_1d": float(rng.normal(0.0, 0.02)),
                "volume": float(rng.uniform(1e6, 5e7)),
                "margin_balance_z30": float(rng.normal(0.0, 1.0)),
                "amihud_illiquidity": float(rng.uniform(0.0005, 0.004)),
            })
    df = pd.DataFrame(rows)
    df["lob_shield"] = compute_lob_liquidity_shock_shield(df, window=10, min_periods=2)
    audit = audit_lob_liquidity_causality(df, signal_col="lob_shield")
    print(f"LOB Liquidity Shock Shield demo: passed={audit['passed']} cs_std={audit['mean_cross_sectional_std']:.4f}")
    return 0 if audit["passed"] else 1


if __name__ == "__main__":
    sys.exit(_demo())
'''

LOB_LIQUIDITY_SKILL_BLUEPRINT = SkillUpgradeBlueprint(
    name="lob-liquidity-shock-shield",
    plugin="fin-microstructure",
    module_name="lob_liquidity_shock_shield",
    description=(
        "[fin-microstructure] Compute a 20-day order-book volatility-of-volatility liquidity absorption "
        "shield and estimate L2 queue-position slippage, limit-lockout penalty, and A-share short-borrow "
        "fee schedules. TRIGGER - lob_liquidity_shock_shield, compute_lob_liquidity_shock_shield, "
        "estimate_intraday_lob_execution_cost, audit_lob_liquidity_causality, check_lob_liquidity_gate, "
        "volatility of volatility liquidity absorption shield, 20-day Garman-Klass range vol-of-vol "
        "stability, L2 queue position participation slippage and short borrow fee schedule, "
        "限价订单簿波动率之波动率流动性吸收护盾与L2队列冲击融券费率建模. "
        "SKIP for Cont-Stoikov-Talreja birth-death queue arrival rates (limit-order-book-models), "
        "for Lee-Ready tick classification and Kyle lambda estimation (intraday-microstructure), "
        "for Almgren-Chriss parent order scheduling (execution-algorithms), and for A-share board "
        "sign bifurcation (cross-board-supply-chain-rsi)."
    ),
    body_markdown=r"""# 20-Day LOB Volatility-of-Volatility Liquidity Shield & L2 Execution Cost Engine (`lob-liquidity-shock-shield`)

## Why 1-Day Liquidity Snapshots Miss Order-Book Replenishment Fragility

Single-day range volatility ($\text{GK}_{i,t}$) or single-day Amihud illiquidity ($\text{Amihud}_{i,t}$) conflates transient news volume with structural limit-order-book (`LOB`) depletion. Following Chordia, Subrahmanyam & Anshuman (2001) and Lou & Shu (2017), **second-moment liquidity instability (volatility-of-volatility and illiquidity dispersion over a 20-day rolling window)** captures whether passive market makers replenish depth consistently or withdraw during stress:

$$S_{i,t}^{\text{LOB}} = \text{CS-ZScore}_t\!\Bigl(-0.32\,R_t(\sigma_{20\text{d}}(r_i)) - 0.24\,R_t(\sigma_{20\text{d}}(\text{GK}_i)) - 0.20\,R_t(\mu_{20\text{d}}(\text{GK}_i)) - 0.10\,R_t(\sigma_{20\text{d}}(\log V_i)) - 0.06\,R_t(\sigma_{20\text{d}}(\text{Amihud}_i)) - 0.08\,R_t(Z_{30\text{d}}(\text{Margin}_i))\Bigr)$$

## L2 Queue-Position Slippage, Limit-Lockout Penalty & Short-Borrow Schedule

`scripts/lob_liquidity_shock_shield.py` (`fin_skills.microstructure.lob_liquidity_shock_shield`) also provides `estimate_intraday_lob_execution_cost(...)` for live/paper trading (`14:30 CST` execution window):

1. **Passive Queue Fill Probability & Adverse Selection**:
   $$P_{\text{fill}}(q, \pi) = \exp\!\bigl(-1.25\,q\,(1 + 4\pi) - 0.85\,\max(0, \rho_{\text{lim}} - 0.65)\bigr)$$
2. **Square-Root Participation Impact + Price-Limit Lockout Penalty**:
   $$\text{Slippage}_{\text{1-way, bps}} = \text{QueueImpact}_{\text{bps}} + 10^4\,\eta_{\text{board}}\,\sigma_{\text{GK}}\sqrt{\frac{Q}{\text{ADV}}} + 3200\,\max(0, \rho_{\text{lim}} - 0.72)^2$$
3. **A-Share Tiered Short-Borrow (`融券`) Fee Schedule**:
   - `2.50%` p.a. (`Broad_Index_ETF`), `4.50%` p.a. (`10%` Main Board), `6.50%` p.a. (`20%` STAR/ChiNext), plus `+2.00%` p.a. crowding surcharge when $Z_{\text{margin}} > 1.0$.

## Executable API & Guard Verification

```bash
python3 -c "from fin_skills.microstructure.lob_liquidity_shock_shield import _demo; raise SystemExit(_demo())"
```

## Cross-References

- `limit-order-book-models` — Cont-Stoikov-Talreja birth-death queue arrival rates and passive fill probabilities.
- `intraday-microstructure` — Tick-level Lee-Ready classification, Kyle's lambda, and VPIN.
- `cross-board-supply-chain-rsi` — Leave-one-out (`j != i`) A-share peer spillover and `10%` vs `20%` board sign bifurcation.
""",
    script_source=LOB_LIQUIDITY_SCRIPT_SOURCE,
    verified_on="2026-10-02",
    xref_neighbors=(
        "limit-order-book-models",
        "intraday-microstructure",
        "cross-board-supply-chain-rsi",
    ),
    guard_name="lob_liquidity_gate",
    probe_queries=[
        {
            "q": "How do I use lob_liquidity_shock_shield and compute_lob_liquidity_shock_shield with estimate_intraday_lob_execution_cost to model 20-day Garman-Klass range vol-of-vol stability and L2 queue position participation slippage?",
            "expect": "lob-liquidity-shock-shield",
        },
        {
            "q": "如何构建限价订单簿波动率之波动率流动性吸收护盾与L2队列冲击融券费率建模 (lob_liquidity_shock_shield) 并调用 check_lob_liquidity_gate 检查因果性？",
            "expect": "lob-liquidity-shock-shield",
        },
    ],
)

