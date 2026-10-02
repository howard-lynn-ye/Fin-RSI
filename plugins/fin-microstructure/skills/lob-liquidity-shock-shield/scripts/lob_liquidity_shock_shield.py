#!/usr/bin/env python3
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
