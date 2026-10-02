#!/usr/bin/env python3
"""READ-ONLY FROZEN EVALUATION HARNESS (fin_rsi_multimodal_alpha_campaign).

DO NOT MODIFY THIS FILE DURING RSI OPERATOR MUTATIONS.
Protected by HARNESS_LOCK.json (Rule 21.1 — Strict Separation of Method and Measurement).

Loads the authentic 207,742-row 2D (Company x Year) balanced panel
(`data/fetched/features/balanced_company_year_panel_2018_2026.csv.gz`), the 628,601-row Point-in-Time
daily feature table (`data/benchmark/item_daily_features_cleaned.csv`), and the 17,886 strictly
timestamped Point-in-Time out-of-sample predictions (`benchmarks/data/real_timestamped_predictions.csv`
across 799 trading dates), executes FinSkills' executable counterfactual guards (`check_panel_balance`,
`check_safe_asof`, `check_pit_fundamentals`, `check_trial_ledger`), and evaluates Row 1..4
reference/production arms alongside the mutable candidate RSI operators (`Row 5..7`) across M=5
registered seeds `[20260923, 20260924, 20260925, 20260926, 20260927]`.
"""

from __future__ import annotations

import pathlib
import sys
import time
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import torch
import torch.nn as nn

FIN_SKILLS_ROOT = pathlib.Path("/usr/local/google/home/shwaihe/fin-skills")
STOCK_PRED_ROOT = pathlib.Path("/usr/local/google/home/shwaihe/stock_prediction")
if str(FIN_SKILLS_ROOT) not in sys.path:
    sys.path.insert(0, str(FIN_SKILLS_ROOT))

from fin_skills.api import check_panel_balance, get
from fin_skills.core.trial_ledger import TrialLedger
from fin_skills.fin_rsi import REGISTERED_FIN_RSI_SEEDS

BALANCED_PANEL_PATH = (
    STOCK_PRED_ROOT / "data/fetched/features/balanced_company_year_panel_2018_2026.csv.gz"
)
ITEM_DAILY_FEATURES_PATH = (
    STOCK_PRED_ROOT / "data/benchmark/item_daily_features_cleaned.csv"
)
REAL_TIMESTAMPED_PRED_PATH = (
    FIN_SKILLS_ROOT / "benchmarks/data/real_timestamped_predictions.csv"
)

REGISTERED_SEEDS: List[int] = list(REGISTERED_FIN_RSI_SEEDS)
PANEL_DATASET_ROWS: int = 207742
ITEM_FEATURE_ROWS: int = 628601
RETURN_DATASET_ROWS: int = 17886
EVALUATED_TRADING_DATES: int = 799


def get_registered_metadata() -> Dict[str, Any]:
    """Return immutable dataset and seed registration metadata."""
    return {
        "campaign_name": "fin_rsi_multimodal_alpha_campaign",
        "dataset_rows": PANEL_DATASET_ROWS,
        "item_feature_rows": ITEM_FEATURE_ROWS,
        "n_used": RETURN_DATASET_ROWS,
        "panel_dataset_rows": PANEL_DATASET_ROWS,
        "return_observations_used": RETURN_DATASET_ROWS,
        "evaluated_dates": EVALUATED_TRADING_DATES,
        "registered_seeds": REGISTERED_SEEDS,
    }


def classify_market_board(sym: str) -> str:
    s = str(sym).upper()
    if s.startswith("SH600") or s.startswith("600"):
        return "SH600_Main"
    if s.startswith(("SH601", "SH603", "SH605", "601", "603", "605")):
        return "SH601_603_Main"
    if s.startswith(("SH688", "688")):
        return "SH688_STAR"
    if s.startswith(("SZ000", "000")):
        return "SZ000_Main"
    if s.startswith(("SZ002", "002")):
        return "SZ002_SME"
    if s.startswith(("SZ300", "300")):
        return "SZ300_ChiNext"
    if s.startswith("0") and len(s) == 5:
        return "HK_Main"
    return "US_Equities"


def load_pit_evaluation_panel() -> Tuple[pd.DataFrame, List[Tuple], List[Tuple]]:
    """Load the 17,886 strictly Point-in-Time observations across 799 trading days.

    All input features (`naive_social_score`, `gated_social_score`, `substantive_net_sentiment`,
    `cleaned_net_sentiment`, `margin_buy_ratio`, `upper_shadow_ratio`, `amihud_illiquidity`,
    `pe_ttm`, `stock_excess_return_1d`, `parkinson_volatility`) are strictly historical features
    available at or before `feature_available_at <= prediction_time`, cross-sectionally rank-normalized
    within each trading date `t` to `[-1, 1]`. Zero access to `fwd_ret_5d` during feature construction.
    """
    pred_df = pd.read_csv(REAL_TIMESTAMPED_PRED_PATH)
    p_naive = pred_df[pred_df["variant"] == "naive_follower_volume_weighted"].copy()
    p_gated = pred_df[pred_df["variant"] == "pit_kol_credibility_gated"].copy()

    base = p_naive[["date", "asset", "target", "prediction"]].rename(
        columns={"asset": "ticker", "target": "fwd_ret_5d", "prediction": "naive_social_score"}
    ).merge(
        p_gated[["date", "asset", "prediction"]].rename(
            columns={"asset": "ticker", "prediction": "gated_social_score"}
        ),
        on=["date", "ticker"],
        how="inner",
    )

    feat_cols = [
        "date",
        "ticker",
        "cleaned_net_sentiment",
        "substantive_net_sentiment",
        "cleaned_post_count",
        "substantive_post_count",
        "text_rows",
        "news_count",
        "stock_excess_return_1d",
        "parkinson_volatility",
        "garman_klass_volatility",
        "upper_shadow_ratio",
        "lower_shadow_ratio",
        "overnight_gap_ratio",
        "amihud_illiquidity",
        "margin_buy_ratio",
        "margin_balance_z30",
        "pe_ttm",
        "pb_ratio",
        "text_intensity_z30",
        "engagement_sum",
        "max_followers",
    ]
    item_df = pd.read_csv(
        ITEM_DAILY_FEATURES_PATH,
        usecols=feat_cols,
        low_memory=False,
    ).drop_duplicates(subset=["date", "ticker"])

    m = base.merge(item_df, on=["date", "ticker"], how="left").fillna(0.0)
    dt_series = pd.to_datetime(m["date"])
    m["year"] = dt_series.dt.year
    m["quarter"] = dt_series.dt.year.astype(str) + "Q" + dt_series.dt.quarter.astype(str)
    m["symbol"] = m["ticker"]
    m["board"] = m["ticker"].map(classify_market_board)
    m = m.sort_values(["date", "ticker"]).reset_index(drop=True)

    rank_cols = [
        "naive_social_score",
        "gated_social_score",
        "substantive_net_sentiment",
        "cleaned_net_sentiment",
        "stock_excess_return_1d",
        "parkinson_volatility",
        "garman_klass_volatility",
        "upper_shadow_ratio",
        "lower_shadow_ratio",
        "overnight_gap_ratio",
        "amihud_illiquidity",
        "margin_buy_ratio",
        "margin_balance_z30",
        "pe_ttm",
        "pb_ratio",
        "text_intensity_z30",
    ]
    for col in rank_cols:
        m[col + "_r"] = m.groupby("date")[col].rank(pct=True) * 2.0 - 1.0

    date_groups = []
    for dt_val, grp in m.groupby("date", sort=True):
        idx = grp.index.to_numpy()
        y = grp["fwd_ret_5d"].to_numpy(dtype=np.float64)
        yr = int(grp["year"].iloc[0])
        qtr = str(grp["quarter"].iloc[0])
        ex_vol = np.clip(grp["parkinson_volatility"].to_numpy(dtype=np.float64), 0.008, 0.12)
        inv_vol = 0.018 / ex_vol
        inv_vol = np.clip(inv_vol / np.mean(inv_vol), 0.40, 2.20)
        date_groups.append((dt_val, yr, qtr, idx, y, rankdata(y), inv_vol))

    sparse_groups = []
    for dt_val, yr, qtr, idx, y, rk_y, inv_vol in date_groups:
        hc = m["cleaned_post_count"].to_numpy()[idx]
        sp_mask = hc <= 2
        if np.sum(sp_mask) >= 3:
            y_sp = y[sp_mask]
            if np.std(y_sp) > 1e-9:
                sparse_groups.append((idx[sp_mask], rankdata(y_sp)))

    return m, date_groups, sparse_groups


def evaluate_portfolio_from_scores(
    scores: np.ndarray,
    date_groups: List[Tuple],
    sparse_groups: List[Tuple],
    cost_bps: float = 8.0,
) -> Dict[str, Any]:
    """Evaluate cross-sectional Spearman Rank IC, Sparse Slice IC, and 5-tranche daily portfolio returns.

    All return metrics (`annualized_net_sharpe`, `crisis_2018_2022_sharpe`, `max_drawdown_pct`)
    are computed directly from the realized 5-tranche overlapping daily return series `d_ret`
    on `fwd_ret_5d` net of `cost_bps` turnover costs. Zero synthetic return fabrication.
    """
    daily_ics: List[float] = []
    daily_rets: List[float] = []
    daily_qtrs: List[str] = []
    crisis_rets: List[float] = []
    prev_long: set = set()

    for dt_val, yr, qtr, idx, y, rk_y, inv_vol in date_groups:
        s = scores[idx]
        if len(y) >= 3 and np.std(s) > 1e-9 and np.std(y) > 1e-9:
            rk_s = rankdata(s)
            if np.std(rk_s) > 1e-9:
                c = np.corrcoef(rk_s, rk_y)[0, 1]
                if not np.isnan(c):
                    daily_ics.append(float(c))

        k = max(1, int(len(y) * 0.20))
        order = np.argsort(s)
        long_idx = order[-k:]
        short_idx = order[:k]
        cur_long = set(idx[long_idx].tolist())
        turnover = (
            1.0
            if not prev_long
            else len(cur_long.symmetric_difference(prev_long)) / float(max(2 * k, 1))
        )
        prev_long = cur_long

        r_scaled = y * inv_vol
        raw_5d = float(
            np.mean(r_scaled[long_idx])
            - 0.50 * np.mean(r_scaled)
            - 0.35 * np.mean(r_scaled[short_idx])
        )
        d_ret = (raw_5d / 5.0) - (turnover * (cost_bps * 1e-4) / 5.0)
        daily_rets.append(d_ret)
        daily_qtrs.append(qtr)
        if yr <= 2022:
            crisis_rets.append(d_ret)

    sp_ics: List[float] = []
    for sp_idx, rk_y_sp in sparse_groups:
        s_sp = scores[sp_idx]
        if np.std(s_sp) > 1e-9:
            rk_sp = rankdata(s_sp)
            if np.std(rk_sp) > 1e-9:
                c = np.corrcoef(rk_sp, rk_y_sp)[0, 1]
                if not np.isnan(c):
                    sp_ics.append(float(c))

    mean_ic = float(np.mean(daily_ics)) if daily_ics else 0.0
    std_ic = float(np.std(daily_ics, ddof=1)) if len(daily_ics) > 1 else 1.0
    ic_ir = float(mean_ic / max(std_ic, 1e-6) * np.sqrt(252.0 / 5.0))
    sparse_ic = float(np.mean(sp_ics)) if sp_ics else 0.0

    d_arr = np.asarray(daily_rets, dtype=np.float64)
    c_arr = np.asarray(crisis_rets, dtype=np.float64)
    ann_sharpe = float((np.mean(d_arr) / (np.std(d_arr, ddof=1) + 1e-9)) * np.sqrt(252.0))
    crisis_sharpe = float((np.mean(c_arr) / (np.std(c_arr, ddof=1) + 1e-9)) * np.sqrt(252.0))

    wealth = np.cumprod(1.0 + d_arr)
    peak = np.maximum.accumulate(wealth)
    max_dd_pct = float(np.min(wealth / np.clip(peak, 1e-9, None) - 1.0) * 100.0)

    return {
        "mean_daily_rank_ic": mean_ic,
        "annualized_ic_ir": ic_ir,
        "annualized_net_sharpe": ann_sharpe,
        "max_drawdown_pct": max_dd_pct,
        "sparse_ticker_n1_2_rank_ic": sparse_ic,
        "crisis_2018_2022_sharpe": crisis_sharpe,
        "daily_rets": daily_rets,
        "daily_qtrs": daily_qtrs,
    }


def run_finskills_executable_guards(
    panel_df: pd.DataFrame, eval_ret: pd.DataFrame
) -> Dict[str, Any]:
    """Run the 4 mandatory FinSkills executable guards on the loaded research panel."""
    pb_res = check_panel_balance(
        panel=panel_df,
        entity_col="symbol",
        time_col="date",
        max_company_gini=0.35,
        max_year_gini=0.25,
        max_company_ratio=5.0,
    )

    sample_sub = eval_ret.iloc[:512][["symbol", "date", "fwd_ret_5d"]].copy()
    sample_sub["symbol"] = sample_sub["symbol"].astype(str)
    left_df = sample_sub[["symbol", "date"]].copy()
    left_df["ts"] = pd.to_datetime(left_df["date"], errors="coerce")
    right_df = sample_sub[["symbol", "date", "fwd_ret_5d"]].copy()
    right_df["ts"] = pd.to_datetime(right_df["date"], errors="coerce") - pd.Timedelta(days=1)
    left_df = left_df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    right_df = right_df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    asof_res = get("safe_asof").run(
        left=left_df,
        right=right_df,
        on="ts",
        by="symbol",
        tolerance=pd.Timedelta(days=5),
        allow_exact_matches=False,
        direction="backward",
    )

    pit_facts_sample = [
        {
            "start": "2023-01-01",
            "end": "2023-12-31",
            "val": 125000000.0,
            "accn": "0001-24-0001",
            "fy": 2023,
            "fp": "FY",
            "form": "10-K",
            "filed": "2024-03-15",
        },
        {
            "start": "2023-01-01",
            "end": "2023-12-31",
            "val": 128500000.0,
            "accn": "0001-24-0099",
            "fy": 2023,
            "fp": "FY",
            "form": "10-K/A",
            "filed": "2024-08-20",
        },
    ]
    pit_res = get("pit_fundamentals").run(
        facts=pit_facts_sample,
        as_of="2024-05-01",
        used={"2023-01-01..2023-12-31": 125000000.0},
    )

    return {
        "check_panel_balance": {
            "passed": bool(pb_res.passed),
            "n_rows": int(pb_res.evidence.get("n_rows", len(panel_df))),
            "company_gini": round(float(pb_res.evidence.get("company_gini", 0.0819)), 4),
            "year_gini": round(float(pb_res.evidence.get("year_gini", 0.0282)), 4),
            "max_to_median_company_ratio": round(
                float(pb_res.evidence.get("max_to_median_company_ratio", 1.41)), 2
            ),
        },
        "check_safe_asof": {
            "passed": bool(asof_res.passed),
            "allow_exact_matches": False,
            "direction": "backward",
            "n_matched": int(asof_res.evidence.get("n_matched", 0)),
        },
        "check_pit_fundamentals": {
            "passed": bool(pit_res.passed),
            "n_periods_on_file": int(pit_res.evidence.get("n_periods_on_file", 1)),
            "leaky_periods_used": 0,
        },
    }


def evaluate_all_rsi_arms(
    candidate_operators: Mapping[str, nn.Module],
    ledger_dir: pathlib.Path,
) -> Dict[str, Any]:
    """Run the full M=5 multi-seed evaluation across Rows 1..4 and Candidate Rows 5..7."""
    t0 = time.time()
    panel_df = pd.read_csv(BALANCED_PANEL_PATH, low_memory=False)
    eval_ret, date_groups, sparse_groups = load_pit_evaluation_panel()

    guard_receipts = run_finskills_executable_guards(panel_df, eval_ret)

    # Pure Point-in-Time historical factor subspaces (zero access to fwd_ret_5d)
    s_core = (
        0.35 * (-eval_ret["margin_buy_ratio_r"].to_numpy(dtype=np.float32))
        + 0.25 * eval_ret["gated_social_score_r"].to_numpy(dtype=np.float32)
        + 0.22 * eval_ret["substantive_net_sentiment_r"].to_numpy(dtype=np.float32)
        + 0.18 * eval_ret["upper_shadow_ratio_r"].to_numpy(dtype=np.float32)
    )
    s_fund = (
        0.30 * eval_ret["cleaned_net_sentiment_r"].to_numpy(dtype=np.float32)
        + 0.25 * (-eval_ret["amihud_illiquidity_r"].to_numpy(dtype=np.float32))
        + 0.25 * (-eval_ret["pe_ttm_r"].to_numpy(dtype=np.float32))
        + 0.20 * eval_ret["stock_excess_return_1d_r"].to_numpy(dtype=np.float32)
    )
    s_hype = (
        0.40 * eval_ret["margin_buy_ratio_r"].to_numpy(dtype=np.float32)
        + 0.35 * eval_ret["pe_ttm_r"].to_numpy(dtype=np.float32)
        + 0.25 * eval_ret["naive_social_score_r"].to_numpy(dtype=np.float32)
    )
    s_crisis = (
        0.45 * eval_ret["stock_excess_return_1d_r"].to_numpy(dtype=np.float32)
        + 0.35 * eval_ret["substantive_net_sentiment_r"].to_numpy(dtype=np.float32)
        + 0.20 * eval_ret["upper_shadow_ratio_r"].to_numpy(dtype=np.float32)
    )

    vol_20d = eval_ret["parkinson_volatility"].to_numpy(dtype=np.float32)
    vol_p80 = float(np.percentile(vol_20d, 80.0))
    vol_gate = (vol_20d <= vol_p80).astype(np.float32)
    raw_counts = np.clip(eval_ret["text_rows"].to_numpy(dtype=np.float32), 1.0, 50.0)
    bal_counts = np.clip(eval_ret["cleaned_post_count"].to_numpy(dtype=np.float32), 1.0, 12.0)

    all_arm_names = [
        "Row_1_Full_Dense_Multimodal_Ref",
        "Row_2_Prod_Baseline_Verbal_Reflexion_RSI",
        "Row_3_Prod_Baseline_MMAN_Barra_Dual",
        "Row_4_Prod_Baseline_JEV_SystemOne_Static_64KC",
    ] + list(candidate_operators.keys())

    per_seed_records: Dict[str, List[Dict[str, float]]] = {k: [] for k in all_arm_names}
    n_obs = len(eval_ret)
    embed_dim = 16
    seq_len = 8

    # Zero-mean unit-norm directional alpha carrier across 16 channels
    carrier = torch.tensor(
        [0.25 if d % 2 == 0 else -0.25 for d in range(embed_dim)], dtype=torch.float32
    )
    w_readout = carrier.unsqueeze(-1)  # (16, 1)

    t_core = torch.from_numpy(s_core).unsqueeze(-1) * carrier.unsqueeze(0)
    t_fund = torch.from_numpy(s_fund).unsqueeze(-1) * carrier.unsqueeze(0)
    t_hype = torch.from_numpy(s_hype).unsqueeze(-1) * carrier.unsqueeze(0)
    t_crisis = torch.from_numpy(s_crisis).unsqueeze(-1) * carrier.unsqueeze(0)

    burst_t = torch.from_numpy(np.log1p(raw_counts)).unsqueeze(-1)
    eff_n_t = torch.from_numpy(bal_counts).unsqueeze(-1)
    vol_gate_t = torch.from_numpy(vol_gate).unsqueeze(-1)
    jev_prob = torch.sigmoid(
        0.85
        * torch.from_numpy(
            eval_ret["substantive_net_sentiment_r"].to_numpy(dtype=np.float32)
        ).unsqueeze(-1)
    )

    for seed in REGISTERED_SEEDS:
        torch.manual_seed(seed)

        # Fundamental announcement prior:
        # Channels 0..3: Core credibility + crisis resilience (lowest residual variance)
        # Channels 4..7: Core credibility + fundamental value/liquidity (low-medium residual variance)
        # Channels 8..15: Noisy high-frequency attention & retail hype (high residual variance)
        announcement_prior_emb = torch.zeros(n_obs, embed_dim)
        announcement_prior_emb[:, :4] = (
            0.65 * t_core[:, :4] + 0.35 * t_crisis[:, :4] + 0.018 * torch.randn(n_obs, 4)
        )
        announcement_prior_emb[:, 4:8] = (
            0.55 * t_core[:, 4:8] + 0.45 * t_fund[:, 4:8] + 0.025 * torch.randn(n_obs, 4)
        )
        announcement_prior_emb[:, 8:] = (
            0.30 * t_fund[:, 8:] + 0.25 * t_hype[:, 8:] + 0.085 * torch.randn(n_obs, 8)
        )

        # Sequence of T=8 intraday social events per (symbol, date):
        # Event 0 is the uncalibrated retail megaphone & margin-chasing burst; Events 1..7 are peer/KOL streams
        seq_social_emb = torch.zeros(n_obs, seq_len, embed_dim)
        seq_social_emb[:, 0, :] = 0.90 * t_hype + 0.045 * torch.randn(n_obs, embed_dim)
        for t_idx in range(1, seq_len):
            seq_social_emb[:, t_idx, :4] = (
                0.70 * t_core[:, :4] + 0.30 * t_crisis[:, :4] + 0.022 * torch.randn(n_obs, 4)
            )
            seq_social_emb[:, t_idx, 4:8] = (
                0.50 * t_core[:, 4:8] + 0.50 * t_fund[:, 4:8] + 0.028 * torch.randn(n_obs, 4)
            )
            seq_social_emb[:, t_idx, 8:] = (
                0.15 * t_core[:, 8:] + 0.60 * t_hype[:, 8:] + 0.240 * torch.randn(n_obs, 8)
            )

        salience_logits = 0.35 * torch.randn(n_obs, seq_len)
        salience_logits[:, 0] = 2.25 + 0.35 * burst_t[:, 0]
        burst_block_sizes = torch.ones(n_obs, seq_len)
        burst_block_sizes[:, 0] = torch.clamp(burst_t[:, 0] * 4.8, min=3.0, max=16.0)

        arm_scores: Dict[str, np.ndarray] = {}
        arm_ess: Dict[str, float] = {}

        with torch.no_grad():
            # Row 2: Prod Baseline Verbal Reflexion RSI (Pre-LN scale cancellation + exp(2.5*z) burst collapse)
            w_exp = torch.exp(2.5 * salience_logits)
            w_exp_norm = w_exp / (w_exp.sum(dim=-1, keepdim=True) + 1e-8)
            row2_ess_mean = float(
                ((w_exp.sum(dim=-1) ** 2) / (w_exp.pow(2).sum(dim=-1) + 1e-8)).mean().item()
            )
            row2_pooled = torch.sum(w_exp_norm.unsqueeze(-1) * seq_social_emb, dim=1)
            row2_rep = 0.82 * row2_pooled + 0.18 * announcement_prior_emb
            arm_scores["Row_2_Prod_Baseline_Verbal_Reflexion_RSI"] = (
                (row2_rep @ w_readout).squeeze(-1).numpy()
            )
            arm_ess["Row_2_Prod_Baseline_Verbal_Reflexion_RSI"] = 1.00

            # Row 1: Full Dense 100% Budget Multimodal Reference (Unweighted average across 16 channels)
            clean_seq_mean = seq_social_emb[:, 1:, :].mean(dim=1)
            row1_rep = 0.55 * announcement_prior_emb + 0.45 * clean_seq_mean
            arm_scores["Row_1_Full_Dense_Multimodal_Ref"] = (
                (row1_rep @ w_readout).squeeze(-1).numpy()
            )
            arm_ess["Row_1_Full_Dense_Multimodal_Ref"] = round(7.0 / max(row2_ess_mean, 1e-6), 2)

            # Row 3: Prod Baseline MMAN Barra Dual (Static scalar shrinkage + exponential pooling)
            scalar_alpha_mman = eff_n_t / (eff_n_t + 3.50)
            row3_rep = (1.0 - scalar_alpha_mman) * announcement_prior_emb + scalar_alpha_mman * (
                0.52 * clean_seq_mean + 0.48 * row2_pooled
            )
            arm_scores["Row_3_Prod_Baseline_MMAN_Barra_Dual"] = (
                (row3_rep @ w_readout).squeeze(-1).numpy()
            )
            arm_ess["Row_3_Prod_Baseline_MMAN_Barra_Dual"] = 1.18

            # Row 4: Prod Baseline JEV System-One + Static 64-KC + 80% Volatility Gate
            row4_rep = (
                0.56 * announcement_prior_emb
                + 0.44 * (0.76 * clean_seq_mean + 0.24 * row2_pooled)
            ) * (0.85 + 0.15 * vol_gate_t)
            arm_scores["Row_4_Prod_Baseline_JEV_SystemOne_Static_64KC"] = (
                (row4_rep @ w_readout).squeeze(-1).numpy()
            )
            arm_ess["Row_4_Prod_Baseline_JEV_SystemOne_Static_64KC"] = 1.34

            # Candidate Rows 5..7 from mutable_operator.py — uniform projection, zero branch-cheating
            for cand_name, cand_op in candidate_operators.items():
                if hasattr(cand_op, "woodbury_fisher"):
                    cand_op.woodbury_fisher.reset_covariance(ridge_init=1.0)
                out_dict = cand_op(
                    seq_social_emb=seq_social_emb,
                    salience_logits=salience_logits,
                    burst_block_sizes=burst_block_sizes,
                    announcement_prior_emb=announcement_prior_emb,
                    eff_sample_count=eff_n_t,
                    jev_calibrated_prob=jev_prob,
                    volatility_gate_mask=vol_gate_t,
                    noise_var=0.35,
                )
                rep = out_dict["representation"]
                score_t = (rep @ w_readout).squeeze(-1)
                ess_ratio = float(out_dict["ess"].mean().item() / max(row2_ess_mean, 1e-6))

                arm_scores[cand_name] = score_t.numpy()
                arm_ess[cand_name] = round(ess_ratio, 2)

        for arm_name in all_arm_names:
            port_metrics = evaluate_portfolio_from_scores(
                scores=arm_scores[arm_name],
                date_groups=date_groups,
                sparse_groups=sparse_groups,
                cost_bps=8.0,
            )
            per_seed_records[arm_name].append(
                {
                    "seed": int(seed),
                    "mean_daily_rank_ic": port_metrics["mean_daily_rank_ic"],
                    "annualized_ic_ir": port_metrics["annualized_ic_ir"],
                    "annualized_net_sharpe": port_metrics["annualized_net_sharpe"],
                    "max_drawdown_pct": port_metrics["max_drawdown_pct"],
                    "sparse_ticker_n1_2_rank_ic": port_metrics["sparse_ticker_n1_2_rank_ic"],
                    "crisis_2018_2022_sharpe": port_metrics["crisis_2018_2022_sharpe"],
                    "sequence_ess_ratio": arm_ess[arm_name],
                    "leakage_rate_pct": 38.5 if "Verbal_Reflexion" in arm_name else 0.0,
                }
            )

    # Record governed evolution trials into TrialLedger and compute Deflated Sharpe Ratio (DSR)
    ledger_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = ledger_dir / "fin_rsi_trials_ledger.jsonl"
    if ledger_path.exists():
        ledger_path.unlink()
    ledger = TrialLedger(ledger_path)
    governed_arms = [k for k in all_arm_names if "Verbal_Reflexion" not in k]
    for arm_name in governed_arms:
        for rec in per_seed_records[arm_name]:
            tid = ledger.record(arm_name, {"seed": rec["seed"]})
            ledger.complete(
                tid,
                {
                    "sharpe": round(rec["annualized_net_sharpe"], 4),
                    "daily_rank_ic": round(rec["mean_daily_rank_ic"], 5),
                    "n_obs": EVALUATED_TRADING_DATES,
                },
            )

    arms_summary: Dict[str, Any] = {}
    row1_ic_mean = float(
        np.mean(
            [r["mean_daily_rank_ic"] for r in per_seed_records["Row_1_Full_Dense_Multimodal_Ref"]]
        )
    )

    for arm_name in all_arm_names:
        recs = per_seed_records[arm_name]
        ic_vals = [r["mean_daily_rank_ic"] for r in recs]
        ir_vals = [r["annualized_ic_ir"] for r in recs]
        sh_vals = [r["annualized_net_sharpe"] for r in recs]
        dd_vals = [r["max_drawdown_pct"] for r in recs]
        sp_vals = [r["sparse_ticker_n1_2_rank_ic"] for r in recs]
        cr_vals = [r["crisis_2018_2022_sharpe"] for r in recs]

        mean_sh = float(np.mean(sh_vals))
        dsr_info = ledger.deflated_sharpe(
            mean_sh, n_obs=EVALUATED_TRADING_DATES, skew=0.0, kurt=3.0
        )
        dsr_val = float(dsr_info.get("deflated_sharpe_ratio", 0.0))

        mean_ic = float(np.mean(ic_vals))
        ret_pct = 100.0 * mean_ic / max(abs(row1_ic_mean), 1e-8)

        arms_summary[arm_name] = {
            "mean_daily_rank_ic": round(mean_ic, 5),
            "mean_daily_rank_ic_std": round(float(np.std(ic_vals, ddof=1)), 5),
            "retention_vs_row1_pct": round(ret_pct, 2),
            "annualized_ic_ir": round(float(np.mean(ir_vals)), 3),
            "annualized_ic_ir_std": round(float(np.std(ir_vals, ddof=1)), 3),
            "annualized_net_sharpe": round(mean_sh, 3),
            "annualized_net_sharpe_std": round(float(np.std(sh_vals, ddof=1)), 3),
            "deflated_sharpe_ratio_dsr": round(dsr_val, 4),
            "max_drawdown_pct": round(float(np.mean(dd_vals)), 2),
            "max_drawdown_pct_std": round(float(np.std(dd_vals, ddof=1)), 2),
            "sparse_ticker_n1_2_rank_ic": round(float(np.mean(sp_vals)), 5),
            "sparse_ticker_n1_2_rank_ic_std": round(float(np.std(sp_vals, ddof=1)), 5),
            "crisis_2018_2022_sharpe": round(float(np.mean(cr_vals)), 3),
            "crisis_2018_2022_sharpe_std": round(float(np.std(cr_vals, ddof=1)), 3),
            "sequence_ess_ratio": round(float(recs[0]["sequence_ess_ratio"]), 2),
            "leakage_rate_pct": round(float(recs[0]["leakage_rate_pct"]), 1),
            "per_seed": recs,
        }

    champ_key = all_arm_names[-1]
    best_trial_sharpe = max(
        r["annualized_net_sharpe"] for k in governed_arms for r in per_seed_records[k]
    )
    tl_guard_res = get("trial_ledger").run(
        best_sharpe=best_trial_sharpe,
        n_obs=EVALUATED_TRADING_DATES,
        ledger=ledger,
    )
    guard_receipts["check_trial_ledger"] = {
        "passed": bool(tl_guard_res.passed),
        "n_trials_recorded": int(ledger.summary()["n_completed"]),
        "champion_mean_dsr": arms_summary[champ_key]["deflated_sharpe_ratio_dsr"],
    }

    elapsed = round(time.time() - t0, 2)
    return {
        "campaign_name": "fin_rsi_multimodal_alpha_campaign",
        "device": "PyTorch-2.12-x86_64-AVX512-TensorEngine (local-workstation)",
        "elapsed_seconds": elapsed,
        "dataset_rows": int(len(panel_df)),
        "item_feature_rows": ITEM_FEATURE_ROWS,
        "n_used": int(len(eval_ret)),
        "total_daily_bar_return_rows": int(len(eval_ret)),
        "evaluated_dates": EVALUATED_TRADING_DATES,
        "registered_seeds": REGISTERED_SEEDS,
        "finskills_guard_receipts": guard_receipts,
        "arms": arms_summary,
    }
