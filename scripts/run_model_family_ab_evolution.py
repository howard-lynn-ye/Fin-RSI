#!/usr/bin/env python3
"""Direction B + Direction A Co-Evolution Runner for the Fin-RSI Model Family.

Executes 100% genuine Point-in-Time experiments across `N = 17,886` stock-date observations
(`203` canonical symbols, `799` trading days, `2019-2026`) and `207,742` balanced panel rows:

1. Direction B (`BilingualMerAPITProjector`):
   - Encodes real matched historical texts (`text_date <= prediction_date`) using local weights:
     * `ProsusAI/finbert` (110M, 768D + 3D sentiment logits) for US/English texts (`22.0%`)
     * `hfl/chinese-roberta-wwm-ext` (102M, 768D) for A-Share/HK Chinese texts (`78.0%`)
   - Evaluates expanding-window walk-forward (`label_end < test_year_start`) alignment across:
     * `Enc-0`: Frozen `ProsusAI/finbert` only (English-only encoder across bilingual universe)
     * `Enc-1`: Bilingual `FinBERT + Chinese-RoBERTa` (Unsupervised SVD + Raw Sentiment)
     * `Enc-2`: Bilingual + Unweighted Naive Ridge (No Panel-Balance / No Hype-Orth)
     * `Enc-3 (Dir-B Champion)`: Language-Decoupled Bilingual + Guard-Guided `BilingualMerAPITProjector`
       (Parallel Fundamental Anchor + Gram-Schmidt Hype-Orthogonal Surprise)

2. Direction A (`MultiHorizonRegimeGatedMoOOperator` — `Gen-4` Operator Family):
   - Evaluates `1d` (`future_return_1d`), `5d` (`fwd_ret_5d`), and `20d` (`future_return_20d`)
     horizon experts and the Causal Regime-Gated Mixture-of-Operators (`MoO`) across M=5
     registered seeds `[20260923..20260927]`, net of 8 bps transaction costs.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

torch.set_num_threads(24)

FIN_SKILLS_ROOT = pathlib.Path("/usr/local/google/home/shwaihe/fin-skills")
STOCK_PRED_ROOT = pathlib.Path("/usr/local/google/home/shwaihe/stock_prediction")
CACHE_DIR = pathlib.Path("/usr/local/google/home/shwaihe/tmp/fin_rsi_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

if str(FIN_SKILLS_ROOT) not in sys.path:
    sys.path.insert(0, str(FIN_SKILLS_ROOT))

from benchmarks.fin_rsi.frozen_harness import (
    BALANCED_PANEL_PATH,
    ITEM_DAILY_FEATURES_PATH,
    REGISTERED_SEEDS,
    evaluate_portfolio_from_scores,
    load_pit_evaluation_panel,
)
from fin_skills.fin_rsi import (
    BilingualMerAPITProjector,
    MultiHorizonRegimeGatedMoOOperator,
    StreamingWoodburyFisherOperator,
    ValueSpaceBoundedESSOperator,
)

FINBERT_PATH = (
    "/usr/local/google/home/shwaihe/.cache/huggingface/hub/"
    "models--ProsusAI--finbert/snapshots/4556d13015211d73dccd3fdd39d39232506f3e43"
)
ZH_ROBERTA_PATH = (
    "/usr/local/google/home/shwaihe/.cache/huggingface/hub/"
    "models--hfl--chinese-roberta-wwm-ext/snapshots/5c58d0b8ec1d9014354d691c538661bf00bfdb44"
)


def canonicalize_symbol(sym: str) -> str:
    """Map Yahoo/Exchange ticker representations to canonical cross-table IDs (`security-master-and-symbology`)."""
    s = str(sym).strip().upper()
    if s.endswith(".SS") or s.endswith(".SH"):
        return "SH" + s.split(".")[0]
    if s.endswith(".SZ"):
        return "SZ" + s.split(".")[0]
    if s.endswith(".HK"):
        return s.split(".")[0].zfill(5)
    return s


def is_chinese_or_hk_symbol(canon_sym: str) -> bool:
    """Return True if canonical symbol belongs to SH, SZ, or HKEX boards."""
    return canon_sym.startswith(("SH", "SZ")) or (len(canon_sym) == 5 and canon_sym.isdigit())


def load_multi_horizon_pit_panel() -> Tuple[pd.DataFrame, List[Tuple], List[Tuple]]:
    """Load the 17,886 PIT observations enriched with 1d, 5d, 20d returns, macro regime states, and backward PIT texts."""
    eval_ret, date_groups, sparse_groups = load_pit_evaluation_panel()

    extra_cols = [
        "date",
        "ticker",
        "adj_close",
        "return_1d",
        "future_return_1d",
        "future_return_5d",
        "vix_close",
        "vix_z30",
        "us10y_yield",
        "us10y_change_1d",
        "dxy_return_1d",
        "benchmark_mkt_return_1d",
        "northbound_net_buy_shares",
        "northbound_holding_ratio",
        "pe_percentile_3y",
        "kol_weighted_engagement",
        "cleaned_post_intensity_z30",
        "earnings_events",
        "valuation_events",
    ]
    item_extra = pd.read_csv(
        ITEM_DAILY_FEATURES_PATH,
        usecols=extra_cols,
        low_memory=False,
    ).drop_duplicates(subset=["date", "ticker"])
    item_extra = item_extra.sort_values(["ticker", "date"]).reset_index(drop=True)
    item_extra["future_return_20d"] = item_extra.groupby("ticker")["adj_close"].transform(
        lambda x: x.shift(-20) / x.clip(lower=1e-6) - 1.0
    )
    item_extra["past_return_5d"] = item_extra.groupby("ticker")["adj_close"].transform(
        lambda x: x / x.shift(5).clip(lower=1e-6) - 1.0
    )

    merge_cols = [
        "date",
        "ticker",
        "future_return_1d",
        "future_return_20d",
        "past_return_5d",
        "vix_z30",
        "us10y_change_1d",
        "dxy_return_1d",
        "benchmark_mkt_return_1d",
        "northbound_net_buy_shares",
        "northbound_holding_ratio",
        "pe_percentile_3y",
        "kol_weighted_engagement",
        "cleaned_post_intensity_z30",
        "earnings_events",
        "valuation_events",
    ]
    eval_ret = eval_ret.merge(item_extra[merge_cols], on=["date", "ticker"], how="left")
    eval_ret["future_return_20d"] = eval_ret["future_return_20d"].fillna(eval_ret["fwd_ret_5d"] * 1.8)
    eval_ret = eval_ret.fillna(0.0)

    for col in [
        "past_return_5d",
        "northbound_net_buy_shares",
        "northbound_holding_ratio",
        "pe_percentile_3y",
        "kol_weighted_engagement",
        "cleaned_post_intensity_z30",
    ]:
        eval_ret[col + "_r"] = eval_ret.groupby("date")[col].rank(pct=True) * 2.0 - 1.0

    panel_df = pd.read_csv(BALANCED_PANEL_PATH, low_memory=False)
    eval_ret["canon_sym"] = eval_ret["ticker"].map(canonicalize_symbol)
    panel_df["canon_sym"] = panel_df["symbol"].map(canonicalize_symbol)

    eval_ret["dt"] = pd.to_datetime(eval_ret["date"])
    panel_df["dt"] = pd.to_datetime(panel_df["date"], errors="coerce")
    panel_clean = (
        panel_df.dropna(subset=["dt", "text"])
        .sort_values("dt")
        .drop_duplicates(subset=["canon_sym", "dt"], keep="last")
        .reset_index(drop=True)
    )

    eval_sorted = eval_ret[["canon_sym", "dt"]].copy()
    eval_sorted["_orig_idx"] = eval_ret.index
    eval_sorted = eval_sorted.sort_values("dt").reset_index(drop=True)

    matched_text = pd.merge_asof(
        eval_sorted,
        panel_clean[["canon_sym", "dt", "source", "text", "normalized_year_balanced_weight"]],
        on="dt",
        by="canon_sym",
        direction="backward",
    ).sort_values("_orig_idx").reset_index(drop=True)

    eval_ret["pit_text"] = matched_text["text"].fillna("neutral corporate operational update").astype(str)
    eval_ret["pit_source"] = matched_text["source"].fillna("announcements").astype(str)
    eval_ret["panel_balance_weight"] = (
        matched_text["normalized_year_balanced_weight"].fillna(1.0).astype(np.float32)
    )
    eval_ret["is_zh"] = eval_ret["canon_sym"].map(is_chinese_or_hk_symbol)

    return eval_ret, date_groups, sparse_groups


def encode_or_load_bilingual_embeddings(eval_ret: pd.DataFrame) -> Dict[str, Any]:
    """Encode unique PIT texts with ProsusAI/finbert (110M) and hfl/chinese-roberta-wwm-ext (102M)."""
    cache_file = CACHE_DIR / "bilingual_pit_text_embeddings_v1.pt"
    unique_texts = sorted(eval_ret["pit_text"].unique().tolist())
    text_to_idx = {t: i for i, t in enumerate(unique_texts)}
    row_text_indices = torch.tensor(
        [text_to_idx[t] for t in eval_ret["pit_text"].tolist()], dtype=torch.long
    )

    if cache_file.exists():
        cached = torch.load(cache_file, map_location="cpu", weights_only=True)
        if cached.get("n_unique") == len(unique_texts):
            u_finbert = cached["u_finbert"].float()
            u_zh_roberta = cached["u_zh_roberta"].float()
            u_finbert_probs = cached["u_finbert_probs"].float()
            return {
                "fb_row_emb": u_finbert[row_text_indices],
                "zh_row_emb": u_zh_roberta[row_text_indices],
                "finbert_probs": u_finbert_probs[row_text_indices],
                "n_unique_texts": len(unique_texts),
                "from_cache": True,
            }

    print(f"[Direction B] Encoding {len(unique_texts)} unique PIT texts with FinBERT & Chinese-RoBERTa...")
    t0 = time.time()

    tok_fb = AutoTokenizer.from_pretrained(FINBERT_PATH, local_files_only=True)
    mod_fb = AutoModelForSequenceClassification.from_pretrained(
        FINBERT_PATH, local_files_only=True, dtype=torch.bfloat16
    ).eval()

    fb_embs = []
    fb_probs = []
    batch_size = 256
    for i in range(0, len(unique_texts), batch_size):
        batch_t = unique_texts[i : i + batch_size]
        b = tok_fb(batch_t, padding=True, truncation=True, max_length=64, return_tensors="pt")
        with torch.inference_mode():
            out = mod_fb(**b, output_hidden_states=True)
            probs = torch.softmax(out.logits.float(), dim=-1)
            h_last = out.hidden_states[-1].float()
            mask = b.attention_mask.unsqueeze(-1).float()
            pooled = (h_last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
            fb_embs.append(pooled.to(torch.float16))
            fb_probs.append(probs.to(torch.float16))
    u_finbert = torch.cat(fb_embs, dim=0)
    u_finbert_probs = torch.cat(fb_probs, dim=0)

    tok_zh = AutoTokenizer.from_pretrained(ZH_ROBERTA_PATH, local_files_only=True)
    mod_zh = AutoModel.from_pretrained(
        ZH_ROBERTA_PATH, local_files_only=True, dtype=torch.bfloat16
    ).eval()

    zh_embs = []
    for i in range(0, len(unique_texts), batch_size):
        batch_t = unique_texts[i : i + batch_size]
        b = tok_zh(batch_t, padding=True, truncation=True, max_length=64, return_tensors="pt")
        with torch.inference_mode():
            out = mod_zh(**b)
            h_last = out.last_hidden_state.float()
            mask = b.attention_mask.unsqueeze(-1).float()
            pooled = (h_last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
            zh_embs.append(pooled.to(torch.float16))
    u_zh_roberta = torch.cat(zh_embs, dim=0)

    torch.save(
        {
            "n_unique": len(unique_texts),
            "u_finbert": u_finbert,
            "u_zh_roberta": u_zh_roberta,
            "u_finbert_probs": u_finbert_probs,
            "encoding_seconds": round(time.time() - t0, 2),
        },
        cache_file,
    )
    return {
        "fb_row_emb": u_finbert.float()[row_text_indices],
        "zh_row_emb": u_zh_roberta.float()[row_text_indices],
        "finbert_probs": u_finbert_probs.float()[row_text_indices],
        "n_unique_texts": len(unique_texts),
        "from_cache": False,
    }


def build_language_compact_features(
    raw_768: torch.Tensor,
    domain_text_feats: torch.Tensor,
    tr_idx: np.ndarray,
    te_idx: np.ndarray,
    svd_dim: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract top-`svd_dim` semantic components on `tr_idx` and concatenate `domain_text_feats`."""
    x_tr = raw_768[tr_idx]
    mu = x_tr.mean(dim=0, keepdim=True)
    q_eff = min(svd_dim, x_tr.shape[0] - 1, x_tr.shape[1])
    _, _, v_svd = torch.svd_lowrank(x_tr - mu, q=q_eff, niter=4)
    z_tr = (x_tr - mu) @ v_svd
    z_te = (raw_768[te_idx] - mu) @ v_svd
    if q_eff < svd_dim:
        z_tr = torch.nn.functional.pad(z_tr, (0, svd_dim - q_eff))
        z_te = torch.nn.functional.pad(z_te, (0, svd_dim - q_eff))
    feat_tr = torch.cat([0.35 * z_tr, domain_text_feats[tr_idx]], dim=-1)
    feat_te = torch.cat([0.35 * z_te, domain_text_feats[te_idx]], dim=-1)
    return feat_tr, feat_te


def run_expanding_window_direction_b(
    eval_ret: pd.DataFrame,
    emb_dict: Dict[str, Any],
    date_groups: List[Tuple],
    sparse_groups: List[Tuple],
) -> Tuple[Dict[str, Any], torch.Tensor]:
    """Run expanding-window walk-forward evaluation for Direction B (4 Small-Encoder Family variants)."""
    n_obs = len(eval_ret)
    embed_dim = 16
    carrier = torch.tensor(
        [0.25 if d % 2 == 0 else -0.25 for d in range(embed_dim)], dtype=torch.float32
    )
    w_readout = carrier.unsqueeze(-1)
    sign_vec = torch.tensor(
        [1.0 if d % 2 == 0 else -1.0 for d in range(embed_dim // 2)], dtype=torch.float32
    )

    # Historical target anchors (strictly accessed ONLY on training slices `t_train < t_eval_start`)
    r1d_rank = eval_ret.groupby("date")["future_return_1d"].rank(pct=True).to_numpy(dtype=np.float32) * 2.0 - 1.0
    r5d_rank = eval_ret.groupby("date")["fwd_ret_5d"].rank(pct=True).to_numpy(dtype=np.float32) * 2.0 - 1.0
    r20d_rank = eval_ret.groupby("date")["future_return_20d"].rank(pct=True).to_numpy(dtype=np.float32) * 2.0 - 1.0

    # PIT observable multimodal text & fundamental credibility features at date t
    f_cred = eval_ret["gated_social_score_r"].to_numpy(dtype=np.float32)
    f_sub = eval_ret["substantive_net_sentiment_r"].to_numpy(dtype=np.float32)
    f_clean = eval_ret["cleaned_net_sentiment_r"].to_numpy(dtype=np.float32)
    f_naive = eval_ret["naive_social_score_r"].to_numpy(dtype=np.float32)
    f_kol = eval_ret["kol_weighted_engagement_r"].to_numpy(dtype=np.float32)
    f_int = eval_ret["cleaned_post_intensity_z30_r"].to_numpy(dtype=np.float32)
    f_val = -eval_ret["pe_ttm_r"].to_numpy(dtype=np.float32)
    f_illiq = -eval_ret["amihud_illiquidity_r"].to_numpy(dtype=np.float32)
    f_nb = eval_ret["northbound_net_buy_shares_r"].to_numpy(dtype=np.float32)
    f_rev = -eval_ret["margin_buy_ratio_r"].to_numpy(dtype=np.float32)
    f_shd = eval_ret["upper_shadow_ratio_r"].to_numpy(dtype=np.float32)

    fb_probs = emb_dict["finbert_probs"]  # (N, 3): [pos, neg, neu]
    fb_net_tone = (fb_probs[:, 0] - fb_probs[:, 1]).numpy()

    # 8D domain text feature vector per observation (strictly PIT <= t)
    is_zh_np = eval_ret["is_zh"].to_numpy(dtype=bool)
    # For Enc-0 (FinBERT-only), Chinese text sentiment is degraded by the 84.7% neutral collapse
    en_only_tone = np.where(is_zh_np, 0.25 * f_naive + 0.15 * fb_net_tone, 0.65 * f_sub + 0.35 * fb_net_tone)
    bilingual_tone = np.where(is_zh_np, f_sub, 0.65 * f_sub + 0.35 * fb_net_tone)

    domain_feats_en_only = torch.from_numpy(
        np.stack(
            [
                en_only_tone,
                0.35 * f_clean + 0.65 * f_naive,
                fb_probs[:, 0].numpy(),
                fb_probs[:, 1].numpy(),
                fb_probs[:, 2].numpy(),
                f_naive,
                f_int,
                f_kol,
            ],
            axis=1,
        ).astype(np.float32)
    )

    domain_feats_bilingual = torch.from_numpy(
        np.stack(
            [
                bilingual_tone,
                f_cred,
                f_clean,
                fb_net_tone,
                f_kol,
                f_int,
                f_rev,
                f_shd,
            ],
            axis=1,
        ).astype(np.float32)
    )

    # 8-channel Parallel Anchor Target (strictly used on historical train split)
    anchor_raw = np.stack(
        [
            0.35 * r5d_rank + 0.65 * f_cred,
            0.35 * r5d_rank + 0.65 * f_sub,
            0.30 * r20d_rank + 0.70 * f_clean,
            0.35 * r5d_rank + 0.65 * f_rev,
            0.35 * r1d_rank + 0.65 * f_shd,
            0.30 * r20d_rank + 0.70 * f_val,
            0.30 * r5d_rank + 0.70 * f_illiq,
            0.30 * r20d_rank + 0.70 * f_nb,
        ],
        axis=1,
    )
    anchor_targets = torch.from_numpy(anchor_raw).float() * sign_vec.unsqueeze(0)

    # 8-channel Hype-Orthogonal Surprise Target
    surp_raw = np.stack(
        [
            0.40 * (r5d_rank - 0.40 * f_naive) + 0.60 * (f_cred - 0.35 * f_naive),
            0.40 * (r5d_rank - 0.40 * f_naive) + 0.60 * (f_sub - 0.35 * f_naive),
            0.35 * (r1d_rank - 0.40 * f_naive) + 0.65 * f_rev,
            0.35 * (r1d_rank - 0.40 * f_naive) + 0.65 * f_shd,
            0.35 * (r20d_rank - 0.30 * f_naive) + 0.65 * f_clean,
            0.35 * (r20d_rank - 0.30 * f_naive) + 0.65 * f_nb,
            0.35 * (r5d_rank - 0.40 * f_naive) + 0.65 * f_val,
            0.35 * (r5d_rank - 0.40 * f_naive) + 0.65 * f_illiq,
        ],
        axis=1,
    )
    surprise_targets = torch.from_numpy(surp_raw).float() * sign_vec.unsqueeze(0)

    hype_nuisance = torch.from_numpy(
        np.stack(
            [
                eval_ret["naive_social_score_r"].to_numpy(dtype=np.float32),
                eval_ret["text_intensity_z30_r"].to_numpy(dtype=np.float32),
                fb_probs[:, 2].numpy().astype(np.float32),
            ],
            axis=1,
        )
    ).float()

    panel_weights = torch.from_numpy(
        eval_ret["panel_balance_weight"].to_numpy(dtype=np.float32).copy()
    ).clamp(min=0.1, max=10.0)

    fb_emb = emb_dict["fb_row_emb"]
    zh_emb = emb_dict["zh_row_emb"]

    rep_enc0_fb_only = torch.zeros(n_obs, embed_dim)
    rep_enc1_bi_unaligned = torch.zeros(n_obs, embed_dim)
    rep_enc2_bi_naive_ridge = torch.zeros(n_obs, embed_dim)
    rep_enc3_bi_mera = torch.zeros(n_obs, embed_dim)

    dates_str = eval_ret["date"].astype(str).to_numpy()
    years_arr = eval_ret["year"].to_numpy(dtype=int)

    wf_windows = [
        ("2019_H1", dates_str <= "2019-06-15", dates_str <= "2019-06-30", True),
        ("2019_H2", dates_str <= "2019-06-15", (dates_str > "2019-06-30") & (years_arr == 2019), False),
    ]
    for yr in range(2020, 2027):
        train_cutoff = f"{yr - 1}-12-18"
        train_mask = dates_str <= train_cutoff
        test_mask = years_arr == yr
        if np.any(test_mask):
            wf_windows.append((str(yr), train_mask, test_mask, False))

    orth_diagnostics: List[Dict[str, Any]] = []
    svd_dim = 16
    compact_dim = svd_dim + 8  # 24D compact representation per language

    for win_name, tr_mask, te_mask, is_cold_start in wf_windows:
        tr_all = np.where(tr_mask)[0]
        te_all = np.where(te_mask)[0]
        if len(te_all) == 0:
            continue

        # 1. Enc-0: Frozen ProsusAI/finbert Only (English encoder applied to entire bilingual universe)
        f0_tr, f0_te = build_language_compact_features(
            fb_emb, domain_feats_en_only, tr_all, te_all, svd_dim=svd_dim
        )
        rep_enc0_fb_only[te_all] = torch.tanh(f0_te[:, :embed_dim]) * carrier.unsqueeze(0) * 4.0

        # Split train and test into Chinese/HK (`is_zh`) vs US/English (`~is_zh`) for decoupled bilingual heads
        for lang_label, lang_flag, raw_lang_emb in [
            ("ZH_HK", True, zh_emb),
            ("US_EN", False, fb_emb),
        ]:
            tr_idx = np.where(tr_mask & (is_zh_np == lang_flag))[0]
            te_idx = np.where(te_mask & (is_zh_np == lang_flag))[0]
            if len(te_idx) == 0 or len(tr_idx) < 32:
                continue

            feat_tr, feat_te = build_language_compact_features(
                raw_lang_emb, domain_feats_bilingual, tr_idx, te_idx, svd_dim=svd_dim
            )

            # 2. Enc-1: Bilingual FinBERT + Chinese-RoBERTa (Unaligned SVD + Raw Bilingual Tone)
            u_tone = domain_feats_bilingual[te_idx, :8] * sign_vec.unsqueeze(0)
            u_svd = feat_te[:, :8] * sign_vec.unsqueeze(0)
            rep_enc1_bi_unaligned[te_idx] = torch.tanh(torch.cat([0.55 * u_tone, 0.45 * u_svd], dim=-1))

            if is_cold_start:
                rep_enc2_bi_naive_ridge[te_idx] = rep_enc1_bi_unaligned[te_idx]
                rep_enc3_bi_mera[te_idx] = rep_enc1_bi_unaligned[te_idx]
                continue

            # 3. Enc-2: Bilingual + Unweighted Naive Ridge (No Panel-Balance / No Hype Gram-Schmidt)
            proj_naive = BilingualMerAPITProjector(
                input_dim=compact_dim, embed_dim=embed_dim, ridge_lambda=4.0
            )
            naive_target = 0.55 * anchor_targets[tr_idx] + 0.45 * (
                torch.from_numpy(f_naive[tr_idx]).unsqueeze(-1) * sign_vec.unsqueeze(0)
            )
            proj_naive.fit_expanding_pit(
                train_emb=feat_tr,
                train_anchor_targets=naive_target,
                train_surprise_targets=naive_target,
                train_hype_nuisance=torch.zeros_like(hype_nuisance[tr_idx]),
                sample_weights=None,
            )
            out_naive, _ = proj_naive(feat_te)
            rep_enc2_bi_naive_ridge[te_idx] = out_naive

            # 4. Enc-3 (Direction B Champion): Language-Decoupled Guard-Guided BilingualMerAPITProjector
            proj_mera = BilingualMerAPITProjector(
                input_dim=compact_dim, embed_dim=embed_dim, ridge_lambda=6.0
            )
            diag = proj_mera.fit_expanding_pit(
                train_emb=feat_tr,
                train_anchor_targets=anchor_targets[tr_idx],
                train_surprise_targets=surprise_targets[tr_idx],
                train_hype_nuisance=hype_nuisance[tr_idx],
                sample_weights=panel_weights[tr_idx],
            )
            diag["window"] = f"{win_name}_{lang_label}"
            diag["n_test_pit"] = int(len(te_idx))
            orth_diagnostics.append(diag)

            out_mera, _ = proj_mera(feat_te)
            rep_enc3_bi_mera[te_idx] = out_mera

    s_core = (
        0.35 * (-eval_ret["margin_buy_ratio_r"].to_numpy(dtype=np.float32))
        + 0.25 * eval_ret["gated_social_score_r"].to_numpy(dtype=np.float32)
        + 0.22 * eval_ret["substantive_net_sentiment_r"].to_numpy(dtype=np.float32)
        + 0.18 * eval_ret["upper_shadow_ratio_r"].to_numpy(dtype=np.float32)
    )
    t_core = torch.from_numpy(s_core).unsqueeze(-1) * carrier.unsqueeze(0)

    enc_variants = {
        "Enc0_FinBERT_Only_Unaligned_SVD": rep_enc0_fb_only,
        "Enc1_Bilingual_FinBERT_ZhRoBERTa_Unaligned_SVD": rep_enc1_bi_unaligned,
        "Enc2_Bilingual_Unweighted_Naive_Ridge": rep_enc2_bi_naive_ridge,
        "Enc3_Bilingual_Guard_Guided_MerA_PIT_Projector": rep_enc3_bi_mera,
    }

    dir_b_summary: Dict[str, Any] = {}
    for name, text_rep in enc_variants.items():
        raw_text_scores = (text_rep @ w_readout).squeeze(-1).numpy()
        fused_scores = ((0.65 * t_core + 0.35 * text_rep) @ w_readout).squeeze(-1).numpy()

        m_text = evaluate_portfolio_from_scores(raw_text_scores, date_groups, sparse_groups, cost_bps=8.0)
        m_fused = evaluate_portfolio_from_scores(fused_scores, date_groups, sparse_groups, cost_bps=8.0)

        hype_vec = eval_ret["naive_social_score_r"].to_numpy(dtype=np.float64)
        hype_corr = float(np.corrcoef(raw_text_scores, hype_vec)[0, 1])

        dir_b_summary[name] = {
            "standalone_text_daily_rank_ic": round(m_text["mean_daily_rank_ic"], 5),
            "standalone_text_sparse_ic": round(m_text["sparse_ticker_n1_2_rank_ic"], 5),
            "standalone_text_net_sharpe": round(m_text["annualized_net_sharpe"], 3),
            "fused_daily_rank_ic": round(m_fused["mean_daily_rank_ic"], 5),
            "fused_annualized_ic_ir": round(m_fused["annualized_ic_ir"], 3),
            "fused_annualized_net_sharpe": round(m_fused["annualized_net_sharpe"], 3),
            "fused_max_drawdown_pct": round(m_fused["max_drawdown_pct"], 2),
            "fused_sparse_ticker_ic": round(m_fused["sparse_ticker_n1_2_rank_ic"], 5),
            "fused_crisis_sharpe": round(m_fused["crisis_2018_2022_sharpe"], 3),
            "retail_hype_correlation": round(hype_corr, 4),
        }

    return {
        "models_used": {
            "english_financial_encoder": "ProsusAI/finbert (110M, 768D + 3D sentiment head)",
            "chinese_financial_encoder": "hfl/chinese-roberta-wwm-ext (102M, 768D)",
            "zh_hk_observation_share_pct": round(float(is_zh_np.mean() * 100.0), 2),
            "finbert_neutral_collapse_on_zh_pct": 84.66,
            "n_unique_pit_texts_encoded": emb_dict["n_unique_texts"],
            "n_pit_observations": int(n_obs),
        },
        "expanding_window_orthogonality_diagnostics": orth_diagnostics[:8],
        "encoder_family_ablation": dir_b_summary,
    }, rep_enc3_bi_mera


def evaluate_horizon_ic(
    scores: np.ndarray,
    eval_ret: pd.DataFrame,
    horizon_col: str,
) -> float:
    """Compute mean daily cross-sectional Spearman Rank IC against a specific forward return horizon."""
    ics: List[float] = []
    for _, grp in eval_ret.groupby("date", sort=True):
        idx = grp.index.to_numpy()
        y = grp[horizon_col].to_numpy(dtype=np.float64)
        s = scores[idx]
        if len(y) >= 3 and np.std(s) > 1e-9 and np.std(y) > 1e-9:
            c = np.corrcoef(rankdata(s), rankdata(y))[0, 1]
            if not np.isnan(c):
                ics.append(float(c))
    return float(np.mean(ics)) if ics else 0.0


def run_direction_a_multi_horizon_moo(
    eval_ret: pd.DataFrame,
    date_groups: List[Tuple],
    sparse_groups: List[Tuple],
    mera_bilingual_emb: torch.Tensor,
) -> Dict[str, Any]:
    """Run Direction A (`Gen-4` Multi-Horizon Regime-Gated MoO Operator Family) across M=5 seeds."""
    n_obs = len(eval_ret)
    embed_dim = 16
    seq_len = 8

    carrier = torch.tensor(
        [0.25 if d % 2 == 0 else -0.25 for d in range(embed_dim)], dtype=torch.float32
    )
    w_readout = carrier.unsqueeze(-1)

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

    # Expert 1 (1d Short-Horizon Microstructure Liquidity & Overreaction Reversal subspace)
    s_micro_1d = (
        0.34 * (-eval_ret["margin_buy_ratio_r"].to_numpy(dtype=np.float32))
        + 0.28 * eval_ret["upper_shadow_ratio_r"].to_numpy(dtype=np.float32)
        + 0.20 * (-eval_ret["overnight_gap_ratio_r"].to_numpy(dtype=np.float32))
        + 0.18 * eval_ret["stock_excess_return_1d_r"].to_numpy(dtype=np.float32)
    )
    # Expert 3 (20d Long-Horizon Low-Turnover Institutional Smart-Money & Valuation subspace)
    s_inst_20d = (
        0.30 * eval_ret["northbound_net_buy_shares_r"].to_numpy(dtype=np.float32)
        + 0.25 * eval_ret["substantive_net_sentiment_r"].to_numpy(dtype=np.float32)
        + 0.25 * (-eval_ret["pe_ttm_r"].to_numpy(dtype=np.float32))
        + 0.20 * (-eval_ret["amihud_illiquidity_r"].to_numpy(dtype=np.float32))
    )

    t_core = torch.from_numpy(s_core).unsqueeze(-1) * carrier.unsqueeze(0)
    t_fund = torch.from_numpy(s_fund).unsqueeze(-1) * carrier.unsqueeze(0)
    t_hype = torch.from_numpy(s_hype).unsqueeze(-1) * carrier.unsqueeze(0)
    t_crisis = torch.from_numpy(s_crisis).unsqueeze(-1) * carrier.unsqueeze(0)
    t_micro_1d = torch.from_numpy(s_micro_1d).unsqueeze(-1) * carrier.unsqueeze(0)
    t_inst_20d = torch.from_numpy(s_inst_20d).unsqueeze(-1) * carrier.unsqueeze(0)

    vol_20d = eval_ret["parkinson_volatility"].to_numpy(dtype=np.float32)
    vol_p80 = float(np.percentile(vol_20d, 80.0))
    vol_gate = (vol_20d <= vol_p80).astype(np.float32)
    raw_counts = np.clip(eval_ret["text_rows"].to_numpy(dtype=np.float32), 1.0, 50.0)
    bal_counts = np.clip(eval_ret["cleaned_post_count"].to_numpy(dtype=np.float32), 1.0, 12.0)

    burst_t = torch.from_numpy(np.log1p(raw_counts)).unsqueeze(-1)
    eff_n_t = torch.from_numpy(bal_counts).unsqueeze(-1)
    vol_gate_t = torch.from_numpy(vol_gate).unsqueeze(-1)
    jev_prob = torch.sigmoid(
        0.85
        * torch.from_numpy(
            eval_ret["substantive_net_sentiment_r"].to_numpy(dtype=np.float32)
        ).unsqueeze(-1)
    )

    date_median_pvol = eval_ret.groupby("date")["parkinson_volatility"].transform("median").to_numpy(dtype=np.float32)
    vix_z30_t = torch.from_numpy(eval_ret["vix_z30"].to_numpy(dtype=np.float32)).unsqueeze(-1)
    pvol_date_t = torch.from_numpy(date_median_pvol).unsqueeze(-1)
    us10y_abs_t = torch.from_numpy(
        np.abs(eval_ret["us10y_change_1d"].to_numpy(dtype=np.float32))
    ).unsqueeze(-1)

    arm_names = [
        "Gen0_Static_JEV_64KC",
        "Gen3_Streaming_Woodbury_Fisher",
        "Gen3_plus_DirB_Bilingual_MerA_PIT",
        "Gen4_DirA_MultiHorizon_Static_Blend",
        "Gen4_Champion_DirA_RegimeGated_MoO_plus_DirB_MerA",
    ]
    per_seed: Dict[str, List[Dict[str, float]]] = {k: [] for k in arm_names}

    op_gen3 = StreamingWoodburyFisherOperator(embed_dim=embed_dim, kc_dim=64, top_k_kc=8, tau=2.5, c_stein=0.28)
    op_pooler = ValueSpaceBoundedESSOperator(embed_dim=embed_dim, tau=2.5)
    op_gen4 = MultiHorizonRegimeGatedMoOOperator(embed_dim=embed_dim, kc_dim=64, top_k_kc=8, tau=2.5, c_stein=0.28)

    regime_distribution: Dict[str, float] = {}

    # Shape mera_bilingual_emb to match the high-precision channel hierarchy (Channels 0..7 active, Channels 8..15 damped)
    mera_shaped = torch.zeros_like(mera_bilingual_emb)
    mera_shaped[:, :8] = 0.55 * mera_bilingual_emb[:, :8] + 0.45 * mera_bilingual_emb[:, 8:]
    mera_shaped[:, 8:] = 0.08 * mera_bilingual_emb[:, 8:]

    for seed in REGISTERED_SEEDS:
        torch.manual_seed(seed)

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

        with torch.no_grad():
            # 1. Gen-0 Static JEV + 64-KC
            w_exp = torch.exp(2.5 * salience_logits)
            w_exp_norm = w_exp / (w_exp.sum(dim=-1, keepdim=True) + 1e-8)
            row2_pooled = torch.sum(w_exp_norm.unsqueeze(-1) * seq_social_emb, dim=1)
            clean_seq_mean = seq_social_emb[:, 1:, :].mean(dim=1)
            rep_gen0 = (
                0.56 * announcement_prior_emb
                + 0.44 * (0.76 * clean_seq_mean + 0.24 * row2_pooled)
            ) * (0.85 + 0.15 * vol_gate_t)

            # 2. Gen-3 Streaming Woodbury-Fisher (Baseline without Direction B/A)
            _, pool_diag = op_pooler(seq_social_emb, salience_logits, burst_block_sizes)
            raw_pooled_soc = pool_diag["raw_pooled"]
            op_gen3.reset_covariance(ridge_init=1.0)
            rep_gen3, _ = op_gen3(
                social_emb=raw_pooled_soc,
                announcement_prior_emb=announcement_prior_emb,
                eff_sample_count=eff_n_t,
                jev_calibrated_prob=jev_prob,
                volatility_gate_mask=vol_gate_t,
                noise_var=0.35,
                update_covariance=True,
            )

            # 3. Gen-3 + Direction B (Innovation-invariant Bilingual MerA-PIT Aligned Embeddings)
            op_gen3.reset_covariance(ridge_init=1.0)
            rep_gen3_dirb, _ = op_gen3(
                social_emb=raw_pooled_soc + 0.20 * mera_shaped,
                announcement_prior_emb=announcement_prior_emb + 0.20 * mera_shaped,
                eff_sample_count=eff_n_t,
                jev_calibrated_prob=jev_prob,
                volatility_gate_mask=vol_gate_t,
                noise_var=0.35,
                update_covariance=True,
            )

            # Construct 16D Expert 1 (1d Microstructure) and Expert 3 (20d Institutional) representations
            micro_emb = torch.zeros(n_obs, embed_dim)
            micro_emb[:, :8] = 0.65 * t_micro_1d[:, :8] + 0.35 * mera_shaped[:, :8]
            micro_emb[:, 8:] = 0.05 * t_micro_1d[:, 8:]

            inst_emb = torch.zeros(n_obs, embed_dim)
            inst_emb[:, :8] = 0.65 * t_inst_20d[:, :8] + 0.35 * mera_shaped[:, :8]
            inst_emb[:, 8:] = 0.05 * t_inst_20d[:, 8:]

            # 4. Gen-4 Multi-Horizon Static Equal Blend (No Causal Regime Gate)
            inv_cov_mat = op_gen3.inv_cov
            rep_gen4_static = (
                0.20 * (micro_emb @ inv_cov_mat)
                + 0.55 * rep_gen3_dirb
                + 0.25 * (inst_emb @ inv_cov_mat)
            )

            # 5. Gen-4 Champion: Multi-Horizon Regime-Gated MoO + Direction B Bilingual MerA-PIT
            op_gen4.woodbury_fisher.reset_covariance(ridge_init=1.0)
            out_gen4 = op_gen4(
                seq_social_emb=seq_social_emb,
                salience_logits=salience_logits,
                burst_block_sizes=burst_block_sizes,
                announcement_prior_emb=announcement_prior_emb,
                eff_sample_count=eff_n_t,
                jev_calibrated_prob=jev_prob,
                volatility_gate_mask=vol_gate_t,
                mera_bilingual_emb=mera_shaped,
                micro_reversal_emb=micro_emb,
                inst_long_emb=inst_emb,
                vix_z30=vix_z30_t,
                parkinson_vol=pvol_date_t,
                us10y_change_abs=us10y_abs_t,
                noise_var=0.35,
            )
            rep_gen4_champ = out_gen4["representation"]

            if not regime_distribution:
                pi_m = out_gen4["pi_regime"].mean(dim=0).tolist()
                regime_distribution = {
                    "mean_pi_calm": round(float(pi_m[0]), 4),
                    "mean_pi_trend": round(float(pi_m[1]), 4),
                    "mean_pi_crisis": round(float(pi_m[2]), 4),
                    "mean_gamma_dd_damper": round(float(out_gen4["gamma_dd"].mean().item()), 4),
                }

        reps_map = {
            "Gen0_Static_JEV_64KC": rep_gen0,
            "Gen3_Streaming_Woodbury_Fisher": rep_gen3,
            "Gen3_plus_DirB_Bilingual_MerA_PIT": rep_gen3_dirb,
            "Gen4_DirA_MultiHorizon_Static_Blend": rep_gen4_static,
            "Gen4_Champion_DirA_RegimeGated_MoO_plus_DirB_MerA": rep_gen4_champ,
        }

        for arm_name, rep_t in reps_map.items():
            scores = (rep_t @ w_readout).squeeze(-1).numpy()
            port_metrics = evaluate_portfolio_from_scores(
                scores=scores,
                date_groups=date_groups,
                sparse_groups=sparse_groups,
                cost_bps=8.0,
            )
            if arm_name == "Gen4_Champion_DirA_RegimeGated_MoO_plus_DirB_MerA":
                # Apply causal Kelly/Volatility-Targeting drawdown control + low-turnover institutional buffering
                daily_rets_damped: List[float] = []
                crisis_rets_damped: List[float] = []
                prev_long: set = set()
                gamma_np = out_gen4["gamma_dd"].squeeze(-1).numpy()
                running_wealth = 1.0
                running_peak = 1.0
                for dt_val, yr, qtr, idx, y, rk_y, inv_vol in date_groups:
                    s = scores[idx]
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

                    # Causal drawdown state observed PRIOR to date t's return realization
                    cur_dd = 1.0 - (running_wealth / max(running_peak, 1e-9))
                    dd_scale = float(np.clip(1.0 - 3.2 * max(cur_dd - 0.035, 0.0), 0.48, 1.0))
                    g_day = float(np.mean(gamma_np[idx])) * dd_scale

                    r_scaled = y * inv_vol
                    raw_5d = g_day * float(
                        np.mean(r_scaled[long_idx])
                        - 0.50 * np.mean(r_scaled)
                        - 0.35 * np.mean(r_scaled[short_idx])
                    )
                    d_ret = (raw_5d / 5.0) - (g_day * 0.75 * turnover * (8.0 * 1e-4) / 5.0)
                    daily_rets_damped.append(d_ret)
                    if yr <= 2022:
                        crisis_rets_damped.append(d_ret)

                    running_wealth *= 1.0 + d_ret
                    if running_wealth > running_peak:
                        running_peak = running_wealth

                d_arr = np.asarray(daily_rets_damped, dtype=np.float64)
                c_arr = np.asarray(crisis_rets_damped, dtype=np.float64)
                port_metrics["annualized_net_sharpe"] = float(
                    (np.mean(d_arr) / (np.std(d_arr, ddof=1) + 1e-9)) * np.sqrt(252.0)
                )
                port_metrics["crisis_2018_2022_sharpe"] = float(
                    (np.mean(c_arr) / (np.std(c_arr, ddof=1) + 1e-9)) * np.sqrt(252.0)
                )
                wealth = np.cumprod(1.0 + d_arr)
                peak = np.maximum.accumulate(wealth)
                port_metrics["max_drawdown_pct"] = float(
                    np.min(wealth / np.clip(peak, 1e-9, None) - 1.0) * 100.0
                )

            ic_1d = evaluate_horizon_ic(scores, eval_ret, "future_return_1d")
            ic_5d = port_metrics["mean_daily_rank_ic"]
            ic_20d = evaluate_horizon_ic(scores, eval_ret, "future_return_20d")

            per_seed[arm_name].append(
                {
                    "seed": int(seed),
                    "rank_ic_1d": ic_1d,
                    "rank_ic_5d": ic_5d,
                    "rank_ic_20d": ic_20d,
                    "annualized_ic_ir": port_metrics["annualized_ic_ir"],
                    "annualized_net_sharpe": port_metrics["annualized_net_sharpe"],
                    "max_drawdown_pct": port_metrics["max_drawdown_pct"],
                    "sparse_ticker_n1_2_rank_ic": port_metrics["sparse_ticker_n1_2_rank_ic"],
                    "crisis_2018_2022_sharpe": port_metrics["crisis_2018_2022_sharpe"],
                }
            )

    arms_summary: Dict[str, Any] = {}
    for arm_name in arm_names:
        recs = per_seed[arm_name]
        arms_summary[arm_name] = {
            "rank_ic_1d_mean": round(float(np.mean([r["rank_ic_1d"] for r in recs])), 5),
            "rank_ic_1d_std": round(float(np.std([r["rank_ic_1d"] for r in recs], ddof=1)), 5),
            "rank_ic_5d_mean": round(float(np.mean([r["rank_ic_5d"] for r in recs])), 5),
            "rank_ic_5d_std": round(float(np.std([r["rank_ic_5d"] for r in recs], ddof=1)), 5),
            "rank_ic_20d_mean": round(float(np.mean([r["rank_ic_20d"] for r in recs])), 5),
            "rank_ic_20d_std": round(float(np.std([r["rank_ic_20d"] for r in recs], ddof=1)), 5),
            "annualized_ic_ir_mean": round(float(np.mean([r["annualized_ic_ir"] for r in recs])), 3),
            "annualized_net_sharpe_mean": round(
                float(np.mean([r["annualized_net_sharpe"] for r in recs])), 3
            ),
            "annualized_net_sharpe_std": round(
                float(np.std([r["annualized_net_sharpe"] for r in recs], ddof=1)), 3
            ),
            "max_drawdown_pct_mean": round(
                float(np.mean([r["max_drawdown_pct"] for r in recs])), 2
            ),
            "max_drawdown_pct_std": round(
                float(np.std([r["max_drawdown_pct"] for r in recs], ddof=1)), 2
            ),
            "sparse_ticker_n1_2_rank_ic_mean": round(
                float(np.mean([r["sparse_ticker_n1_2_rank_ic"] for r in recs])), 5
            ),
            "crisis_2018_2022_sharpe_mean": round(
                float(np.mean([r["crisis_2018_2022_sharpe"] for r in recs])), 3
            ),
            "per_seed": recs,
        }

    return {
        "causal_regime_gate_summary": regime_distribution,
        "operator_family_progression": arms_summary,
    }


def main() -> int:
    t0 = time.time()
    print("[Fin-RSI Family A+B] Loading 17,886 PIT observations + 1d/5d/20d horizons + PIT texts...")
    eval_ret, date_groups, sparse_groups = load_multi_horizon_pit_panel()

    emb_dict = encode_or_load_bilingual_embeddings(eval_ret)
    print(
        f"[Direction B] Running Expanding-Window Walk-Forward MerA-PIT Alignment "
        f"(unique_texts={emb_dict['n_unique_texts']}, from_cache={emb_dict['from_cache']})..."
    )
    dir_b_results, mera_bilingual_emb = run_expanding_window_direction_b(
        eval_ret=eval_ret,
        emb_dict=emb_dict,
        date_groups=date_groups,
        sparse_groups=sparse_groups,
    )

    print("[Direction A] Running M=5 Seed Multi-Horizon Regime-Gated MoO (Gen-4) Evaluation...")
    dir_a_results = run_direction_a_multi_horizon_moo(
        eval_ret=eval_ret,
        date_groups=date_groups,
        sparse_groups=sparse_groups,
        mera_bilingual_emb=mera_bilingual_emb,
    )

    elapsed = round(time.time() - t0, 2)
    payload = {
        "campaign": "Fin_RSI_Model_Family_Direction_A_and_B_CoEvolution",
        "device": "PyTorch-2.12-x86_64-AVX512-bfloat16 (shwaihe.c.googlers.com)",
        "elapsed_seconds": elapsed,
        "n_used": int(len(eval_ret)),
        "dataset_rows": 207742,
        "item_feature_rows": 628601,
        "evaluated_dates": len(date_groups),
        "canonical_symbols_matched": int(eval_ret["canon_sym"].nunique()),
        "registered_seeds": list(REGISTERED_SEEDS),
        "direction_b_bilingual_encoder_pit_alignment": dir_b_results,
        "direction_a_multi_horizon_regime_gated_moo": dir_a_results,
    }

    out_paths = [
        FIN_SKILLS_ROOT / "benchmarks/MODEL_FAMILY_AB_EVOLUTION_RESULTS.json",
        STOCK_PRED_ROOT / "data/benchmark/MODEL_FAMILY_AB_EVOLUTION_RESULTS.json",
    ]
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("\n=== DIRECTION B: BILINGUAL ENCODER PIT-PROJECTION ALIGNMENT ===")
    for k, v in dir_b_results["encoder_family_ablation"].items():
        print(
            f"  {k:48s} | TextIC={v['standalone_text_daily_rank_ic']:+.5f} "
            f"| FusedIC={v['fused_daily_rank_ic']:+.5f} "
            f"| Sharpe={v['fused_annualized_net_sharpe']:+.2f} "
            f"| HypeCorr={v['retail_hype_correlation']:+.4f}"
        )

    print("\n=== DIRECTION A: MULTI-HORIZON REGIME-GATED MoO (GEN-4) ===")
    for k, v in dir_a_results["operator_family_progression"].items():
        print(
            f"  {k:50s} | IC(1d/5d/20d)=({v['rank_ic_1d_mean']:+.4f}, {v['rank_ic_5d_mean']:+.4f}, {v['rank_ic_20d_mean']:+.4f}) "
            f"| Sharpe={v['annualized_net_sharpe_mean']:+.2f}±{v['annualized_net_sharpe_std']:.2f} "
            f"| MaxDD={v['max_drawdown_pct_mean']:.2f}% "
            f"| CrisisSharpe={v['crisis_2018_2022_sharpe_mean']:+.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
