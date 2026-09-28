"""Unit tests for fin_skills.fin_rsi (Governed Financial Recursive Self-Improvement)."""

from __future__ import annotations

import pathlib

import pytest
import torch

from fin_skills.fin_rsi import (
    BilingualMerAPITProjector,
    CrossBoardBarraResidualMoOOperator,
    FinRSIFamily,
    MultiHorizonRegimeGatedMoOOperator,
    REGISTERED_FIN_RSI_SEEDS,
    StreamingWoodburyFisherOperator,
    SubspacePrecisionSteinOperator,
    ValueSpaceBoundedESSOperator,
    diagnose_financial_operator_physics,
    evaluate_rsi_pareto_gate,
    verify_rsi_harness_lock,
)


def test_verify_rsi_harness_lock_roundtrip_and_tamper_detection(tmp_path: pathlib.Path) -> None:
    harness_file = tmp_path / "frozen_harness.py"
    harness_file.write_text(
        '"""Clean read-only harness."""\n'
        "def score_arm(x):\n"
        "    return x.mean(dim=-1)\n",
        encoding="utf-8",
    )

    lock_res = verify_rsi_harness_lock(
        sandbox_dir=tmp_path,
        action="lock",
        seeds=REGISTERED_FIN_RSI_SEEDS,
        dataset_rows=207742,
        n_used=107999,
    )
    assert lock_res["passed"] is True
    assert len(lock_res["sha256"]) == 64

    check_res = verify_rsi_harness_lock(sandbox_dir=tmp_path, action="check")
    assert check_res["passed"] is True
    assert check_res["sha256"] == lock_res["sha256"]

    # Tamper with frozen_harness.py -> must fail closed
    harness_file.write_text(
        '"""Tampered harness."""\n'
        "def score_arm(x):\n"
        "    return x.sum(dim=-1)\n",
        encoding="utf-8",
    )
    tampered_res = verify_rsi_harness_lock(sandbox_dir=tmp_path, action="check")
    assert tampered_res["passed"] is False
    assert "HARNESS TAMPER DETECTED" in tampered_res["error"]


def test_verify_rsi_harness_lock_rejects_artificial_baseline_penalty(tmp_path: pathlib.Path) -> None:
    harness_file = tmp_path / "frozen_harness.py"
    harness_file.write_text(
        "def eval_row(arm_name, logit, torch):\n"
        "    if 'row_2' in arm_name:\n"
        "        logit = logit - 0.25 * torch.randn_like(logit)\n"
        "    return logit\n",
        encoding="utf-8",
    )
    res = verify_rsi_harness_lock(sandbox_dir=tmp_path, action="lock")
    assert res["passed"] is False
    assert len(res["findings"]) >= 1


def test_diagnose_financial_operator_physics_probes() -> None:
    diag = diagnose_financial_operator_physics(tau_bounded=2.5, seed=20260923)

    pa = diag["probe_a_scale_cancellation"]
    assert pa["scale_cancellation_detected"] is True
    assert pa["pre_ln_grad_norm_l2"] < 1e-4
    assert pa["post_encoder_value_pool_grad_norm_l2"] > 0.1

    pb = diag["probe_b_sequence_ess"]
    assert pb["T64_ess_ratio"] >= 3.0
    assert pb["T128_ess_ratio"] >= 3.0

    pc = diag["probe_c_slice_gap"]
    assert pc["sparse_n1_2_subspace_stein_mse"] < pc["sparse_n1_2_scalar_shrinkage_mse"]
    assert pc["sparse_mse_reduction_pct"] > 20.0


def test_gen1_value_space_bounded_ess_operator_gradients_and_ess() -> None:
    torch.manual_seed(20260923)
    op = ValueSpaceBoundedESSOperator(embed_dim=16, tau=2.5)
    seq_emb = torch.randn(8, 16, 16, requires_grad=True)
    logits = torch.randn(8, 16, requires_grad=True)
    bursts = torch.ones(8, 16)
    bursts[:, 0] = 6.0

    pooled, info = op(seq_emb, logits, burst_block_sizes=bursts)
    assert pooled.shape == (8, 16)
    assert info["ess"].shape == (8,)
    assert float(info["ess"].mean().item()) > 4.0

    loss = pooled.pow(2).sum()
    loss.backward()
    assert logits.grad is not None
    assert float(logits.grad.norm().item()) > 1e-3


def test_gen2_subspace_precision_stein_operator() -> None:
    torch.manual_seed(20260924)
    op = SubspacePrecisionSteinOperator(embed_dim=16, tau=2.5, c_stein=0.28)
    soc = torch.randn(12, 16)
    ann = torch.randn(12, 16)
    eff_n = torch.tensor([1.0, 2.0, 5.0, 10.0] * 3)

    out, diag = op(soc, ann, eff_n, noise_var=0.35)
    assert out.shape == (12, 16)
    assert diag["alpha_d"].shape == (12, 16)
    assert torch.all((diag["alpha_d"] > 0.0) & (diag["alpha_d"] < 1.0))
    assert torch.all((diag["stein_multiplier"] >= 0.0) & (diag["stein_multiplier"] <= 1.0))


def test_gen3_streaming_woodbury_fisher_operator() -> None:
    torch.manual_seed(20260925)
    op = StreamingWoodburyFisherOperator(embed_dim=16, kc_dim=64, top_k_kc=8)
    soc = torch.randn(10, 16)
    ann = torch.randn(10, 16)
    eff_n = torch.full((10, 1), 3.0)
    probs = torch.full((10, 1), 0.5)
    v_mask = torch.ones(10, 1)

    out, diag = op(soc, ann, eff_n, probs, volatility_gate_mask=v_mask, update_covariance=True)
    assert out.shape == (10, 16)
    assert torch.allclose(diag["fisher_gate"], torch.ones_like(diag["fisher_gate"]), atol=1e-5)
    assert diag["kc_sparse_features"].shape == (10, 64)
    # Verify k-sparsity: at most 8 non-zero active Kenyon cells per row
    nonzero_per_row = (diag["kc_sparse_features"] > 0).sum(dim=-1)
    assert torch.all(nonzero_per_row <= 8)
    # Verify inverse covariance matrix remains symmetric and positive-definite
    assert torch.allclose(op.inv_cov, op.inv_cov.T, atol=1e-5)
    eigvals = torch.linalg.eigvalsh(op.inv_cov)
    assert torch.all(eigvals > 0.0)


def test_evaluate_rsi_pareto_gate_promotion_and_rejection() -> None:
    mock_summary = {
        "elapsed_seconds": 12.4,
        "n_used": 107999,
        "dataset_rows": 207742,
        "registered_seeds": list(REGISTERED_FIN_RSI_SEEDS),
        "harness_lock_verified": True,
        "arms": {
            "Row_1_Full_Dense_Multimodal_Ref": {
                "mean_daily_rank_ic": 0.0310,
                "annualized_net_sharpe": 1.75,
                "sparse_ticker_n1_2_rank_ic": 0.0220,
                "crisis_2018_2022_sharpe": 1.15,
                "sequence_ess_ratio": 3.10,
                "leakage_rate_pct": 0.0,
                "deflated_sharpe_ratio_dsr": 0.99,
            },
            "Row_2_Prod_Baseline_Verbal_Reflexion_RSI": {
                "mean_daily_rank_ic": 0.0045,
                "annualized_net_sharpe": 0.25,
                "sparse_ticker_n1_2_rank_ic": -0.0080,
                "crisis_2018_2022_sharpe": -0.45,
                "sequence_ess_ratio": 1.00,
                "leakage_rate_pct": 38.5,
                "deflated_sharpe_ratio_dsr": 0.12,
            },
            "Row_3_Prod_Baseline_MMAN_Barra_Dual": {
                "mean_daily_rank_ic": 0.0255,
                "annualized_net_sharpe": 1.25,
                "sparse_ticker_n1_2_rank_ic": 0.0110,
                "crisis_2018_2022_sharpe": 0.65,
                "sequence_ess_ratio": 1.18,
                "leakage_rate_pct": 0.0,
                "deflated_sharpe_ratio_dsr": 0.96,
            },
            "Row_4_Prod_Baseline_JEV_SystemOne_Static_64KC": {
                "mean_daily_rank_ic": 0.0290,
                "annualized_net_sharpe": 1.68,
                "sparse_ticker_n1_2_rank_ic": 0.0140,
                "crisis_2018_2022_sharpe": 0.92,
                "sequence_ess_ratio": 1.34,
                "leakage_rate_pct": 0.0,
                "deflated_sharpe_ratio_dsr": 0.99,
            },
            "Row_5_RSI_Gen1_ValueSpace_BoundedESS": {
                "mean_daily_rank_ic": 0.0305,
                "annualized_net_sharpe": 1.79,
                "sparse_ticker_n1_2_rank_ic": 0.0135,  # Below Row 4 slice -> fails Gate 2
                "crisis_2018_2022_sharpe": 1.10,
                "sequence_ess_ratio": 3.48,
                "leakage_rate_pct": 0.0,
                "deflated_sharpe_ratio_dsr": 0.99,
            },
            "Row_7_RSI_Gen3_Streaming_Woodbury_Fisher_JEV_64KC": {
                "mean_daily_rank_ic": 0.0358,
                "annualized_net_sharpe": 2.18,
                "sparse_ticker_n1_2_rank_ic": 0.0312,
                "crisis_2018_2022_sharpe": 1.65,
                "sequence_ess_ratio": 3.62,
                "leakage_rate_pct": 0.0,
                "deflated_sharpe_ratio_dsr": 1.00,
            },
        },
    }
    res = evaluate_rsi_pareto_gate(mock_summary)
    assert res["passed"] is True
    assert res["status"] == "PROMOTED_CHAMPION"
    assert (
        res["arm_evaluations"]["Row_5_RSI_Gen1_ValueSpace_BoundedESS"]["verdict"]
        == "HONEST_BELOW_THRESHOLD"
    )
    assert (
        res["arm_evaluations"]["Row_7_RSI_Gen3_Streaming_Woodbury_Fisher_JEV_64KC"]["verdict"]
        == "PROMOTED_CHAMPION"
    )


def test_skill_level_rsi_evolution_report_and_harness_gates() -> None:
    import hashlib
    import json

    root = pathlib.Path(__file__).resolve().parent.parent
    lock_path = root / "benchmarks/fin_rsi/SKILL_HARNESS_LOCK.json"
    report_path = root / "benchmarks/SKILL_RSI_EVOLUTION_REPORT.json"
    assert lock_path.exists()
    assert report_path.exists()

    lock_data = json.loads(lock_path.read_text(encoding="utf-8"))
    for rel, info in lock_data["files"].items():
        actual_sha = hashlib.sha256((root / rel).read_bytes()).hexdigest()
        assert actual_sha == info["sha256"], f"Harness lock mismatch on {rel}"

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["total_skills_in_catalog"] == 129
    assert report["max_description_length_chars"] <= 1024
    assert report["gate_receipts"]["eval_triggers_exit_code"] == 0
    assert report["gate_receipts"]["eval_blind_exit_code"] == 0
    assert report["gate_receipts"]["validate_exit_code"] == 0

    g0 = report["generations"]["Gen-0_Unoptimized_Wave2_Catalog"]
    g1 = report["generations"]["Gen-1_Contrastive_TRIGGER_Amplification"]
    g2 = report["generations"]["Gen-2_Orthogonal_Negative_SKIP_Disambiguation"]
    g3 = report["generations"]["Gen-3_Champion_1Hop_Xref_Graph_and_Package_Sync"]

    assert g0["eval_triggers"]["top1_hits"] == 72
    assert g1["eval_triggers"]["top1_hits"] >= 103
    assert g2["eval_triggers"]["top1_hits"] == 108
    assert g2["eval_triggers"]["thin_margins_count"] == 0
    assert g3["eval_triggers"]["top1_hits"] == 108
    assert g3["eval_triggers"]["routed_correct"] == 108
    assert g3["eval_triggers"]["thin_margins_count"] == 0
    assert g3["eval_triggers"]["min_margin"] >= 0.15
    assert g3["eval_blind"]["correct"] == 108
    assert g3["multi_encoder_routing"]["jev_system_one_calibrated_router_ours"]["top1_shuffled"] >= 97


def test_direction_b_bilingual_mera_pit_projector_orthogonality() -> None:
    torch.manual_seed(20260923)
    proj = BilingualMerAPITProjector(input_dim=64, embed_dim=16, ridge_lambda=5.0)
    train_emb = torch.randn(128, 64)
    anchor_y = torch.randn(128, 8)
    surprise_y = torch.randn(128, 8)
    hype_nuis = torch.randn(128, 3)
    weights = torch.rand(128) + 0.5

    diag = proj.fit_expanding_pit(
        train_emb=train_emb,
        train_anchor_targets=anchor_y,
        train_surprise_targets=surprise_y,
        train_hype_nuisance=hype_nuis,
        sample_weights=weights,
    )
    assert proj.is_fitted.item() is True
    assert diag["n_train_pit"] == 128
    # Verify Gram-Schmidt orthogonalization: W_perp is strictly orthogonal to hype_basis
    assert diag["max_cos_perp_hype"] < 1e-4

    out_16d, out_diag = proj(torch.randn(16, 64))
    assert out_16d.shape == (16, 16)
    assert out_diag["z_parallel"].shape == (16, 8)
    assert out_diag["z_perp"].shape == (16, 8)


def test_direction_a_multi_horizon_regime_gated_moo_operator() -> None:
    torch.manual_seed(20260924)
    op = MultiHorizonRegimeGatedMoOOperator(embed_dim=16, kc_dim=64, top_k_kc=8)
    b_sz, seq_len, dim = 10, 8, 16
    seq_soc = torch.randn(b_sz, seq_len, dim)
    logits = torch.randn(b_sz, seq_len)
    bursts = torch.ones(b_sz, seq_len)
    ann_prior = torch.randn(b_sz, dim)
    eff_n = torch.full((b_sz, 1), 4.0)
    probs = torch.full((b_sz, 1), 0.6)
    v_mask = torch.ones(b_sz, 1)
    mera_emb = torch.randn(b_sz, dim)
    micro_emb = torch.randn(b_sz, dim)
    inst_emb = torch.randn(b_sz, dim)
    vix_z = torch.linspace(-1.5, 2.5, b_sz).unsqueeze(-1)
    pvol = torch.full((b_sz, 1), 0.025)
    us10y = torch.full((b_sz, 1), 0.03)

    out = op(
        seq_social_emb=seq_soc,
        salience_logits=logits,
        burst_block_sizes=bursts,
        announcement_prior_emb=ann_prior,
        eff_sample_count=eff_n,
        jev_calibrated_prob=probs,
        volatility_gate_mask=v_mask,
        mera_bilingual_emb=mera_emb,
        micro_reversal_emb=micro_emb,
        inst_long_emb=inst_emb,
        vix_z30=vix_z,
        parkinson_vol=pvol,
        us10y_change_abs=us10y,
    )
    assert out["representation"].shape == (b_sz, dim)
    assert out["pi_regime"].shape == (b_sz, 3)
    # Simplex probabilities sum to 1.0
    assert torch.allclose(out["pi_regime"].sum(dim=-1), torch.ones(b_sz), atol=1e-5)
    # Higher vix_z30 induces higher crisis probability and tighter drawdown damper (smaller gamma_dd)
    assert out["pi_regime"][-1, 2] > out["pi_regime"][0, 2]
    assert out["gamma_dd"][-1, 0] < out["gamma_dd"][0, 0]


def test_gen5_cross_board_barra_residual_moo_operator() -> None:
    torch.manual_seed(20260925)
    op = CrossBoardBarraResidualMoOOperator(
        embed_dim=16, kc_dim=64, top_k_kc=8, barra_neutralization_strength=0.22
    )
    b_sz, seq_len, dim = 12, 8, 16
    seq_soc = torch.randn(b_sz, seq_len, dim, requires_grad=True)
    logits = torch.randn(b_sz, seq_len, requires_grad=True)
    bursts = torch.ones(b_sz, seq_len)
    ann_prior = torch.randn(b_sz, dim)
    eff_n = torch.full((b_sz, 1), 4.0)
    probs = torch.full((b_sz, 1), 0.6)
    v_mask = torch.ones(b_sz, 1)
    board_ids = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2], dtype=torch.int64).unsqueeze(-1)
    lim_prox = torch.linspace(0.0, 0.9, b_sz).unsqueeze(-1)
    style_nuis = torch.randn(b_sz, 2)

    out = op(
        seq_social_emb=seq_soc,
        salience_logits=logits,
        burst_block_sizes=bursts,
        announcement_prior_emb=ann_prior,
        eff_sample_count=eff_n,
        jev_calibrated_prob=probs,
        volatility_gate_mask=v_mask,
        board_type_id=board_ids,
        limit_proximity=lim_prox,
        style_nuisance_factors=style_nuis,
    )
    assert out["representation"].shape == (b_sz, dim)
    assert out["limit_damper"].shape == (b_sz, 1)
    assert out["limit_damper"][-1, 0] < out["limit_damper"][0, 0]

    loss = out["representation"].pow(2).sum()
    loss.backward()
    assert logits.grad is not None
    assert float(logits.grad.norm().item()) > 1e-4


def test_fin_rsi_family_presets_and_live_scoring() -> None:
    import numpy as np

    rng = np.random.default_rng(20260923)
    seq_15 = rng.normal(0.0, 0.08, size=(20, 15)).astype(np.float32)

    for preset, expected_gen in (("nano", "gen3"), ("base", "gen4"), ("pro", "gen5")):
        fam = FinRSIFamily.from_preset(preset=preset)
        meta = fam.describe()
        assert meta["preset"] == preset
        assert meta["generation"] == expected_gen

        for sym in ("SH600036", "SZ300729", "BEKE"):
            res = fam.score_live_symbol(
                symbol=sym,
                features_seq=seq_15,
                base_onnx_p_up=0.56,
                kol_sentiment=0.70,
            )
            assert 0.10 <= res["p_up"] <= 0.90
            assert 0.10 <= res["p_up_1d"] <= 0.90
            assert 0.10 <= res["p_up_20d"] <= 0.90
            assert res["sequence_ess"] >= 1.0
            assert 0.50 <= res["gamma_dd"] <= 1.05
