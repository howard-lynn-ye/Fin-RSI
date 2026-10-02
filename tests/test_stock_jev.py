"""Unit tests for Financial JEV Structured Decision Framework (fin_skills.jev)."""

import pandas as pd
import pytest
import torch

from fin_skills.api.guards.stock_jev_gate import check_stock_jev_gate
from fin_skills.jev.neural_jev import NeuralStockJEVAdapter, NeuralStockJEVNet
from fin_skills.jev.stock_jev import (
    ORDINAL_RATING_LEVELS,
    REGIME_CHOICE_OPTIONS,
    AnalyticalStockJEV,
    audit_stock_jev_causality,
    build_disqualification_question,
    build_macro_regime_question,
    build_ordinal_rating_question,
)
from fin_skills.model_zoo import create_model


def test_analytical_stock_jev_all_primitives():
    engine = AnalyticalStockJEV()
    questions = {
        **build_macro_regime_question("2026-10-02 macro context"),
        **build_disqualification_question("SH600036", "PB=0.8, ROE=15%"),
        **build_ordinal_rating_question("SH600036", "Composite alpha = 0.8"),
    }
    state = {
        "macro_fx_shock": -0.012,
        "vol_shock_20d": 0.45,
        "market_turnover_zscore": 1.2,
        "lob_resilience_score": 0.85,
        "composite_alpha_score": 0.65,
        "tail_risk_zscore": -0.5,
        "is_st_warning": 0.0,
        "is_illiquid_halt": 0.0,
    }
    res = engine.predict(state, questions=questions)
    assert res["model"] == "stock-jev-analytical-v1"
    answers = res["answers"]
    assert "regime_gate" in answers
    assert "disqualify_SH600036" in answers
    assert "rating_SH600036" in answers

    # Choice assertion
    regime = answers["regime_gate"]
    assert regime["type"] == "choice"
    assert regime["choice"] in REGIME_CHOICE_OPTIONS
    assert abs(sum(regime["probabilities"].values()) - 1.0) < 1e-4

    # Noul assertion
    disq = answers["disqualify_SH600036"]
    assert disq["type"] == "noul"
    assert 0.0 <= disq["noul"] <= 1.0

    # Score assertion
    rating = answers["rating_SH600036"]
    assert rating["type"] == "score"
    assert 0.0 <= rating["score"] <= 4.0
    assert abs(sum(rating["probabilities"].values()) - 1.0) < 1e-4

    # Guard assertion
    outcome = check_stock_jev_gate(res)
    assert outcome.passed


def test_neural_stock_jev_net_and_adapter():
    net = NeuralStockJEVNet(dim_macro=8, dim_stock=16, dim_micro=8, hidden_dim=32)
    adapter = NeuralStockJEVAdapter(model=net, device="cpu")

    questions = {
        **build_macro_regime_question(),
        **build_disqualification_question("SZ300750"),
        **build_ordinal_rating_question("SZ300750"),
    }
    state = {
        "macro_vector": [0.01, -0.02, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0],
        "stock_vector": [0.3] * 16,
        "micro_vector": [0.8, 12.0, 0.1, -0.2, 0.0, 0.0, 0.0, 0.0],
    }
    res = adapter.predict(state, questions=questions)
    assert res["model"] == "neural-stock-jev-v1"
    assert "regime_gate" in res["answers"]
    assert "disqualify_SZ300750" in res["answers"]
    assert "rating_SZ300750" in res["answers"]

    outcome = check_stock_jev_gate(res)
    assert outcome.passed


def test_model_zoo_factory():
    m_analytical = create_model("stock_jev")
    assert isinstance(m_analytical, AnalyticalStockJEV)

    m_neural = create_model("stock_jev", use_neural=True)
    assert isinstance(m_neural, NeuralStockJEVAdapter)


def test_causality_audit():
    df_clean = pd.DataFrame({"date": ["2026-09-30", "2026-10-01", "2026-10-02"], "value": [1, 2, 3]})
    assert audit_stock_jev_causality(df_clean, asof_date="2026-10-02")

    df_future = pd.DataFrame({"date": ["2026-09-30", "2026-10-03"], "value": [1, 2]})
    with pytest.raises(ValueError, match="Look-ahead violation"):
        audit_stock_jev_causality(df_future, asof_date="2026-10-02")


def test_guard_planted_defect_rejection():
    # 1. Invalid response: missing answers
    outcome = check_stock_jev_gate({"model": "bad", "usage": {}})
    assert not outcome.passed

    # 2. Probability does not sum to 1
    bad_prob_res = {
        "model": "bad-prob",
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "answers": {
            "regime_gate": {
                "type": "choice",
                "choice": "BULL_MOMENTUM",
                "confidence": 0.8,
                "probabilities": {
                    "BULL_MOMENTUM": 0.4,
                    "CHOP_MEAN_REVERSION": 0.3,
                    "HIGH_VOL_SHOCK": 0.1,
                    "LIQUIDITY_CONTRACTION": 0.1,
                },
            }
        },
    }
    outcome = check_stock_jev_gate(bad_prob_res)
    assert not outcome.passed

    # 3. Disqualification rate exceeds 35%
    bad_disq_res = {
        "model": "high-disq",
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "answers": {
            "disq_1": {"type": "noul", "noul": 0.95, "confidence": 0.95},
            "disq_2": {"type": "noul", "noul": 0.85, "confidence": 0.85},
            "disq_3": {"type": "noul", "noul": 0.05, "confidence": 0.95},
        },
    }
    outcome = check_stock_jev_gate(bad_disq_res, max_disqualification_rate=0.35)
    assert not outcome.passed  # 2/3 = 66.7% > 35%
