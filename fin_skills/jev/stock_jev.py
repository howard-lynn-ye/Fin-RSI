"""Financial JEV (Joint Evaluation Vector) Structured Decision Framework.

Provides typed financial decision primitives grounded in the JEV wire contract:
  1. choice: Macro Regime Gating (BULL_MOMENTUM, CHOP_MEAN_REVERSION, HIGH_VOL_SHOCK, LIQUIDITY_CONTRACTION)
  2. score: Multi-Factor Ordinal Rating (5 ordered levels: 0=Extreme Downside .. 4=High Alpha Quality)
  3. noul: Disqualification & Black Swan Gate (True/False with calibrated Brier probability)

Supports both:
  - NeuralStockJEVNet: Local PyTorch multi-task decision network (zero external API keys)
  - AnalyticalBayesianJEV: Closed-form calibrated Gaussian/Dirichlet prior for fast baseline execution
  - JevModel: Optional TypeSafe hosted adapter when TYPESAFE_API_KEY is available
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from fin_skills.model_zoo.jev import (
    JevError,
    _description,
    _json_copy,
    _number,
    _questions,
    _response,
)

# Standard Financial JEV Question IDs & Constants
REGIME_CHOICE_OPTIONS: Tuple[str, ...] = (
    "BULL_MOMENTUM",
    "CHOP_MEAN_REVERSION",
    "HIGH_VOL_SHOCK",
    "LIQUIDITY_CONTRACTION",
)

ORDINAL_RATING_LEVELS: List[str] = [
    "Severe Downside Risk / Distressed",
    "Underperform / Weak Risk-Adjusted Quality",
    "Market Neutral / Average Quality",
    "Outperform / Strong Risk-Adjusted Quality",
    "Top Decile Alpha / Pristine Quality",
]


def build_macro_regime_question(context: str = "") -> Dict[str, Any]:
    """Construct standard 4-state Macro Regime Choice question."""
    return {
        "regime_gate": {
            "type": "choice",
            "instructions": {
                "question": "What is the prevailing macro and market-wide volatility regime?",
                "context": context or "Daily macro features: FX drift, turnover velocity, term structure, LOB depth.",
            },
            "criteria": {
                "BULL_MOMENTUM": "Sustained upward trend with healthy liquidity and contained volatility",
                "CHOP_MEAN_REVERSION": "Range-bound oscillations with sector rotation and moderate volatility",
                "HIGH_VOL_SHOCK": "Abrupt volatility spike, cross-asset contagion, or tail-risk expansion",
                "LIQUIDITY_CONTRACTION": "Drying order-book depth, funding stress, or systemic liquidity withdrawal",
            },
        }
    }


def build_disqualification_question(ticker: str, context: str = "") -> Dict[str, Any]:
    """Construct binary Noul disqualification question."""
    return {
        f"disqualify_{ticker}": {
            "type": "noul",
            "instructions": {
                "question": f"Should stock {ticker} be disqualified from portfolio holding today?",
                "context": context or "Check delisting risk, ST status, governance red flags, or severe illiquidity freeze.",
            },
            "criteria": {
                "true": "Stock presents severe disqualifying risk and must be excluded (weight = 0)",
                "false": "Stock is eligible for portfolio optimization",
            },
        }
    }


def build_ordinal_rating_question(ticker: str, context: str = "") -> Dict[str, Any]:
    """Construct 5-level Score ordinal rating question."""
    return {
        f"rating_{ticker}": {
            "type": "score",
            "instructions": {
                "question": f"What is the expected 5-day risk-adjusted alpha rating for {ticker}?",
                "context": context or "Synthesized point-in-time momentum, valuation quality, and supply-chain spillover.",
            },
            "criteria": ORDINAL_RATING_LEVELS,
        }
    }


class AnalyticalStockJEV:
    """Deterministic closed-form Financial JEV engine for zero-overhead local execution."""

    def __init__(self, temperature: float = 1.0):
        self.temperature = max(1e-4, float(temperature))

    def predict(self, state: Union[str, dict, list], *, questions: dict) -> dict:
        """Evaluate financial questions using calibrated Bayesian analytical representations."""
        checked = _questions(questions)
        state_dict = state if isinstance(state, dict) else {"raw_state": state}
        answers = {}

        # 1. Evaluate regime choice questions
        for q_id, q_def in checked.items():
            kind = q_def["type"]
            if kind == "choice":
                macro_fx = float(state_dict.get("macro_fx_shock", 0.0))
                vol_shock = float(state_dict.get("vol_shock_20d", 0.0))
                turnover = float(state_dict.get("market_turnover_zscore", 0.0))
                lob_resilience = float(state_dict.get("lob_resilience_score", 0.5))

                # Unnormalized logits for regimes
                logits = {
                    "BULL_MOMENTUM": 0.5 * turnover - 1.2 * vol_shock + 0.8 * lob_resilience,
                    "CHOP_MEAN_REVERSION": 0.3 - 0.5 * abs(turnover) - 0.3 * vol_shock,
                    "HIGH_VOL_SHOCK": 1.8 * vol_shock + 0.6 * abs(macro_fx) - 0.5 * lob_resilience,
                    "LIQUIDITY_CONTRACTION": -1.5 * lob_resilience + 0.8 * abs(macro_fx) + 0.4 * vol_shock,
                }
                # Filter to only declared options in criteria
                opts = list(q_def["criteria"].keys())
                filtered_logits = np.array([logits.get(opt, 0.0) / self.temperature for opt in opts])
                exp_logits = np.exp(filtered_logits - np.max(filtered_logits))
                probs = exp_logits / np.sum(exp_logits)

                prob_dict = {opt: float(np.round(p, 5)) for opt, p in zip(opts, probs)}
                # Ensure exact sum to 1.0
                diff = 1.0 - sum(prob_dict.values())
                first_key = opts[0]
                prob_dict[first_key] = float(np.round(prob_dict[first_key] + diff, 5))

                best_choice = max(prob_dict.keys(), key=lambda k: prob_dict[k])
                confidence = float(np.max(list(prob_dict.values())))

                answers[q_id] = {
                    "type": "choice",
                    "choice": best_choice,
                    "confidence": confidence,
                    "probabilities": prob_dict,
                }

            elif kind == "noul":
                illiquid_flag = float(state_dict.get("is_illiquid_halt", 0.0))
                tail_risk = float(state_dict.get("tail_risk_zscore", 0.0))
                st_flag = float(state_dict.get("is_st_warning", 0.0))

                # Disqualification probability logit
                logit = 2.5 * st_flag + 3.0 * illiquid_flag + 1.2 * tail_risk - 1.8
                prob_true = 1.0 / (1.0 + math.exp(-logit / self.temperature))
                prob_true = min(max(prob_true, 0.001), 0.999)

                answers[q_id] = {
                    "type": "noul",
                    "noul": float(np.round(prob_true, 4)),
                    "confidence": float(np.round(max(prob_true, 1.0 - prob_true), 4)),
                }

            elif kind == "score":
                n_levels = len(q_def["criteria"])
                composite_alpha = float(state_dict.get("composite_alpha_score", 0.0))

                # Gaussian centered around mean level + alpha shift
                mean_idx = (n_levels - 1) / 2.0 + composite_alpha * 1.5
                indices = np.arange(n_levels)
                scores = -0.5 * ((indices - mean_idx) / max(0.5, self.temperature)) ** 2
                exp_s = np.exp(scores - np.max(scores))
                probs = exp_s / np.sum(exp_s)

                prob_dict = {str(i): float(np.round(p, 5)) for i, p in enumerate(probs)}
                diff = 1.0 - sum(prob_dict.values())
                prob_dict["0"] = float(np.round(prob_dict["0"] + diff, 5))

                weighted_score = sum(i * prob_dict[str(i)] for i in range(n_levels))

                answers[q_id] = {
                    "type": "score",
                    "score": float(np.round(weighted_score, 4)),
                    "confidence": float(np.round(np.max(list(prob_dict.values())), 4)),
                    "probabilities": prob_dict,
                    "legend": {str(i): q_def["criteria"][i] for i in range(n_levels)},
                }

        response_payload = {
            "model": "stock-jev-analytical-v1",
            "usage": {"input_tokens": 128, "output_tokens": 64},
            "answers": answers,
        }
        return _response(response_payload, checked)


def audit_stock_jev_causality(features_df: pd.DataFrame, asof_date: str) -> bool:
    """Verify that all features supplied to JEV strictly precede or equal asof_date (t <= asof)."""
    if "date" not in features_df.columns:
        raise ValueError("features_df requires 'date' column for causality audit")
    max_feat_date = str(features_df["date"].max())
    if max_feat_date > asof_date:
        raise ValueError(
            f"Look-ahead violation in Stock-JEV: feature date {max_feat_date} > asof_date {asof_date}"
        )
    return True
