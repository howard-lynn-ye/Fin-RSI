"""Financial JEV question templates adapted from Shwai He's v2 implementation.

These helpers define questions for a caller-selected model. They do not forecast,
calibrate probabilities, select exposures, or create local model answers.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
from numbers import Number
import pandas as pd

from fin_skills.model_zoo.jev import _questions, _response


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
    _ticker(ticker)
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
    _ticker(ticker)
    return {
        f"rating_{ticker}": {
            "type": "score",
            "instructions": {
                "question": f"What is the expected 5-day risk-adjusted alpha rating for {ticker}?",
                "context": context or "Synthesized point-in-time momentum, valuation quality, and supply-chain spillover.",
            },
            "criteria": list(ORDINAL_RATING_LEVELS),
        }
    }


def _ticker(ticker: str) -> None:
    if not isinstance(ticker, str) or not ticker.strip() or ticker != ticker.strip():
        raise ValueError("ticker must be a nonempty string without surrounding whitespace")


def validate_stock_jev_decision(decision_result: dict, questions: dict) -> dict:
    """Validate caller-supplied answers against their actual questions and wire contract.

    Checks IDs, answer types, finite bounded probabilities, declared choices, ordinal
    expectations, model identity and usage. This does not measure forecast calibration.
    """
    return _response(decision_result, _questions(questions))


def audit_stock_jev_causality(features_df: pd.DataFrame, asof_date: str) -> bool:
    """Check supplied feature timestamps against an inclusive UTC cutoff.

    A timestamp check cannot establish how a feature was calculated or when its source
    became available. Pass availability timestamps in the date column; date-only values
    mean midnight. Empty, missing, numeric and unparseable dates are rejected.
    """
    if not isinstance(features_df, pd.DataFrame) or features_df.empty:
        raise ValueError("features_df must be a nonempty DataFrame")
    if "date" not in features_df or features_df["date"].isna().any():
        raise ValueError("features_df requires nonmissing dates")
    if any(isinstance(value, Number) for value in features_df["date"]):
        raise ValueError("date requires timestamps, not numeric offsets")
    if not isinstance(asof_date, str) or not asof_date.strip():
        raise ValueError("asof_date must be a timestamp string")
    dates = pd.to_datetime(features_df["date"], format="mixed", utc=True, errors="raise")
    cutoff = pd.to_datetime(asof_date, utc=True, errors="raise")
    if dates.isna().any() or pd.isna(cutoff):
        raise ValueError("dates and cutoff must not be NaT")
    if dates.max() > cutoff:
        raise ValueError("Look-ahead violation: feature timestamp exceeds asof_date")
    return True
