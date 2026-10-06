"""Unit tests for Multi-Source Financial Signal Conflict Resolution Engine."""
from __future__ import annotations

import pytest
from fin_skills.china.signal_reconciler import (
    ChannelSignal,
    ReconciliationResult,
    SignalReconciler,
)


@pytest.fixture
def reconciler() -> SignalReconciler:
    return SignalReconciler(max_tilt=0.015)


def test_empty_signals_neutral(reconciler: SignalReconciler):
    res = reconciler.reconcile_asset_signals("510300", [])
    assert res.conflict_type == "NO_SIGNAL_NEUTRAL"
    assert res.final_score == 0.0
    assert res.recommended_tilt == 0.0


def test_hard_veto_override(reconciler: SignalReconciler):
    # Regulatory veto or extreme QDII premium must override euphoric retail sentiment
    signals = [
        ChannelSignal("RETAIL_SOCIAL_US", score=+0.90, evidence="StockTwits extreme bullish"),
        ChannelSignal("SMART_MONEY_FLOW", score=+0.50, evidence="Inflows"),
    ]
    res = reconciler.reconcile_asset_signals("513100", signals, qdii_premium_pct=2.85)
    assert res.conflict_type == "CONFLICT_VETO_OVERRIDE"
    assert res.final_score == -1.0
    assert res.recommended_tilt == -0.015
    assert res.dominant_channel == "REGULATORY_POLICY"


def test_distribution_trap_detection(reconciler: SignalReconciler):
    # Smart money selling heavily while retail is euphoric -> Distribution Trap
    signals = [
        ChannelSignal("SMART_MONEY_FLOW", score=-0.75, evidence="Northbound dumping -6.2B"),
        ChannelSignal("RETAIL_SOCIAL_CN", score=+0.85, evidence="Retail FOMO buying"),
    ]
    res = reconciler.reconcile_asset_signals("510300", signals)
    assert res.conflict_type == "CONFLICT_DISTRIBUTION_TRAP"
    assert res.final_score < -0.5
    assert res.recommended_tilt == -0.015
    assert res.dominant_channel == "SMART_MONEY_FLOW"


def test_contrarian_bottom_detection(reconciler: SignalReconciler):
    # Fundamentals & Smart money accumulating while retail is in capitulation panic
    signals = [
        ChannelSignal("SMART_MONEY_FLOW", score=+0.60, evidence="Northbound buying +4.5B"),
        ChannelSignal("FUNDAMENTAL_VALUATION", score=+0.85, evidence="Dividend yield 4.8%"),
        ChannelSignal("RETAIL_SOCIAL_CN", score=-0.80, evidence="Retail panic selling"),
    ]
    res = reconciler.reconcile_asset_signals("510880", signals)
    assert res.conflict_type == "CONFLICT_CONTRARIAN_BOTTOM"
    assert res.final_score > 0.7
    assert res.recommended_tilt == +0.015


def test_cross_border_premium_disconnect(reconciler: SignalReconciler):
    # US bullish but domestic QDII premium elevated (1.8%)
    signals = [
        ChannelSignal("RETAIL_SOCIAL_US", score=+0.65, evidence="US tech rally"),
    ]
    res = reconciler.reconcile_asset_signals("513500", signals, qdii_premium_pct=1.80)
    assert res.conflict_type == "CONFLICT_CROSS_BORDER_PREMIUM"
    assert res.recommended_tilt < 0.0


def test_high_dispersion_uncertainty_damping(reconciler: SignalReconciler):
    # Moderate mixed signals with high disagreement -> dampened score
    signals = [
        ChannelSignal("SMART_MONEY_FLOW", score=+0.30),
        ChannelSignal("FUNDAMENTAL_VALUATION", score=-0.40),
        ChannelSignal("MAINSTREAM_NEWS", score=+0.50),
    ]
    res = reconciler.reconcile_asset_signals("510500", signals)
    assert res.disagreement_index > 0.35
    assert res.confidence_multiplier < 1.0


@pytest.mark.parametrize("channel", ["SMART_MONEY_FLOW", "FUNDAMENTAL_VALUATION", "MAINSTREAM_NEWS"])
@pytest.mark.parametrize("score", [float("nan"), float("inf"), -1.01, 1.01, True])
def test_invalid_scores_cannot_become_maximum_positive_tilt(reconciler, channel, score):
    with pytest.raises(ValueError):
        reconciler.reconcile_asset_signals("fixture", [ChannelSignal(channel, score)])


@pytest.mark.parametrize("signals", [
    [ChannelSignal("UNKNOWN", .3)],
    [ChannelSignal("SMART_MONEY_FLOW", .3), ChannelSignal("SMART_MONEY_FLOW", -.3)],
    [ChannelSignal("MAINSTREAM_NEWS", .3, confidence=float("nan"))],
    [ChannelSignal("MAINSTREAM_NEWS", .3, confidence=1.1)],
    [ChannelSignal("MAINSTREAM_NEWS", .3, veto_flag=True)],
])
def test_ambiguous_or_invalid_channels_are_rejected(reconciler, signals):
    with pytest.raises(ValueError):
        reconciler.reconcile_asset_signals("fixture", signals)


@pytest.mark.parametrize("options", [
    {"max_tilt": float("nan")}, {"max_tilt": -0.1},
    {"dispersion_damping": float("inf")}, {"dispersion_damping": -1},
])
def test_invalid_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        SignalReconciler(**options)


def test_nonfinite_premium_is_rejected_even_without_signals(reconciler):
    with pytest.raises(ValueError):
        reconciler.reconcile_asset_signals("fixture", [], qdii_premium_pct=float("nan"))
