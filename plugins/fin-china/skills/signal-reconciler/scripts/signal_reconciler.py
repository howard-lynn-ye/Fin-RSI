"""Heuristic conflict rules for caller-supplied financial channel scores.

The fixed institutional-flow and valuation preference is a design assumption,
not verified source credibility or a profitable strategy. Branch names classify
input patterns; they do not establish market manipulation or a market bottom.
The China/QDII examples do not generalize automatically to other instruments.
No data collection, event-time validation, execution guard or order placement is
implemented here. Read the accompanying skill for input and confidence limits.
The __main__ demonstrations use invented observations, not investment results.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from numbers import Real
from typing import Any


# Legacy channel registry values, retained for compatibility. Numerical branches
# below do not read these values; changing them does not change the allocation.
CHANNEL_BASE_WEIGHTS = {
    "REGULATORY_POLICY": 1.00,      # Gatekeeper / Hard Veto
    "SMART_MONEY_FLOW": 0.40,       # Primary Institutional Anchor
    "FUNDAMENTAL_VALUATION": 0.35,  # Safety Margin Anchor
    "MAINSTREAM_NEWS": 0.10,        # Catalyst Context
    "RETAIL_SOCIAL_CN": 0.15,       # Contrarian Indicator (A-share/HK)
    "RETAIL_SOCIAL_US": 0.15,       # Contrarian Indicator (US Tech/StockTwits)
}


@dataclass(frozen=True)
class ChannelSignal:
    channel: str  # Key in CHANNEL_BASE_WEIGHTS
    score: float  # -1.0 (extreme bearish) to +1.0 (extreme bullish)
    confidence: float = 1.0  # 0.0 to 1.0; only social scores are weighted by this
    veto_flag: bool = False  # Only supported on REGULATORY_POLICY
    evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReconciliationResult:
    asset_code: str
    final_score: float  # -1.0 to +1.0
    recommended_tilt: float  # Within +/-max_tilt, default 0.015
    conflict_type: str
    disagreement_index: float  # 0.0 (unanimous) to 1.0 (polar opposite)
    confidence_multiplier: float  # 0.25 to 1.0; heuristic, not a probability
    dominant_channel: str
    resolution_rationale: str
    channel_breakdown: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SignalReconciler:
    """Hierarchical conflict resolver for multi-modal financial signals."""

    def __init__(self, max_tilt: float = 0.015, dispersion_damping: float = 1.2):
        _finite(max_tilt, "max_tilt", 0.0, 1.0)
        _finite(dispersion_damping, "dispersion_damping", 0.0)
        self.max_tilt = max_tilt
        self.dispersion_damping = dispersion_damping

    def reconcile_asset_signals(
        self,
        asset_code: str,
        signals: list[ChannelSignal],
        qdii_premium_pct: float = 0.0,
    ) -> ReconciliationResult:
        """Return a heuristic tilt from validated inputs; do not execute a trade."""
        _finite(qdii_premium_pct, "qdii_premium_pct")
        if not isinstance(signals, (list, tuple)):
            raise ValueError("signals must be a list or tuple")
        seen = set()
        for signal in signals:
            if not isinstance(signal, ChannelSignal):
                raise ValueError("signals must contain ChannelSignal values")
            if (not isinstance(signal.channel, str) or
                    signal.channel not in CHANNEL_BASE_WEIGHTS or signal.channel in seen):
                raise ValueError("channels must be recognized and unique")
            seen.add(signal.channel)
            _finite(signal.score, "score", -1.0, 1.0)
            _finite(signal.confidence, "confidence", 0.0, 1.0)
            if not isinstance(signal.veto_flag, bool):
                raise ValueError("veto_flag must be boolean")
            if signal.veto_flag and signal.channel != "REGULATORY_POLICY":
                raise ValueError("veto_flag is supported only on REGULATORY_POLICY")
        if not signals:
            return ReconciliationResult(
                asset_code=asset_code,
                final_score=0.0,
                recommended_tilt=0.0,
                conflict_type="NO_SIGNAL_NEUTRAL",
                disagreement_index=0.0,
                confidence_multiplier=1.0,
                dominant_channel="NONE",
                resolution_rationale="没有输入信号；返回零倾斜，不代表市场中性或已构造基准组合",
                channel_breakdown={},
            )

        sig_map: dict[str, ChannelSignal] = {s.channel: s for s in signals}
        breakdown = {s.channel: round(s.score, 3) for s in signals}

        # Extract key tier scores (default 0.0 if channel absent)
        reg_sig = sig_map.get("REGULATORY_POLICY")
        flow_score = sig_map["SMART_MONEY_FLOW"].score if "SMART_MONEY_FLOW" in sig_map else 0.0
        val_score = sig_map["FUNDAMENTAL_VALUATION"].score if "FUNDAMENTAL_VALUATION" in sig_map else 0.0
        news_score = sig_map["MAINSTREAM_NEWS"].score if "MAINSTREAM_NEWS" in sig_map else 0.0

        # Combine CN and US retail social if both exist
        social_scores = [
            s.score * s.confidence
            for k, s in sig_map.items()
            if k in ("RETAIL_SOCIAL_CN", "RETAIL_SOCIAL_US")
        ]
        social_score = sum(social_scores) / len(social_scores) if social_scores else 0.0

        # Compute Cross-Channel Disagreement Index (standard deviation of active scores)
        active_scores = [s.score for s in signals if abs(s.score) > 0.05]
        if len(active_scores) >= 2:
            mean_s = sum(active_scores) / len(active_scores)
            variance = sum((x - mean_s) ** 2 for x in active_scores) / len(active_scores)
            disagreement = min(1.0, round(math.sqrt(variance), 4))
        else:
            disagreement = 0.0

        # =====================================================================
        # ARCHETYPE 1: Hard Veto / Regulatory & Structural Circuit Breaker
        # =====================================================================
        if (reg_sig and (reg_sig.veto_flag or reg_sig.score <= -0.8)) or qdii_premium_pct >= 2.5:
            reason = (
                f"输入触发配置的否决规则(QDII溢价 {qdii_premium_pct:.2f}% 或政策分数/标志)，"
                f"输出防守倾斜建议；须另行验证适用规则，未执行交易"
            )
            return ReconciliationResult(
                asset_code=asset_code,
                final_score=-1.0,
                recommended_tilt=-self.max_tilt,
                conflict_type="CONFLICT_VETO_OVERRIDE",
                disagreement_index=disagreement,
                confidence_multiplier=1.0,
                dominant_channel="REGULATORY_POLICY",
                resolution_rationale=reason,
                channel_breakdown=breakdown,
            )

        # =====================================================================
        # ARCHETYPE 2: Cross-Border QDII Premium Disconnect (US Bullish vs CN Premium)
        # =====================================================================
        if qdii_premium_pct >= 1.5 and social_score > 0.3:
            reason = (
                f"输入社交分数偏正向({social_score:+.2f})且基金溢价达 {qdii_premium_pct:.2f}%，"
                f"按配置的溢价规则建议减配；须核对基金估值与适用市场，未执行交易"
            )
            return ReconciliationResult(
                asset_code=asset_code,
                final_score=-0.6,
                recommended_tilt=-round(self.max_tilt * 0.8, 4),
                conflict_type="CONFLICT_CROSS_BORDER_PREMIUM",
                disagreement_index=disagreement,
                confidence_multiplier=0.9,
                dominant_channel="REGULATORY_POLICY",
                resolution_rationale=reason,
                channel_breakdown=breakdown,
            )

        # =====================================================================
        # ARCHETYPE 3: Smart Money Distribution vs Retail FOMO Trap (主力派发 vs 散户接盘)
        # =====================================================================
        if flow_score <= -0.35 and social_score >= 0.45:
            # Heuristic pattern of opposing supplied flow and social scores.
            resolved = max(-1.0, flow_score - 0.35 * social_score)
            tilt = -self.max_tilt
            reason = (
                f"输入符合资金负向({flow_score:+.2f})、社交正向({social_score:+.2f})的规则分支，"
                f"按预设偏好建议减配；未验证机构意图或未来收益"
            )
            return ReconciliationResult(
                asset_code=asset_code,
                final_score=round(resolved, 4),
                recommended_tilt=tilt,
                conflict_type="CONFLICT_DISTRIBUTION_TRAP",
                disagreement_index=disagreement,
                confidence_multiplier=0.95,
                dominant_channel="SMART_MONEY_FLOW",
                resolution_rationale=reason,
                channel_breakdown=breakdown,
            )

        # =====================================================================
        # ARCHETYPE 4: Contrarian Bottom Accumulation (主力吸筹+低估 vs 散户恐慌割肉)
        # =====================================================================
        institutional_anchor = 0.55 * flow_score + 0.45 * val_score
        if institutional_anchor >= 0.25 and social_score <= -0.45:
            # Positive supplied anchor and negative social score; not a verified bottom.
            resolved = min(1.0, institutional_anchor - 0.25 * social_score)  # negative social boosts score
            tilt = +self.max_tilt
            reason = (
                f"输入符合锚定分数正向({institutional_anchor:+.2f})、社交负向({social_score:+.2f})的规则分支，"
                f"按预设偏好建议增配；未证实市场底部或未来收益"
            )
            return ReconciliationResult(
                asset_code=asset_code,
                final_score=round(resolved, 4),
                recommended_tilt=tilt,
                conflict_type="CONFLICT_CONTRARIAN_BOTTOM",
                disagreement_index=disagreement,
                confidence_multiplier=0.95,
                dominant_channel="FUNDAMENTAL_VALUATION" if val_score > flow_score else "SMART_MONEY_FLOW",
                resolution_rationale=reason,
                channel_breakdown=breakdown,
            )

        # =====================================================================
        # ARCHETYPE 5: Standard Hierarchical Synthesis with Disagreement Damping
        # =====================================================================
        # Notice retail social is weighted with a negative contrarian sign when extreme (>0.6),
        # or mild momentum when moderate.
        social_contribution = -0.15 * social_score if abs(social_score) > 0.6 else 0.05 * social_score
        raw_composite = (
            0.45 * flow_score
            + 0.35 * val_score
            + 0.10 * news_score
            + social_contribution
        )

        # Uncertainty Attenuation: high disagreement shrinks bet size toward zero
        conf_mult = round(math.exp(-self.dispersion_damping * (disagreement ** 2)), 4)
        conf_mult = max(0.25, min(1.0, conf_mult))

        final_score = round(max(-1.0, min(1.0, raw_composite * conf_mult)), 4)

        # Convert final_score into bounded weight tilt
        if abs(final_score) < 0.15:
            tilt = 0.0
        else:
            tilt = round(max(-self.max_tilt, min(self.max_tilt, final_score * self.max_tilt * 1.5)), 4)

        # Determine dominant channel
        dominant = "SMART_MONEY_FLOW"
        if abs(val_score) > abs(flow_score) and abs(val_score) > 0.2:
            dominant = "FUNDAMENTAL_VALUATION"

        if disagreement >= 0.45:
            conflict_label = "CONFLICT_HIGH_DISPERSION_DAMPED"
            reason = (
                f"多源信号存在分歧(分歧度 {disagreement:.2f})，启动不确定性阻尼(置信折算 {conf_mult:.0%})，"
                f"按配置权重计算倾斜；标签({dominant})不证明资金意图，未执行交易"
            )
        else:
            conflict_label = "NO_CONFLICT_CONSENSUS"
            reason = f"输入分散度较低({disagreement:.2f})，综合评分 {final_score:+.2f}；低分散度也可能来自输入不足"

        return ReconciliationResult(
            asset_code=asset_code,
            final_score=final_score,
            recommended_tilt=tilt,
            conflict_type=conflict_label,
            disagreement_index=disagreement,
            confidence_multiplier=conf_mult,
            dominant_channel=dominant,
            resolution_rationale=reason,
            channel_breakdown=breakdown,
        )


ReconciledSignal = ReconciliationResult


def _finite(value, name, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    if ((minimum is not None and value < minimum) or
            (maximum is not None and value > maximum)):
        raise ValueError(f"{name} is outside its allowed range")


def compute_belief_entropy(probabilities: list[float]) -> float:
    """Compute Shannon belief entropy across multi-channel probability weights: -sum(p * log2(p)).

    Entropy measures concentration of these masses, not agreement of directional signs.
    This helper is not used by reconcile_asset_signals to choose a branch.
    """
    valid = [p for p in probabilities if p > 0.0]
    if not valid:
        return 0.0
    total = sum(valid)
    norm = [p / total for p in valid]
    return round(-sum(p * math.log2(p) for p in norm if p > 0), 4)


def reconcile_views(
    asset_code: str,
    signals: list[ChannelSignal],
    qdii_premium_pct: float = 0.0,
    reconciler: SignalReconciler | None = None,
) -> ReconciliationResult:
    """Apply the documented heuristic to supplied channel scores.

    Applies the 5-tier hierarchical conflict resolution rules:
    1. Configured policy/QDII veto threshold
    2. Configured premium and positive social-score threshold
    3. Negative flow and positive social-score pattern
    4. Positive anchor and negative social-score pattern
    5. Fixed score combination with dispersion damping

    These branches do not validate source credibility or predict profitable trades.

    Args:
        asset_code: Security ticker (e.g. "510300", "513100").
        signals: List of ChannelSignal observations from various market channels.
        qdii_premium_pct: Domestic ETF market premium percentage over IOPV/NAV.
        reconciler: Optional pre-configured SignalReconciler instance.

    Returns:
        ReconciliationResult (or ReconciledSignal) with resolved score, tilt, and rationale.
    """
    if reconciler is None:
        reconciler = SignalReconciler()
    return reconciler.reconcile_asset_signals(asset_code, signals, qdii_premium_pct=qdii_premium_pct)


__all__ = [
    "CHANNEL_BASE_WEIGHTS",
    "ChannelSignal",
    "ReconciliationResult",
    "ReconciledSignal",
    "SignalReconciler",
    "compute_belief_entropy",
    "reconcile_views",
]


if __name__ == "__main__":
    reconciler = SignalReconciler(max_tilt=0.015)

    print("Testing Multi-Source Conflict Resolution Engine:\n")

    # Case 1: Distribution Trap (Smart Money Selling vs Retail Euphoria)
    case1 = [
        ChannelSignal("SMART_MONEY_FLOW", score=-0.75, evidence="synthetic: Northbound outflow -6.2B RMB"),
        ChannelSignal("FUNDAMENTAL_VALUATION", score=-0.20, evidence="synthetic: PE at 75th percentile"),
        ChannelSignal("RETAIL_SOCIAL_CN", score=+0.85, evidence="synthetic: Xueqiu retail screaming 'To the moon!'"),
    ]
    res1 = reconciler.reconcile_asset_signals("510300", case1)
    print(f"[Case 1: 510300 Distribution Trap]")
    print(f"  Conflict Type: {res1.conflict_type} | Disagreement: {res1.disagreement_index:.2f}")
    print(f"  Final Score: {res1.final_score:+.2f} | Recommended Tilt: {res1.recommended_tilt*100:+.2f}%")
    print(f"  Rationale: {res1.resolution_rationale}\n")

    # Case 2: Contrarian Bottom (Smart Money + Valuation Buying vs Retail Capitulation Panic)
    case2 = [
        ChannelSignal("SMART_MONEY_FLOW", score=+0.60, evidence="synthetic: Northbound inflow +4.5B RMB"),
        ChannelSignal("FUNDAMENTAL_VALUATION", score=+0.80, evidence="synthetic: Dividend yield 4.8%, 10y bottom"),
        ChannelSignal("RETAIL_SOCIAL_CN", score=-0.85, evidence="synthetic: Retail capitulation panic selling"),
    ]
    res2 = reconciler.reconcile_asset_signals("510880", case2)
    print(f"[Case 2: 510880 Contrarian Bottom]")
    print(f"  Conflict Type: {res2.conflict_type} | Disagreement: {res2.disagreement_index:.2f}")
    print(f"  Final Score: {res2.final_score:+.2f} | Recommended Tilt: {res2.recommended_tilt*100:+.2f}%")
    print(f"  Rationale: {res2.resolution_rationale}\n")

    # Case 3: Cross-Border Disconnect (US StockTwits Bullish vs Domestic QDII High Premium)
    case3 = [
        ChannelSignal("RETAIL_SOCIAL_US", score=+0.75, evidence="synthetic: StockTwits $NVDA/$QQQ 85% Bullish"),
        ChannelSignal("SMART_MONEY_FLOW", score=+0.20, evidence="synthetic: Moderate inflow"),
    ]
    res3 = reconciler.reconcile_asset_signals("513100", case3, qdii_premium_pct=2.80)
    print(f"[Case 3: 513100 Cross-Border QDII Veto]")
    print(f"  Conflict Type: {res3.conflict_type} | Disagreement: {res3.disagreement_index:.2f}")
    print(f"  Final Score: {res3.final_score:+.2f} | Recommended Tilt: {res3.recommended_tilt*100:+.2f}%")
    print(f"  Rationale: {res3.resolution_rationale}\n")
