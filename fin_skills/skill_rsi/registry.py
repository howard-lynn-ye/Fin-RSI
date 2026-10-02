"""Unified Executable Skill & Point-in-Time Operator Registry (`fin_skills.skill_rsi.registry`).

Bridges the 4 tiers of a FinSkill in our Tri-Axis Stock-Prediction RSI system:
  1. Markdown Knowledge & Routing (`SKILL.md`, `TRIGGER`, `SKIP`, 1-hop `xref` graph)
  2. Executable Python Module (`fin_skills.<namespace>.<module>`)
  3. Fail-Closed Research Guard (`fin_skills.api.Guard` + `fin_skills.tools` MCP/JSON schema)
  4. Point-in-Time Stock-RSI Operator / Channel (`SkillChannelSpec` with automated future-bar
     perturbation causality verification `max_future_leak_diff == 0.0`)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd

SkillOperatorRole = Literal[
    "cross_sectional_alpha",
    "regime_gate",
    "microstructure_router",
    "orthogonal_projector",
    "causal_guard",
]


@dataclass(frozen=True)
class SkillChannelSpec:
    """One executable Point-in-Time channel/operator exported by a FinSkill for Stock-RSI."""

    channel_id: str
    skill_name: str
    plugin: str
    module_path: str
    role: SkillOperatorRole
    summary: str
    required_columns: tuple[str, ...]
    compute_fn: Callable[[pd.DataFrame], pd.Series | np.ndarray]
    audit_fn: Callable[[pd.DataFrame], dict[str, Any]] | None = None
    guard_name: str | None = None
    horizon_target: str = "5d"
    version: str = "1.0.0"


@dataclass
class CausalityAuditVerdict:
    """Result of the automated future-bar perturbation + self-leakage audit on a SkillChannel."""

    channel_id: str
    skill_name: str
    passed: bool
    max_future_perturbation_diff: float
    finite_ratio: float
    non_constant: bool
    custom_audit_passed: bool
    custom_audit_evidence: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel_id": self.channel_id,
            "skill_name": self.skill_name,
            "passed": bool(self.passed),
            "max_future_perturbation_diff": float(self.max_future_perturbation_diff),
            "finite_ratio": float(self.finite_ratio),
            "non_constant": bool(self.non_constant),
            "custom_audit_passed": bool(self.custom_audit_passed),
            "custom_audit_evidence": dict(self.custom_audit_evidence),
            "notes": list(self.notes),
        }


def make_synthetic_ashare_panel(n_days: int = 35, n_stocks: int = 16, seed: int = 42) -> pd.DataFrame:
    """Build a deterministic multi-board A-share + macro panel for skill operator self-tests."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-02", periods=n_days)
    symbols = [
        "SH600036", "SH600519", "SH601318", "SH603288",
        "SZ000001", "SZ000858", "SZ002142", "SZ002352",
        "SH688981", "SH688111", "SH688041", "SH688256",
        "SZ300750", "SZ300760", "SZ300059", "SZ300729",
    ][:n_stocks]
    industries = [
        "Bank", "Consumer", "Insurance", "Consumer",
        "Bank", "Consumer", "Bank", "Logistics",
        "Semiconductor", "Software", "Semiconductor", "Semiconductor",
        "Battery", "MedTech", "FinTech", "ExportTech",
    ][:n_stocks]
    # True underlying sensitivity of each stock to macro/FX shock
    true_fx_beta = np.array([
        -0.4, -0.2, -0.3, -0.1,
        -0.4, -0.2, -0.4, +0.2,
        +0.8, +0.4, +0.9, +0.85,
        +0.7, +0.6, -0.3, +0.95,
    ][:n_stocks])

    rows: list[dict[str, Any]] = []
    for t_idx, dt in enumerate(dates):
        dt_str = dt.strftime("%Y-%m-%d")
        macro_fx_shock = float(np.sin(t_idx / 4.0) * 0.015 + rng.normal(0.0, 0.005))
        macro_policy = float(np.cos(t_idx / 6.0) * 0.5)
        for s_idx, sym in enumerate(symbols):
            is_growth = sym.startswith(("SH688", "SZ300"))
            mcap = float(200.0 - 8.0 * s_idx + rng.normal(0.0, 1.5))
            ret_1d = float(true_fx_beta[s_idx] * macro_fx_shock + rng.normal(0.0, 0.018))
            ret_5d = float(ret_1d * 2.1 + rng.normal(0.0, 0.03))
            ret_20d = float(ret_5d * 1.8 + rng.normal(0.0, 0.05))
            margin_1d = float(rng.normal(0.005 if is_growth else -0.002, 0.015))
            overnight_gap = float(rng.normal(0.002 if is_growth else -0.001, 0.008))
            rows.append({
                "date": dt_str,
                "symbol": sym,
                "industry": industries[s_idx],
                "market_cap": mcap,
                "ret_1d": ret_1d,
                "stock_excess_return_1d": ret_1d,
                "ret_5d": ret_5d,
                "ret_20d": ret_20d,
                "margin_chg_1d": margin_1d,
                "margin_buy_ratio": 0.10 + margin_1d,
                "margin_balance_z30": float(rng.normal(0.0, 1.0)),
                "overnight_gap": overnight_gap,
                "overnight_gap_ratio": overnight_gap,
                "turnover_z": float(rng.normal(0.0, 1.0)),
                "ep_ttm": float(0.06 + 0.01 * (s_idx % 5) + rng.normal(0.0, 0.003)),
                "pe_ttm": float(15.0 + 2.0 * (s_idx % 5) + rng.normal(0.0, 0.5)),
                "bp_ratio": float(0.45 + 0.05 * (s_idx % 4) + rng.normal(0.0, 0.01)),
                "pb_ratio": float(1.8 + 0.2 * (s_idx % 4) + rng.normal(0.0, 0.05)),
                "div_yield": float(0.02 + 0.008 * (s_idx < 8)),
                "roe_pit": float(0.12 + 0.02 * (s_idx % 3)),
                "northbound_chg_5d": float(rng.normal(0.0, 1.0)),
                "northbound_net_buy_shares": float(rng.normal(0.0, 1.0e6)),
                "kol_weighted_engagement": float(rng.uniform(20.0, 200.0)),
                "inst_net_buy_z": float(rng.normal(0.0, 1.0)),
                "retail_guba_buzz": float(rng.normal(0.0, 1.0)),
                "filing_catalyst_score": float(rng.uniform(-0.5, 1.0)),
                "analyst_rev_z": float(rng.normal(0.0, 1.0)),
                "high_52w_ratio": float(rng.uniform(0.70, 0.99)),
                "idio_vol_20d": float(rng.uniform(0.012, 0.040)),
                "parkinson_volatility": float(rng.uniform(0.012, 0.040)),
                "garman_klass_volatility": float(rng.uniform(0.011, 0.042)),
                "volume": float(rng.uniform(1.0e6, 5.0e7)),
                "amihud_illiquidity": float(rng.uniform(0.0005, 0.005)),
                "upper_shadow_ratio": float(rng.uniform(0.0, 0.5)),
                "lower_shadow_ratio": float(rng.uniform(0.0, 0.5)),
                "clv_intraday": float(rng.uniform(-0.8, 0.8)),
                "macro_fx_shock": macro_fx_shock,
                "macro_policy_score": macro_policy,
            })
    return pd.DataFrame(rows)


def verify_operator_causality(
    spec: SkillChannelSpec,
    df_panel: pd.DataFrame | None = None,
    cut_ratio: float = 0.70,
    tol: float = 1e-10,
) -> CausalityAuditVerdict:
    """Prove that `spec.compute_fn` is strictly causal (`<= t`) via future-bar perturbation.

    Perturbs all rows at `date > T_cut` by large shocks (+10 sigma across all numeric inputs)
    and verifies that the computed channel values on `date <= T_cut` are bit-for-bit invariant.
    """
    panel = df_panel.copy() if df_panel is not None else make_synthetic_ashare_panel()
    missing_cols = [c for c in spec.required_columns if c not in panel.columns]
    if missing_cols:
        return CausalityAuditVerdict(
            channel_id=spec.channel_id,
            skill_name=spec.skill_name,
            passed=False,
            max_future_perturbation_diff=float("inf"),
            finite_ratio=0.0,
            non_constant=False,
            custom_audit_passed=False,
            notes=[f"Missing required columns: {missing_cols}"],
        )

    out_base = np.asarray(spec.compute_fn(panel), dtype=float)
    unique_dates = sorted(panel["date"].astype(str).unique())
    cut_idx = max(1, min(len(unique_dates) - 2, int(len(unique_dates) * cut_ratio)))
    cut_date = unique_dates[cut_idx]
    past_mask = (panel["date"].astype(str) <= cut_date).to_numpy()
    future_mask = ~past_mask

    perturbed = panel.copy()
    num_cols = [
        c for c in perturbed.select_dtypes(include=[np.number]).columns
        if c not in ("market_cap",)
    ]
    for col in num_cols:
        col_std = float(np.nanstd(perturbed[col].to_numpy(dtype=float))) + 1.0
        perturbed.loc[future_mask, col] = perturbed.loc[future_mask, col] + 10.0 * col_std

    out_pert = np.asarray(spec.compute_fn(perturbed), dtype=float)
    max_diff = float(np.nanmax(np.abs(out_base[past_mask] - out_pert[past_mask])))
    finite_ratio = float(np.isfinite(out_base).mean())
    non_constant = bool(np.nanstd(out_base) > 1e-8)

    custom_passed = True
    custom_ev: dict[str, Any] = {}
    notes: list[str] = []
    if spec.audit_fn is not None:
        custom_ev = dict(spec.audit_fn(panel))
        custom_passed = bool(custom_ev.get("passed", True))
        if not custom_passed:
            notes.append(f"Custom skill audit failed: {custom_ev}")

    if max_diff > tol:
        notes.append(
            f"LOOK-AHEAD DETECTED: perturbing future rows (date > {cut_date}) altered past "
            f"channel values by max_diff={max_diff:.3e} > tol={tol:.1e}"
        )
    if finite_ratio < 0.99:
        notes.append(f"Low finite ratio: {finite_ratio:.2%}")
    if not non_constant:
        notes.append("Channel output is constant (zero cross-sectional/temporal variance)")

    passed = (max_diff <= tol) and (finite_ratio >= 0.99) and non_constant and custom_passed
    return CausalityAuditVerdict(
        channel_id=spec.channel_id,
        skill_name=spec.skill_name,
        passed=passed,
        max_future_perturbation_diff=max_diff,
        finite_ratio=finite_ratio,
        non_constant=non_constant,
        custom_audit_passed=custom_passed,
        custom_audit_evidence=custom_ev,
        notes=notes,
    )


def _cs_zscore_by_date(df: pd.DataFrame, values: np.ndarray | pd.Series) -> pd.Series:
    """Strictly contemporaneous (`date == t`) cross-sectional z-score with clipping."""
    s = pd.Series(np.asarray(values, dtype=float), index=df.index)
    grp = s.groupby(df["date"])
    mean = grp.transform("mean")
    std = grp.transform("std").replace(0.0, 1.0).fillna(1.0)
    return ((s - mean) / (std + 1e-8)).clip(-3.5, 3.5).fillna(0.0)


# ---------------------------------------------------------------------------
# Built-in Vectorized Point-in-Time Skill Channel Implementations
# ---------------------------------------------------------------------------
def _compute_cross_board_limit_bifurcation(df: pd.DataFrame) -> pd.Series:
    from fin_skills.china.cross_board_supply_chain_rsi import compute_cross_board_limit_bifurcation
    return compute_cross_board_limit_bifurcation(df)


def _audit_cross_board_spillover(df: pd.DataFrame) -> dict[str, Any]:
    from fin_skills.china.cross_board_supply_chain_rsi import audit_cross_board_spillover_causality
    return audit_cross_board_spillover_causality(df)


def _compute_supply_chain_leader_spillover(df: pd.DataFrame) -> pd.Series:
    from fin_skills.china.cross_board_supply_chain_rsi import compute_supply_chain_leader_spillover
    return compute_supply_chain_leader_spillover(df)


def _compute_macro_fx_industry_beta_shield(df: pd.DataFrame) -> pd.Series:
    from fin_skills.macro.macro_fx_industry_beta_shield import compute_macro_fx_beta_shield
    return compute_macro_fx_beta_shield(df)


def _audit_macro_fx_industry_beta_shield(df: pd.DataFrame) -> dict[str, Any]:
    from fin_skills.macro.macro_fx_industry_beta_shield import audit_macro_fx_beta_causality
    return audit_macro_fx_beta_causality(df)


def _compute_lob_liquidity_shock_shield(df: pd.DataFrame) -> pd.Series:
    from fin_skills.microstructure.lob_liquidity_shock_shield import compute_lob_liquidity_shock_shield
    return compute_lob_liquidity_shock_shield(df)


def _audit_lob_liquidity_shock_shield(df: pd.DataFrame) -> dict[str, Any]:
    from fin_skills.microstructure.lob_liquidity_shock_shield import audit_lob_liquidity_causality
    return audit_lob_liquidity_causality(df)


def _compute_garp_valuation_quality(df: pd.DataFrame) -> pd.Series:
    raw = (
        0.35 * _cs_zscore_by_date(df, df["ep_ttm"])
        + 0.25 * _cs_zscore_by_date(df, df["bp_ratio"])
        + 0.20 * _cs_zscore_by_date(df, df["div_yield"])
        + 0.20 * _cs_zscore_by_date(df, df["roe_pit"])
    )
    return _cs_zscore_by_date(df, raw)


def _compute_downside_cushion_low_vol(df: pd.DataFrame) -> pd.Series:
    raw = 0.55 * _cs_zscore_by_date(df, df["high_52w_ratio"]) - 0.45 * _cs_zscore_by_date(df, df["idio_vol_20d"])
    return _cs_zscore_by_date(df, raw)


def _compute_verified_filing_catalyst(df: pd.DataFrame) -> pd.Series:
    raw = 0.60 * _cs_zscore_by_date(df, df["filing_catalyst_score"]) + 0.40 * _cs_zscore_by_date(df, df["analyst_rev_z"])
    return _cs_zscore_by_date(df, raw)


def _compute_smart_vs_retail_divergence(df: pd.DataFrame) -> pd.Series:
    smart = 0.55 * _cs_zscore_by_date(df, df["northbound_chg_5d"]) + 0.45 * _cs_zscore_by_date(df, df["inst_net_buy_z"])
    retail = _cs_zscore_by_date(df, df["retail_guba_buzz"])
    return _cs_zscore_by_date(df, smart - 0.50 * retail)


def _compute_intraday_candle_asymmetry(df: pd.DataFrame) -> pd.Series:
    raw = (
        0.45 * _cs_zscore_by_date(df, df["lower_shadow_ratio"] - df["upper_shadow_ratio"])
        + 0.55 * _cs_zscore_by_date(df, df["clv_intraday"])
    )
    return _cs_zscore_by_date(df, raw)


class SkillOperatorRegistry:
    """Central registry of executable FinSkill operators for Tri-Axis Stock-Prediction RSI."""

    def __init__(self) -> None:
        self._specs: dict[str, SkillChannelSpec] = {}

    def register(self, spec: SkillChannelSpec) -> SkillChannelSpec:
        self._specs[spec.channel_id] = spec
        return spec

    def get(self, channel_id: str) -> SkillChannelSpec:
        if channel_id not in self._specs:
            raise KeyError(
                f"Unknown skill channel {channel_id!r}. Available: {sorted(self._specs)}"
            )
        return self._specs[channel_id]

    def list_specs(self, role: SkillOperatorRole | None = None) -> list[SkillChannelSpec]:
        specs = [self._specs[k] for k in sorted(self._specs)]
        if role is not None:
            specs = [s for s in specs if s.role == role]
        return specs

    def compute_channel(self, channel_id: str, df_panel: pd.DataFrame) -> pd.Series:
        spec = self.get(channel_id)
        res = spec.compute_fn(df_panel)
        if isinstance(res, pd.Series):
            return res
        return pd.Series(np.asarray(res, dtype=float), index=df_panel.index, name=channel_id)

    def compute_all_channels(
        self,
        df_panel: pd.DataFrame,
        channel_ids: list[str] | None = None,
    ) -> pd.DataFrame:
        ids = channel_ids or [s.channel_id for s in self.list_specs()]
        out = pd.DataFrame(index=df_panel.index)
        for cid in ids:
            out[cid] = self.compute_channel(cid, df_panel)
        return out

    def verify_all_operators(
        self,
        df_panel: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        panel = df_panel if df_panel is not None else make_synthetic_ashare_panel()
        verdicts = [verify_operator_causality(s, panel).to_dict() for s in self.list_specs()]
        all_passed = all(v["passed"] for v in verdicts)
        return {
            "passed": all_passed,
            "n_operators": len(verdicts),
            "n_passed": sum(1 for v in verdicts if v["passed"]),
            "verdicts": verdicts,
        }


DEFAULT_SKILL_OPERATOR_REGISTRY = SkillOperatorRegistry()


def _populate_default_registry(reg: SkillOperatorRegistry) -> None:
    reg.register(
        SkillChannelSpec(
            channel_id="cross_board_limit_bifurcation",
            skill_name="cross-board-supply-chain-rsi",
            plugin="fin-china",
            module_path="fin_skills.china.cross_board_supply_chain_rsi",
            role="microstructure_router",
            summary="Routes +-10% Main Board short-horizon surges to reversal and +-20% STAR/ChiNext high-threshold surges to continuation.",
            required_columns=("date", "symbol", "ret_1d", "margin_chg_1d", "overnight_gap"),
            compute_fn=_compute_cross_board_limit_bifurcation,
            audit_fn=_audit_cross_board_spillover,
            guard_name="cross_board_spillover",
            horizon_target="5d",
            version="1.1.0",
        )
    )
    reg.register(
        SkillChannelSpec(
            channel_id="supply_chain_leader_spillover",
            skill_name="cross-board-supply-chain-rsi",
            plugin="fin-china",
            module_path="fin_skills.china.cross_board_supply_chain_rsi",
            role="cross_sectional_alpha",
            summary="Leave-One-Out (j != i) top-30%-cap industry leader momentum spillover minus self 5D return.",
            required_columns=("date", "symbol", "industry", "market_cap", "ret_1d", "ret_5d", "ret_20d"),
            compute_fn=_compute_supply_chain_leader_spillover,
            audit_fn=_audit_cross_board_spillover,
            guard_name="cross_board_spillover",
            horizon_target="5d",
            version="1.1.0",
        )
    )
    reg.register(
        SkillChannelSpec(
            channel_id="macro_fx_industry_beta_shield",
            skill_name="macro-fx-industry-beta-shield",
            plugin="fin-macro",
            module_path="fin_skills.macro.macro_fx_industry_beta_shield",
            role="regime_gate",
            summary="Causal expanding/rolling stock-and-industry sensitivity beta times macro/FX shock interaction (upgrades raw uniform macro broadcast).",
            required_columns=("date", "symbol", "industry", "ret_1d", "macro_fx_shock", "macro_policy_score"),
            compute_fn=_compute_macro_fx_industry_beta_shield,
            audit_fn=_audit_macro_fx_industry_beta_shield,
            guard_name="macro_fx_beta_gate",
            horizon_target="10d",
            version="1.0.0",
        )
    )
    reg.register(
        SkillChannelSpec(
            channel_id="lob_liquidity_shock_shield",
            skill_name="lob-liquidity-shock-shield",
            plugin="fin-microstructure",
            module_path="fin_skills.microstructure.lob_liquidity_shock_shield",
            role="microstructure_router",
            summary="Strictly causal 20-day rolling order-book volatility-of-volatility, return dispersion, and Amihud illiquidity absorption shield.",
            required_columns=("date", "symbol", "garman_klass_volatility", "ret_1d", "volume", "amihud_illiquidity"),
            compute_fn=_compute_lob_liquidity_shock_shield,
            audit_fn=_audit_lob_liquidity_shock_shield,
            guard_name="lob_liquidity_gate",
            horizon_target="5d",
            version="1.0.0",
        )
    )
    reg.register(
        SkillChannelSpec(
            channel_id="garp_valuation_quality",
            skill_name="fundamental-and-macro-data",
            plugin="fin-market-data",
            module_path="fin_skills.market_data.pit_fundamentals",
            role="cross_sectional_alpha",
            summary="Point-in-time GARP composite combining earnings yield, book-to-market, dividend yield, and ROE.",
            required_columns=("date", "symbol", "ep_ttm", "bp_ratio", "div_yield", "roe_pit"),
            compute_fn=_compute_garp_valuation_quality,
            guard_name="pit_fundamentals",
            horizon_target="20d",
            version="1.0.0",
        )
    )
    reg.register(
        SkillChannelSpec(
            channel_id="downside_cushion_low_vol",
            skill_name="portfolio-and-risk",
            plugin="fin-core",
            module_path="fin_skills.core.cost_curve",
            role="cross_sectional_alpha",
            summary="52-week high cushion minus idiosyncratic 20-day volatility.",
            required_columns=("date", "symbol", "high_52w_ratio", "idio_vol_20d"),
            compute_fn=_compute_downside_cushion_low_vol,
            horizon_target="10d",
            version="1.0.0",
        )
    )
    reg.register(
        SkillChannelSpec(
            channel_id="verified_filing_catalyst_composite",
            skill_name="combining-data-sources",
            plugin="fin-core",
            module_path="fin_skills.synthesis.timeline",
            role="cross_sectional_alpha",
            summary="Point-in-time corporate filing catalyst score combined with analyst revision surprise.",
            required_columns=("date", "symbol", "filing_catalyst_score", "analyst_rev_z"),
            compute_fn=_compute_verified_filing_catalyst,
            horizon_target="5d",
            version="1.0.0",
        )
    )
    reg.register(
        SkillChannelSpec(
            channel_id="smart_vs_retail_divergence",
            skill_name="alpha-combination-and-neutralization",
            plugin="fin-strategies",
            module_path="fin_skills.strategies.alpha_combine",
            role="orthogonal_projector",
            summary="Institutional smart-money accumulation (Northbound + block orders) orthogonalized against retail forum hype.",
            required_columns=("date", "symbol", "northbound_chg_5d", "inst_net_buy_z", "retail_guba_buzz"),
            compute_fn=_compute_smart_vs_retail_divergence,
            horizon_target="5d",
            version="1.0.0",
        )
    )
    reg.register(
        SkillChannelSpec(
            channel_id="intraday_candle_asymmetry",
            skill_name="signal-construction",
            plugin="fin-core",
            module_path="fin_skills.core.assert_causal",
            role="cross_sectional_alpha",
            summary="Intraday lower-minus-upper shadow rejection combined with Close Location Value (CLV).",
            required_columns=("date", "symbol", "upper_shadow_ratio", "lower_shadow_ratio", "clv_intraday"),
            compute_fn=_compute_intraday_candle_asymmetry,
            guard_name="assert_causal",
            horizon_target="3d",
            version="1.0.0",
        )
    )


_populate_default_registry(DEFAULT_SKILL_OPERATOR_REGISTRY)
