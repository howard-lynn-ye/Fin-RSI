"""Unit tests for the FinSkill RSI Scaffolding & Upgrade Engine (`fin_skills.skill_rsi`)."""
from __future__ import annotations

import pandas as pd
import pytest

from fin_skills.api import check_cross_board_spillover, check_macro_fx_beta_gate
from fin_skills.skill_rsi import (
    DEFAULT_SKILL_OPERATOR_REGISTRY,
    SkillChannelSpec,
    SkillUpgradeBlueprint,
    audit_catalog_routing,
    diagnose_query_collision,
    make_synthetic_ashare_panel,
    validate_skill_blueprint,
    verify_operator_causality,
)
from fin_skills.skill_rsi.blueprints import MACRO_FX_SKILL_BLUEPRINT
from fin_skills.tools import call_tool, frame_to_payload


def test_blueprint_validator_enforces_agent_skills_spec():
    assert validate_skill_blueprint(MACRO_FX_SKILL_BLUEPRINT) == []

    bad_bp = SkillUpgradeBlueprint(
        name="Bad_Skill_Name",
        plugin="fin-macro",
        module_name="bad_mod",
        description="lowercase start without trigger or skip",
        body_markdown="# Missing script backtick reference",
        script_source="print('hi')",
        xref_neighbors=("nonexistent-neighbor",),
    )
    errs = validate_skill_blueprint(bad_bp)
    assert any("must match" in e for e in errs)
    assert any("third-person sentence" in e for e in errs)
    assert any("TRIGGER" in e for e in errs)
    assert any("SKIP" in e for e in errs)
    assert any("scripts/bad_mod.py" in e for e in errs)


def test_routing_auditor_and_collision_diagnosis():
    report = audit_catalog_routing(probe_queries=MACRO_FX_SKILL_BLUEPRINT.probe_queries)
    assert report.passed
    assert report.n_skills >= 131
    assert report.top1_accuracy == pytest.approx(1.0)
    assert len(report.thin_margins) == 0
    assert all(p["passed"] and p["margin"] >= 0.15 for p in report.probe_results)

    diag = diagnose_query_collision(
        "How do I compute compute_macro_fx_beta_shield for USD/CNH shock sensitivity?",
        expected_skill="macro-fx-industry-beta-shield",
    )
    assert diag["top1"] == "macro-fx-industry-beta-shield"
    assert diag["margin"] >= 0.15


def test_all_registered_skill_operators_pass_future_perturbation_causality():
    panel = make_synthetic_ashare_panel(n_days=25, n_stocks=12, seed=7)
    res = DEFAULT_SKILL_OPERATOR_REGISTRY.verify_all_operators(panel)
    assert res["passed"] is True
    assert res["n_operators"] == 8
    for v in res["verdicts"]:
        assert v["passed"] is True, v
        assert v["max_future_perturbation_diff"] <= 1e-10
        assert v["finite_ratio"] == pytest.approx(1.0)
        assert v["non_constant"] is True

    # Also verify compute_all_channels returns a clean DataFrame
    mat = DEFAULT_SKILL_OPERATOR_REGISTRY.compute_all_channels(panel)
    assert mat.shape == (len(panel), 8)
    assert not mat.isna().any().any()


def test_future_perturbation_gate_catches_planted_lookahead_operator():
    panel = make_synthetic_ashare_panel(n_days=20, n_stocks=8, seed=13)
    leaking_spec = SkillChannelSpec(
        channel_id="planted_future_leak",
        skill_name="research-integrity-guards",
        plugin="fin-core",
        module_path="fin_skills.core.assert_causal",
        role="cross_sectional_alpha",
        summary="Deliberately leaks tomorrow's return via shift(-1).",
        required_columns=("date", "symbol", "ret_1d"),
        compute_fn=lambda df: df.groupby("symbol")["ret_1d"].shift(-1).fillna(0.0),
    )
    verdict = verify_operator_causality(leaking_spec, panel)
    assert verdict.passed is False
    assert verdict.max_future_perturbation_diff > 1.0
    assert any("LOOK-AHEAD DETECTED" in n for n in verdict.notes)


def test_new_guards_work_in_python_and_over_json_mcp_tools():
    panel = make_synthetic_ashare_panel(n_days=15, n_stocks=8, seed=21)

    # 1. cross_board_spillover guard
    cb_pass = check_cross_board_spillover(panel)
    assert cb_pass.passed is True
    cb_json = call_tool("check_cross_board_spillover", {"panel": frame_to_payload(panel)})
    assert cb_json["passed"] is True

    cb_fail = check_cross_board_spillover(panel, include_self_in_peer=True)
    assert cb_fail.passed is False
    assert "SELF-LEAKAGE" in cb_fail.summary()

    # 2. macro_fx_beta_gate guard
    fx_pass = check_macro_fx_beta_gate(panel)
    assert fx_pass.passed is True
    fx_json = call_tool("check_macro_fx_beta_gate", {"panel": frame_to_payload(panel)})
    assert fx_json["passed"] is True

    defect_panel = panel.copy()
    defect_panel["uniform_macro"] = defect_panel["macro_fx_shock"]
    fx_fail = check_macro_fx_beta_gate(defect_panel, signal_col="uniform_macro")
    assert fx_fail.passed is False
    assert "Uniform 1-D macro broadcast" in fx_fail.summary()
