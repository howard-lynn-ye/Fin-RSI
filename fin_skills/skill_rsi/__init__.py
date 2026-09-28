"""FinSkill Recursive Self-Improvement (RSI) Scaffolding & Upgrade Engine (`fin_skills.skill_rsi`).

Provides the unified foundation for creating, upgrading, auditing, and compiling FinSkills
across:
  1. Declarative Skill Scaffolding & In-Place Upgrades (`SkillUpgradeBlueprint`, `apply_skill_blueprint`)
  2. Contrastive Trigger & Blind-Holdout Routing Audits (`audit_catalog_routing`, `diagnose_query_collision`)
  3. Point-in-Time Executable Operator Registry & Future-Bar Causality Perturbation Gate
     (`SkillChannelSpec`, `DEFAULT_SKILL_OPERATOR_REGISTRY`, `verify_operator_causality`)
  4. Atomic 5-Stage Compiler & Cross-Repo Sync (`compile_and_verify_skills`)
"""
from __future__ import annotations

from fin_skills.skill_rsi.compiler import (
    compile_and_verify_skills,
    sync_cross_repo_package,
    sync_readme_live_counts,
)
from fin_skills.skill_rsi.registry import (
    DEFAULT_SKILL_OPERATOR_REGISTRY,
    CausalityAuditVerdict,
    SkillChannelSpec,
    SkillOperatorRegistry,
    SkillOperatorRole,
    make_synthetic_ashare_panel,
    verify_operator_causality,
)
from fin_skills.skill_rsi.routing_auditor import (
    MARGIN_FLOOR,
    RoutingAuditReport,
    audit_catalog_routing,
    diagnose_query_collision,
)
from fin_skills.skill_rsi.scaffold import (
    ScaffoldResult,
    SkillUpgradeBlueprint,
    apply_skill_blueprint,
    ensure_bidirectional_xref,
    render_skill_markdown,
    sync_marketplace_entry,
    validate_skill_blueprint,
)

__all__ = [
    "CausalityAuditVerdict",
    "DEFAULT_SKILL_OPERATOR_REGISTRY",
    "MARGIN_FLOOR",
    "RoutingAuditReport",
    "ScaffoldResult",
    "SkillChannelSpec",
    "SkillOperatorRegistry",
    "SkillOperatorRole",
    "SkillUpgradeBlueprint",
    "apply_skill_blueprint",
    "audit_catalog_routing",
    "compile_and_verify_skills",
    "diagnose_query_collision",
    "ensure_bidirectional_xref",
    "make_synthetic_ashare_panel",
    "render_skill_markdown",
    "sync_cross_repo_package",
    "sync_marketplace_entry",
    "sync_readme_live_counts",
    "validate_skill_blueprint",
    "verify_operator_causality",
]
