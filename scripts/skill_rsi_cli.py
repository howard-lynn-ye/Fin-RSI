#!/usr/bin/env python3
"""CLI Orchestrator for FinSkill RSI Scaffolding, Blueprint Application & 4-Gate Verification."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fin_skills.skill_rsi.blueprints import MACRO_FX_SKILL_BLUEPRINT  # noqa: E402
from fin_skills.skill_rsi.compiler import compile_and_verify_skills  # noqa: E402
from fin_skills.skill_rsi.scaffold import apply_skill_blueprint  # noqa: E402


def main() -> int:
    print("[1/2] Applying declarative SkillUpgradeBlueprint: macro-fx-industry-beta-shield ...")
    scaffold_res = apply_skill_blueprint(MACRO_FX_SKILL_BLUEPRINT, root=ROOT)
    if not scaffold_res.passed:
        print("FAIL during skill blueprint validation:", scaffold_res.validation_errors)
        return 1
    print(
        f"      Scaffolded {scaffold_res.skill_name} ({scaffold_res.plugin}) | "
        f"desc={scaffold_res.description_chars} chars | "
        f"xref_neighbors_updated={scaffold_res.xref_updated_neighbors}"
    )

    print("[2/2] Running 5-Stage Compilation, 4-Gate Verification & Cross-Repo Sync ...")
    report = compile_and_verify_skills(
        root=ROOT,
        probe_queries=MACRO_FX_SKILL_BLUEPRINT.probe_queries,
        sync_repos=True,
    )
    print(json.dumps({
        "passed": report["passed"],
        "live_counts": report["live_counts"],
        "gate1_spec_validate": report["gates"]["gate1_spec_validate"]["passed"],
        "gate2_trigger_routing": report["gates"]["gate2_trigger_routing"]["passed"],
        "gate3_blind_holdout": report["gates"]["gate3_blind_holdout"]["passed"],
        "gate4_operator_causality": report["gates"]["gate4_operator_causality"]["passed"],
        "probe_results": report["routing_audit"]["probe_results"],
        "sha256": report["sha256"],
    }, indent=2, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
