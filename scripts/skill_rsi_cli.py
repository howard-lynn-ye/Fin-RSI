#!/usr/bin/env python3
"""CLI Orchestrator for FinSkill RSI Scaffolding, Blueprint Application & 4-Gate Verification."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fin_skills.skill_rsi.blueprints import (  # noqa: E402
    LOB_LIQUIDITY_SKILL_BLUEPRINT,
    MACRO_FX_SKILL_BLUEPRINT,
)
from fin_skills.skill_rsi.compiler import compile_and_verify_skills  # noqa: E402
from fin_skills.skill_rsi.scaffold import apply_skill_blueprint  # noqa: E402


def main() -> int:
    blueprints = (MACRO_FX_SKILL_BLUEPRINT, LOB_LIQUIDITY_SKILL_BLUEPRINT)
    all_probes = []
    for idx, bp in enumerate(blueprints, start=1):
        print(f"[1.{idx}/2] Applying declarative SkillUpgradeBlueprint: {bp.name} ...")
        scaffold_res = apply_skill_blueprint(bp, root=ROOT)
        if not scaffold_res.passed:
            print(f"FAIL during skill blueprint validation ({bp.name}):", scaffold_res.validation_errors)
            return 1
        print(
            f"        Scaffolded {scaffold_res.skill_name} ({scaffold_res.plugin}) | "
            f"desc={scaffold_res.description_chars} chars | "
            f"xref_neighbors_updated={scaffold_res.xref_updated_neighbors}"
        )
        all_probes.extend(bp.probe_queries)

    print("[2/2] Running 5-Stage Compilation, 4-Gate Verification & Cross-Repo Sync ...")
    report = compile_and_verify_skills(
        root=ROOT,
        probe_queries=all_probes,
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
