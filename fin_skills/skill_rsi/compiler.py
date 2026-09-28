"""Atomic 5-Stage Skill Compiler, 4-Gate Verifier & Cross-Repo Sync (`fin_skills.skill_rsi.compiler`).

Executes the end-to-end compilation and verification pipeline whenever a FinSkill is added or
upgraded:
  Stage 1: Rebuild catalog index (`scripts/build_index.py`) & quant methods (`scripts/export_quant_methods.py`)
  Stage 2: Regenerate `fin_skills/` importable package (`scripts/build_package.py`)
  Stage 3: Auto-sync `README.md` live Guard and Tool counts
  Stage 4: Run 4 Hard Verification Gates (`validate.py`, `eval_triggers.py`, `eval_blind.py`,
           and `DEFAULT_SKILL_OPERATOR_REGISTRY.verify_all_operators()`)
  Stage 5: Atomically sync `fin_skills/` to `stock_prediction/fin_skills/` and `.gemini/config/skills/`
"""
from __future__ import annotations

import hashlib
import importlib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from fin_skills.skill_rsi.registry import DEFAULT_SKILL_OPERATOR_REGISTRY
from fin_skills.skill_rsi.routing_auditor import audit_catalog_routing


def sync_readme_live_counts(root: Path) -> dict[str, int]:
    """Update `README.md` live guard and JSON tool counts so `validate.py` stays in sync."""
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import fin_skills.api as api
    import fin_skills.tools.runner as runner
    importlib.reload(api)
    importlib.reload(runner)

    n_guards = len(api.registry())
    n_tools = len(runner.list_tools())
    n_skills = len(list(root.glob("plugins/*/skills/*/SKILL.md")))

    readme_path = root / "README.md"
    if readme_path.exists():
        text = readme_path.read_text(encoding="utf-8")
        new_text = re.sub(
            r"(\d+)( guards that return a `GuardResult`)",
            rf"{n_guards}\2",
            text,
        )
        new_text = re.sub(
            r"(\d+)( tools an agent can call over JSON)",
            rf"{n_tools}\2",
            new_text,
        )
        if new_text != text:
            readme_path.write_text(new_text, encoding="utf-8", newline="\n")
    return {"n_skills": n_skills, "n_guards": n_guards, "n_tools": n_tools}


def sync_cross_repo_package(
    root: Path,
    stock_root: Path | None = None,
    gemini_skills_dir: Path | None = None,
) -> dict[str, Any]:
    """Copy the compiled `fin_skills/` package into `stock_prediction` and `.gemini/config/skills`."""
    stock_dir = stock_root or Path("/usr/local/google/home/shwaihe/stock_prediction")
    gemini_dir = gemini_skills_dir or Path("/usr/local/google/home/shwaihe/.gemini/config/skills")

    synced_stock = False
    if stock_dir.is_dir():
        dst_pkg = stock_dir / "fin_skills"
        if dst_pkg.exists():
            shutil.rmtree(dst_pkg)
        shutil.copytree(root / "fin_skills", dst_pkg)
        synced_stock = True

    synced_gemini_count = 0
    if gemini_dir.is_dir():
        for md in sorted(root.glob("plugins/*/skills/*/SKILL.md")):
            skill_name = md.parent.name
            target_skill_dir = gemini_dir / skill_name
            if target_skill_dir.exists() or skill_name in (
                "cross-board-supply-chain-rsi",
                "macro-fx-industry-beta-shield",
            ):
                target_skill_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(md, target_skill_dir / "SKILL.md")
                synced_gemini_count += 1

    return {
        "synced_to_stock_prediction": synced_stock,
        "stock_fin_skills_path": str(stock_dir / "fin_skills"),
        "synced_gemini_skills_count": synced_gemini_count,
    }


def _run_script(root: Path, rel_script: str, args: Sequence[str] = ()) -> dict[str, Any]:
    cmd = [sys.executable, str(root / rel_script), *args]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True)
    return {
        "script": rel_script,
        "returncode": proc.returncode,
        "passed": proc.returncode == 0,
        "elapsed_s": round(time.time() - t0, 3),
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def compile_and_verify_skills(
    root: Path | None = None,
    probe_queries: Sequence[dict[str, str]] | None = None,
    sync_repos: bool = True,
) -> dict[str, Any]:
    """Execute the full 5-Stage Build, 4-Gate Verification, and Cross-Repo Sync pipeline."""
    t0 = time.time()
    repo_root = root or Path(__file__).resolve().parent.parent.parent

    # Stage 1 & 2: Rebuild index, quant methods, and importable fin_skills/ package
    build_index_res = _run_script(repo_root, "scripts/build_index.py")
    build_pkg_res = _run_script(repo_root, "scripts/build_package.py")
    export_qm_res = _run_script(repo_root, "scripts/export_quant_methods.py")

    # Stage 3: Sync live guard & tool counts in README.md before running validate.py
    live_counts = sync_readme_live_counts(repo_root)

    # Stage 4: Execute the 4 Hard Gates
    gate1_validate = _run_script(repo_root, "scripts/validate.py")
    gate2_triggers = _run_script(repo_root, "scripts/eval_triggers.py")
    gate3_blind = _run_script(repo_root, "scripts/eval_blind.py")
    routing_report = audit_catalog_routing(repo_root, probe_queries=probe_queries)
    gate4_operators = DEFAULT_SKILL_OPERATOR_REGISTRY.verify_all_operators()

    # Stage 5: Cross-repo sync
    sync_status = (
        sync_cross_repo_package(repo_root)
        if sync_repos
        else {"synced_to_stock_prediction": False, "synced_gemini_skills_count": 0}
    )

    all_passed = (
        build_index_res["passed"]
        and build_pkg_res["passed"]
        and export_qm_res["passed"]
        and gate1_validate["passed"]
        and gate2_triggers["passed"]
        and gate3_blind["passed"]
        and routing_report.passed
        and gate4_operators["passed"]
    )

    elapsed = round(time.time() - t0, 3)
    report: dict[str, Any] = {
        "campaign": "fin_skills_rsi_scaffolding_and_upgrade",
        "passed": all_passed,
        "elapsed_seconds": elapsed,
        "live_counts": live_counts,
        "stages": {
            "build_index": build_index_res,
            "build_package": build_pkg_res,
            "export_quant_methods": export_qm_res,
        },
        "gates": {
            "gate1_spec_validate": gate1_validate,
            "gate2_trigger_routing": gate2_triggers,
            "gate3_blind_holdout": gate3_blind,
            "gate4_operator_causality": gate4_operators,
        },
        "routing_audit": routing_report.to_dict(),
        "cross_repo_sync": sync_status,
    }
    payload_bytes = json.dumps(report, sort_keys=True).encode("utf-8")
    report["sha256"] = hashlib.sha256(payload_bytes).hexdigest()[:16]

    out_path = repo_root / "benchmarks" / "SKILL_RSI_SCAFFOLD_REPORT.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report
