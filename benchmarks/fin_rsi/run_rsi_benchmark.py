#!/usr/bin/env python3
"""Layer 3 Orchestrator for the Governed Financial RSI (`Fin-RSI`) Campaign.

Execution Flow:
  Step 0: Runs `verify_harness_lock.py check` (Fail-Closed SHA-256 & Anti-Penalty Verification).
  Step 1: Runs the 3 Financial Physical Diagnostic Probes (`diagnose_financial_operator_physics`).
  Step 2: Evaluates Row 1..4 (Frozen Harness) + Row 5..7 (`Gen-1`, `Gen-2`, `Gen-3 Champion`)
          across M=5 registered seeds `[20260923, 20260924, 20260925, 20260926, 20260927]`
          on the 207,742-row balanced panel and 107,999 real OOS return observations.
  Step 3: Evaluates the 3-Way Financial Pareto Promotion Gate (`evaluate_rsi_pareto_gate`).
  Step 4: Persists canonical receipts and updates `.agents/SHARED_CONTEXT_BLACKBOARD.md`.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import pathlib
import sys

os.environ["TMPDIR"] = "/usr/local/google/home/shwaihe/tmp"

SANDBOX_DIR = pathlib.Path(__file__).resolve().parent
FIN_SKILLS_ROOT = pathlib.Path("/usr/local/google/home/shwaihe/fin-skills")
STOCK_PRED_ROOT = pathlib.Path("/usr/local/google/home/shwaihe/stock_prediction")

for p in (SANDBOX_DIR, FIN_SKILLS_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from evaluate_pareto_gate import format_markdown_pareto_table
from fin_skills.fin_rsi import (
    diagnose_financial_operator_physics,
    evaluate_rsi_pareto_gate,
    verify_rsi_harness_lock,
)
from frozen_harness import evaluate_all_rsi_arms
from mutable_operator import build_mutable_rsi_generations


def write_shared_blackboard(
    target_dir: pathlib.Path, summary: dict, gate_res: dict, md_report: str
) -> None:
    bb_dir = target_dir / ".agents"
    bb_dir.mkdir(parents=True, exist_ok=True)
    bb_path = bb_dir / "SHARED_CONTEXT_BLACKBOARD.md"
    bb_path.write_text(md_report, encoding="utf-8")


def main() -> int:
    # Step 0: Mandatory Pre-Run Cryptographic Lock & Zero-Penalty Verification
    lock_check = verify_rsi_harness_lock(sandbox_dir=SANDBOX_DIR, action="check")
    if not lock_check["passed"]:
        print(f"[FAIL-CLOSED] Step 0 Harness Lock Check Failed: {lock_check}", file=sys.stderr)
        return 1
    print(
        f"[STEP 0 PASS] Frozen harness SHA-256 verified ({lock_check['sha256'][:16]}...) "
        "& zero artificial baseline penalties."
    )

    # Step 1: Run Physical Diagnostic Probes before/across operator evolution
    candidates = build_mutable_rsi_generations(embed_dim=16)

    # Step 2: Run M=5 Multi-Seed Evaluation across Row 1..7
    eval_summary = evaluate_all_rsi_arms(
        candidate_operators=candidates,
        ledger_dir=SANDBOX_DIR / "ledger",
    )
    eval_summary["generated_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    eval_summary["harness_lock_verified"] = True
    eval_summary["frozen_harness_sha256"] = lock_check["sha256"]

    # Attach Physical Probes with evaluated arm slice diagnostics
    tmp_summary_path = SANDBOX_DIR / "rsi_summary.json"
    tmp_summary_path.write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")
    probes = diagnose_financial_operator_physics(summary_json_path=tmp_summary_path)
    eval_summary["physical_diagnostic_probes"] = probes

    # Step 3: Evaluate 3-Way Pareto Gate
    pareto_verdict = evaluate_rsi_pareto_gate(summary_data=eval_summary)
    eval_summary["pareto_gate_evaluation"] = pareto_verdict

    # Compute paired improvements of Gen-3 Champion over Row 1, Row 2, and Row 4
    arms = eval_summary["arms"]
    r1 = arms["Row_1_Full_Dense_Multimodal_Ref"]
    r2 = arms["Row_2_Prod_Baseline_Verbal_Reflexion_RSI"]
    r4 = arms["Row_4_Prod_Baseline_JEV_SystemOne_Static_64KC"]
    r7 = arms["Row_7_RSI_Gen3_Streaming_Woodbury_Fisher_JEV_64KC"]

    # Paired t-statistic across M=5 seeds (Gen-3 Champion vs. Static JEV-64KC Row 4)
    r7_ics = [x["mean_daily_rank_ic"] for x in r7["per_seed"]]
    r4_ics = [x["mean_daily_rank_ic"] for x in r4["per_seed"]]
    diffs = [a - b for a, b in zip(r7_ics, r4_ics)]
    import numpy as np

    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))
    paired_t_vs_r4 = float(mean_diff / max(std_diff / np.sqrt(len(diffs)), 1e-8))

    eval_summary["headline_comparisons"] = {
        "gen3_vs_row4_static_jev_64kc": {
            "delta_daily_rank_ic": round(r7["mean_daily_rank_ic"] - r4["mean_daily_rank_ic"], 5),
            "delta_net_sharpe": round(
                r7["annualized_net_sharpe"] - r4["annualized_net_sharpe"], 3
            ),
            "delta_sparse_ticker_ic": round(
                r7["sparse_ticker_n1_2_rank_ic"] - r4["sparse_ticker_n1_2_rank_ic"], 5
            ),
            "delta_crisis_sharpe": round(
                r7["crisis_2018_2022_sharpe"] - r4["crisis_2018_2022_sharpe"], 3
            ),
            "paired_t_stat_ic": round(paired_t_vs_r4, 2),
        },
        "gen3_vs_row2_unguarded_verbal_reflexion": {
            "delta_daily_rank_ic": round(r7["mean_daily_rank_ic"] - r2["mean_daily_rank_ic"], 5),
            "delta_net_sharpe": round(
                r7["annualized_net_sharpe"] - r2["annualized_net_sharpe"], 3
            ),
            "drawdown_reduction_pct_pts": round(
                abs(r2["max_drawdown_pct"]) - abs(r7["max_drawdown_pct"]), 2
            ),
            "leakage_reduction_pct_pts": round(
                r2["leakage_rate_pct"] - r7["leakage_rate_pct"], 1
            ),
        },
        "gen3_retention_vs_row1_full_dense_pct": r7["retention_vs_row1_pct"],
    }

    payload_bytes = json.dumps(eval_summary, indent=2, sort_keys=True).encode("utf-8")
    eval_summary["receipt_sha256"] = hashlib.sha256(payload_bytes).hexdigest()[:16]

    json_text = json.dumps(eval_summary, indent=2, ensure_ascii=False) + "\n"
    md_text = format_markdown_pareto_table(eval_summary, pareto_verdict)

    # Step 4: Write all canonical evidence outputs
    out_paths_json = [
        STOCK_PRED_ROOT / "rsi_campaign/rsi_summary.json",
        STOCK_PRED_ROOT / "data/benchmark/FIN_RSI_PARETO_LEDGER_REPORT.json",
        FIN_SKILLS_ROOT / "benchmarks/FIN_RSI_PARETO_LEDGER_REPORT.json",
        FIN_SKILLS_ROOT / "benchmarks/fin_rsi/rsi_summary.json",
    ]
    for p in out_paths_json:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json_text, encoding="utf-8")

    md_path = STOCK_PRED_ROOT / "data/benchmark/FIN_RSI_PARETO_LEDGER_REPORT.md"
    md_path.write_text(md_text, encoding="utf-8")

    write_shared_blackboard(STOCK_PRED_ROOT / "rsi_campaign", eval_summary, pareto_verdict, md_text)
    write_shared_blackboard(
        FIN_SKILLS_ROOT / "benchmarks/fin_rsi", eval_summary, pareto_verdict, md_text
    )

    print(md_text)
    print(f"[PASS] Saved FIN_RSI_PARETO_LEDGER_REPORT.json (SHA-256: {eval_summary['receipt_sha256']})")
    return 0 if pareto_verdict["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
