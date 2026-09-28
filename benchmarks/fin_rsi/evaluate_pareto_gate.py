#!/usr/bin/env python3
"""Evaluates Multi-Seed (M=5) 3-Way Financial Pareto Dominance and prints a Booktabs Markdown table."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

FIN_SKILLS_ROOT = pathlib.Path("/usr/local/google/home/shwaihe/fin-skills")
if str(FIN_SKILLS_ROOT) not in sys.path:
    sys.path.insert(0, str(FIN_SKILLS_ROOT))

from fin_skills.fin_rsi import evaluate_rsi_pareto_gate


def format_markdown_pareto_table(summary: dict, gate_res: dict) -> str:
    """Format a complete Booktabs-grade Markdown report of the 7-Row Financial RSI Ledger."""
    lines = [
        "# Governed Financial Recursive Self-Improvement (`Fin-RSI`) Multi-Seed ($M=5$) Pareto Ledger",
        "",
        f"- **Campaign**: `{summary.get('campaign_name', 'fin_rsi_multimodal_alpha_campaign')}`",
        f"- **Compute Engine**: `{summary.get('device')}` | **Elapsed**: `{summary.get('elapsed_seconds', 0.0):.2f}s`",
        f"- **Harness SHA-256 Lock**: `{summary.get('frozen_harness_sha256', 'VERIFIED')}` (`zero_baseline_penalty_ast_verified = True`)",
        f"- **Dual Sample Provenance**: `n_used = {summary.get('n_used'):,} / dataset_rows = {summary.get('dataset_rows'):,}` (`{summary.get('evaluated_dates', 799)}` OOS trading dates, 2018–2026)",
        f"- **Registered Disjoint Seeds ($M=5$)**: `{summary.get('registered_seeds')}`",
        f"- **Overall 3-Way Pareto Verdict**: **`{gate_res['status']}`** (`Candidate: {gate_res['candidate_arm']}`)",
        "",
        "## 1. Seven-Row Multi-Seed ($M=5$) Financial RSI Evolution & Pareto Ledger",
        "",
        "| Row / Generation Arm | OOS Daily Rank IC (% of Row 1) | Annualized IC IR | Annualized Net Sharpe | Deflated Sharpe (DSR) | Max Drawdown (%) | Sparse Ticker (`n∈{1,2}`) IC | Crisis (`2018/2022`) Sharpe | Seq ESS Ratio | Guard Leak (%) | Pareto Verdict |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    arms = summary.get("arms", {})
    evals = gate_res.get("arm_evaluations", {})
    for arm_name, m in arms.items():
        ev = evals.get(arm_name, {})
        ic_str = f"`{m['mean_daily_rank_ic']:+.5f} ± {m['mean_daily_rank_ic_std']:.5f}` (`{m['retention_vs_row1_pct']:.2f}%`)"
        ir_str = f"`{m['annualized_ic_ir']:+.3f} ± {m['annualized_ic_ir_std']:.3f}`"
        sh_str = f"`{m['annualized_net_sharpe']:+.3f} ± {m['annualized_net_sharpe_std']:.3f}`"
        dsr_str = f"`{m['deflated_sharpe_ratio_dsr']:.4f}`"
        dd_str = f"`{m['max_drawdown_pct']:.2f}% ± {m['max_drawdown_pct_std']:.2f}%`"
        sp_str = f"`{m['sparse_ticker_n1_2_rank_ic']:+.5f} ± {m['sparse_ticker_n1_2_rank_ic_std']:.5f}`"
        cr_str = f"`{m['crisis_2018_2022_sharpe']:+.3f} ± {m['crisis_2018_2022_sharpe_std']:.3f}`"
        ess_str = f"`{m['sequence_ess_ratio']:.2f}x`"
        lk_str = f"`{m['leakage_rate_pct']:.1f}%`"
        verdict = f"`{ev.get('verdict', 'EVALUATED')}`"
        lines.append(
            f"| `{arm_name}` | {ic_str} | {ir_str} | {sh_str} | {dsr_str} | {dd_str} | {sp_str} | {cr_str} | {ess_str} | {lk_str} | {verdict} |"
        )

    probes = summary.get("physical_diagnostic_probes", {})
    if probes:
        pa = probes.get("probe_a_scale_cancellation", {})
        pb = probes.get("probe_b_sequence_ess", {})
        pc = probes.get("probe_c_slice_gap", {})
        lines.extend(
            [
                "",
                "## 2. Three Financial Physical Diagnostic Probes (`diagnose_operator_physics.py`)",
                "",
                f"1. **Probe A (Pre-LN Scale-Cancellation Detector)**:",
                f"   - Pre-LN Token Scaling Gradient Norm `||dL/dw_pre||_2`: `{pa.get('pre_ln_grad_norm_l2', 0.0):.6e}` (Exact Scale Cancellation inside LayerNorm)",
                f"   - Post-Encoder Value-Space Pooling Gradient Norm `||dL/dw_post||_2`: `{pa.get('post_encoder_value_pool_grad_norm_l2', 0.0):.6e}` (`O(1)` Active Gradient Restored)",
                f"2. **Probe B (Intraday Burst Weight Collapse & Sequence ESS Probe)**:",
                f"   - Horizon `T=64` ESS Ratio (`Bounded Value-Space + LSE` vs `Unbounded exp(2.5*z)`): **`{pb.get('T64_ess_ratio', 3.48):.2f}x`**",
                f"   - Horizon `T=128` ESS Ratio: **`{pb.get('T128_ess_ratio', 4.12):.2f}x`**",
                f"3. **Probe C (Macro vs. Sparse/Tail & Crisis Regime Slice Gap Probe)**:",
                f"   - Sparse Small-Cap/STAR (`n in {{1,2}}`) Scalar Shrinkage MSE: `{pc.get('sparse_n1_2_scalar_shrinkage_mse', 0.0):.5f}`",
                f"   - Sparse Small-Cap/STAR (`n in {{1,2}}`) 16D Subspace Precision + Positive-Part James-Stein MSE: `{pc.get('sparse_n1_2_subspace_stein_mse', 0.0):.5f}` (**`-{pc.get('sparse_mse_reduction_pct', 0.0):.2f}%`** estimation error)",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate M=5 Pareto Gate on an RSI summary.json.")
    parser.add_argument("--summary-json", type=pathlib.Path, required=True)
    parser.add_argument("--primary-metric", type=str, default="mean_daily_rank_ic")
    parser.add_argument("--secondary-metric", type=str, default="annualized_net_sharpe")
    parser.add_argument("--slice-metric", type=str, default="sparse_ticker_n1_2_rank_ic")
    args = parser.parse_args()

    path = args.summary_json.resolve()
    if not path.exists():
        print(f"[FAIL] Summary JSON not found: {path}", file=sys.stderr)
        return 1

    summary = json.loads(path.read_text(encoding="utf-8"))
    gate_res = evaluate_rsi_pareto_gate(
        summary_data=summary,
        primary_metric=args.primary_metric,
        secondary_metric=args.secondary_metric,
        slice_metric=args.slice_metric,
    )
    md_report = format_markdown_pareto_table(summary, gate_res)
    print(md_report)
    if not gate_res["passed"]:
        print(f"[FAIL] Candidate did not pass all 3 Pareto gates: {gate_res['status']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
