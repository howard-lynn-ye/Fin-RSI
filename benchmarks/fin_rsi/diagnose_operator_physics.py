#!/usr/bin/env python3
"""Automated physical diagnostic probes for Financial RSI operator evolution."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

FIN_SKILLS_ROOT = pathlib.Path("/usr/local/google/home/shwaihe/fin-skills")
if str(FIN_SKILLS_ROOT) not in sys.path:
    sys.path.insert(0, str(FIN_SKILLS_ROOT))

from fin_skills.fin_rsi import diagnose_financial_operator_physics


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Financial RSI physical diagnostic probes.")
    sub = parser.add_subparsers(dest="probe", required=True)
    sub.add_parser("probe-scale-cancellation")
    sub.add_parser("probe-ess")
    p_slice = sub.add_parser("probe-slice-gap")
    p_slice.add_argument("--summary-json", type=pathlib.Path, required=False, default=None)
    p_all = sub.add_parser("all")
    p_all.add_argument("--summary-json", type=pathlib.Path, required=False, default=None)

    args = parser.parse_args()
    summary_path = getattr(args, "summary_json", None)
    diag = diagnose_financial_operator_physics(summary_json_path=summary_path)

    if args.probe in ("probe-scale-cancellation", "all"):
        pa = diag["probe_a_scale_cancellation"]
        print(
            f"[PROBE A] Pre-LN Token Scaling Grad Norm  ||dL/dw_pre||_2  = {pa['pre_ln_grad_norm_l2']:.6e} (CANCELED: ~0)"
        )
        print(
            f"[PROBE A] Post-Encoder Value Pool Grad Norm ||dL/dw_post||_2 = {pa['post_encoder_value_pool_grad_norm_l2']:.6e} (ACTIVE: O(1))"
        )

    if args.probe in ("probe-ess", "all"):
        pb = diag["probe_b_sequence_ess"]
        print(
            "\nHorizon T | Unbounded exp(2.5*z) ESS | Bounded Value-Space + LSE ESS | Improvement"
        )
        print("-" * 86)
        for h_key, h_info in pb["horizons"].items():
            print(
                f"T = {h_info['horizon']:3d}   | {h_info['unbounded_exp_ess']:24.4f} | "
                f"{h_info['bounded_value_lse_ess']:29.4f} | {h_info['ess_improvement_ratio']:6.2f}x"
            )

    if args.probe in ("probe-slice-gap", "all"):
        pc = diag["probe_c_slice_gap"]
        print("\n=== Probe C: Sparse Small-Cap/STAR Slice & Crisis Regime Divergence ===")
        print(
            f"  • Sparse (n in {{1,2}}) Scalar Shrinkage MSE:      {pc['sparse_n1_2_scalar_shrinkage_mse']:.5f}"
        )
        print(
            f"  • Sparse (n in {{1,2}}) 16D Subspace + Stein MSE:  {pc['sparse_n1_2_subspace_stein_mse']:.5f} "
            f"(-{pc['sparse_mse_reduction_pct']:.2f}% error)"
        )
        if "evaluated_arms" in pc:
            for arm_name, m in pc["evaluated_arms"].items():
                print(
                    f"  • {arm_name:50s}: Macro IC={m['mean_daily_rank_ic']:+.5f} | "
                    f"Sparse IC={m['sparse_ticker_n1_2_rank_ic']:+.5f} | "
                    f"Crisis Sharpe={m['crisis_2018_2022_sharpe']:+.3f} | "
                    f"ESS={m['sequence_ess_ratio']:.2f}x"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
