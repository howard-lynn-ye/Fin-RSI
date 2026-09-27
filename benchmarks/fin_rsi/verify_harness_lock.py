#!/usr/bin/env python3
"""Fail-closed cryptographic verifier and anti-baseline-penalty auditor for Financial RSI sandboxes."""

from __future__ import annotations

import argparse
import pathlib
import sys

FIN_SKILLS_ROOT = pathlib.Path("/usr/local/google/home/shwaihe/fin-skills")
if str(FIN_SKILLS_ROOT) not in sys.path:
    sys.path.insert(0, str(FIN_SKILLS_ROOT))

from fin_skills.fin_rsi import REGISTERED_FIN_RSI_SEEDS, verify_rsi_harness_lock


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify or seal Financial RSI HARNESS_LOCK.json.")
    sub = parser.add_subparsers(dest="action", required=True)

    p_lock = sub.add_parser("lock", help="Seal HARNESS_LOCK.json for frozen_harness.py")
    p_lock.add_argument("--sandbox-dir", type=pathlib.Path, required=True)
    p_lock.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(REGISTERED_FIN_RSI_SEEDS),
    )

    p_check = sub.add_parser("check", help="Verify frozen_harness.py matches HARNESS_LOCK.json")
    p_check.add_argument("--sandbox-dir", type=pathlib.Path, required=True)

    args = parser.parse_args()
    seeds = getattr(args, "seeds", list(REGISTERED_FIN_RSI_SEEDS))
    res = verify_rsi_harness_lock(
        sandbox_dir=args.sandbox_dir.resolve(),
        action=args.action,
        seeds=seeds,
    )
    if not res["passed"]:
        print(f"[FAIL] {res.get('error', 'Verification failed')}", file=sys.stderr)
        for f in res.get("findings", []):
            print(f"  - {f}", file=sys.stderr)
        return 1

    if args.action == "lock":
        print(
            f"[PASS] Locked frozen_harness.py (SHA-256: {res['sha256']}) -> {res['lock_file']}"
        )
    else:
        print(
            f"[PASS] Harness lock verified (SHA-256: {res['sha256'][:16]}...) "
            f"& zero baseline penalties across {args.sandbox_dir.resolve().name}."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
