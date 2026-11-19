"""P-fix runner (exploratory): one model, inside a Slurm GPU allocation.

Design: docs/P_FIX_STUDY_DESIGN_20261118.md. Units are disjoint:

  check_F0   no library (identical messages to P-adopt L0), 18 episodes
  check_fix  the current library: F1 (interface text) and F2 (F1 + pointer), 36 episodes

Everything else is P-adopt's runner unchanged (scripts/run_p_adopt.py): units write under
<output>/<main|dev>/<model_id>/<unit>/job-<SLURM_JOB_ID>/, a unit with a complete terminal
receipt is skipped on resubmission, the manifest must match earlier jobs of the model, every
declared first prompt must reach the model, and --dev runs one development task and seed only
to check the pipeline (no grade is read).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fin_skills.rsi import adopt, pfix  # noqa: E402
from fin_skills.rsi.run_record import digest  # noqa: E402
from scripts.run_p_adopt import (FIT_SEEDS, check_first_prompts, complete, design,  # noqa: E402
                                 load_model, sha, write)

UNITS = {"check_F0": ((pfix.CheckF0,), False),
         "check_fix": ((pfix.CheckF1, pfix.CheckF2), True),
         "check_doc": ((pfix.CheckF3,), True),
         "check_route": ((pfix.CheckF4,), True)}
PARTS = {"check": ("check_F0", "check_fix"), "check_F0": ("check_F0",),
         "check_fix": ("check_fix",), "check_doc": ("check_doc",),
         "check_route": ("check_route",),
         "all": ("check_F0", "check_fix", "check_route", "check_doc")}


def run(args):
    from fin_skills.rsi import benchmark
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        raise RuntimeError("P-fix execution requires a Slurm allocation")
    tasks, seeds, fit_budget = design(args.dev)
    manifest = pfix.manifest(tasks.heldout, seeds=seeds)
    manifest.update(mode="dev-pipeline-check" if args.dev else "main")
    root = Path(args.output) / ("dev" if args.dev else "main") / args.model_id
    restart = int(os.environ.get("SLURM_RESTART_COUNT", "0") or 0)
    run_id = f"job-{job}" + (f"-r{restart}" if restart else "")
    receipt_path = root / f"{run_id}.json"
    receipt = dict(job=job, run_id=run_id, model_id=args.model_id, part=args.part, dev=args.dev,
                   cluster=os.environ.get("SLURM_CLUSTER_NAME"),
                   confinement=os.environ.get("FIN_TASKS_CONFINE") or "none",
                   manifest_sha256=digest(manifest), runner_sha256=sha(__file__),
                   pfix_sha256=sha(pfix.__file__), started=time.time(), units={})
    if not (root / "manifest.json").exists():
        write(root / "manifest.json", manifest)
    if json.loads((root / "manifest.json").read_text(encoding="utf-8")) != json.loads(
            json.dumps(manifest)):
        raise ValueError("manifest differs from the one this model's earlier jobs used")
    todo = [u for u in PARTS[args.part] if not complete(root / u)]
    receipt["skipped"] = [u for u in PARTS[args.part] if u not in todo]
    write(receipt_path, receipt)
    if not todo:
        print(json.dumps(dict(status="nothing_to_do", **receipt)))
        return receipt
    model, backend, config = load_model(args.model_id, args.protocol, args.inventory,
                                        adopt.OUTPUT_CAP)
    receipt.update(model=model, backend=config, loaded=time.time())
    write(receipt_path, receipt)
    for unit in todo:
        methods, library = UNITS[unit]
        out = root / unit / run_id / f"{unit}.jsonl"
        started, first = time.time(), len(backend.forwarded)
        report = benchmark(methods, backend, tasks, seeds=seeds, library=(library,),
                           budget=adopt.BUDGET, fit_budget=fit_budget, fit_seeds=FIT_SEEDS,
                           heldout=tasks.heldout, out=out, model_id=model["model"],
                           execution_mode="model", study_manifest=manifest)
        receipt["units"][unit] = dict(
            episodes=len(report.rows), seconds=time.time() - started,
            terminal_sha256=sha(str(out) + ".terminal.json"), overflows=backend.overflows,
            first_prompts_forwarded=check_first_prompts(manifest, methods, library,
                                                        backend.forwarded[first:]))
        write(receipt_path, receipt)
    receipt.update(status="complete", finished=time.time())
    write(receipt_path, receipt)
    print(json.dumps(dict(status="complete", units=list(receipt["units"]))))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--part", choices=sorted(PARTS), required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dev", action="store_true")
    run(parser.parse_args())
