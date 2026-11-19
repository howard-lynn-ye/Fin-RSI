"""P-confirm runner: one model, inside a Slurm GPU allocation.

Design: docs/P_CONFIRM_STUDY_DESIGN_20261119.md. Stages:

  dev   development, on the six earlier held-out tasks (now development material) and their
        three seeds: arm C2 with "check" and with "cause" feedback (unit dev_hint, 36 episodes).
        Its grades are read; it chooses the feedback and tells whether the pre-freeze fixes work.
  main  the confirmatory run: the 24 tasks of fin_skills/tasks/confirm.py and their two seeds,
        arms C0-C4 with the feedback chosen from the development run (units c0, route, lib;
        240 episodes). It refuses to start until pconfirm.MAIN_FEEDBACK is set.

Everything else is P-adopt's runner (scripts/run_p_adopt.py): units write under
<output>/<stage>/<model_id>/<unit>/job-<SLURM_JOB_ID>/, a unit with a complete terminal receipt
is skipped on resubmission, the manifest must match earlier jobs of the model, and every
declared first prompt must reach the model. --pipeline runs one task and seed to check the
pipeline.
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

from fin_skills.rsi import adopt, pconfirm  # noqa: E402
from fin_skills.rsi.run_record import digest  # noqa: E402
from scripts.run_p_adopt import FIT_SEEDS, complete, design, load_model, sha, write  # noqa: E402

DEV_UNITS = {"dev_hint": ((pconfirm.C2, pconfirm.C2Cause), True)}


def main_units(feedback=None):
    """c0: no library; lib: interface, upper bound and retrieval; route: the routed arm."""
    feedback = feedback or pconfirm.MAIN_FEEDBACK
    if feedback not in pconfirm.FEEDBACK:
        raise RuntimeError("the confirmatory feedback is not chosen yet (pconfirm.MAIN_FEEDBACK)")
    x0, x1, x2, x3, x4 = pconfirm.METHODS[feedback]
    return {"c0": ((x0,), False), "route": ((x2,), True), "lib": ((x1, x3, x4), True)}


def stage_design(stage, pipeline):
    """(suite, test tasks, seeds, fit budget, units) for a stage."""
    _, seeds, fit_budget = design(False)
    if stage == "dev":
        tasks, units = design(False)[0], DEV_UNITS
    else:
        from fin_skills.tasks.confirm import CONFIRM_SEEDS, confirm_suite
        tasks, seeds, units = confirm_suite(), CONFIRM_SEEDS, main_units()
    heldout = list(tasks.heldout)
    if pipeline:
        heldout, seeds = heldout[:1], seeds[:1]
    return tasks, heldout, seeds, fit_budget, units


def check_first_prompts(manifest, methods, forwarded):
    names = {m.name for m in methods}
    expected = [p["messages_sha256"] for p in manifest["packets"] if p["method"] in names]
    missing = len(set(expected) - set(forwarded))
    if missing:
        raise ValueError(f"{missing} declared first prompts never reached the model")
    return len(expected)


def run(args):
    from fin_skills.rsi import benchmark
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        raise RuntimeError("P-confirm execution requires a Slurm allocation")
    tasks, heldout, seeds, fit_budget, units = stage_design(args.stage, args.pipeline)
    methods = tuple(m for unit in units.values() for m in unit[0])
    manifest = pconfirm.manifest(heldout, methods, seeds=seeds, budget=adopt.BUDGET,
                                 stage=args.stage)
    manifest.update(mode="pipeline-check" if args.pipeline else args.stage)
    root = (Path(args.output) / (args.stage + ("-pipeline" if args.pipeline else ""))
            / args.model_id)
    restart = int(os.environ.get("SLURM_RESTART_COUNT", "0") or 0)
    run_id = f"job-{job}" + (f"-r{restart}" if restart else "")
    receipt_path = root / f"{run_id}.json"
    receipt = dict(job=job, run_id=run_id, model_id=args.model_id, stage=args.stage,
                   pipeline=args.pipeline, cluster=os.environ.get("SLURM_CLUSTER_NAME"),
                   confinement=os.environ.get("FIN_TASKS_CONFINE") or "none",
                   manifest_sha256=digest(manifest), runner_sha256=sha(__file__),
                   pconfirm_sha256=sha(pconfirm.__file__), started=time.time(), units={})
    if not (root / "manifest.json").exists():
        write(root / "manifest.json", manifest)
    if json.loads((root / "manifest.json").read_text(encoding="utf-8")) != json.loads(
            json.dumps(manifest)):
        raise ValueError("manifest differs from the one this model's earlier jobs used")
    todo = [u for u in units if not complete(root / u)]
    receipt["skipped"] = [u for u in units if u not in todo]
    write(receipt_path, receipt)
    if not todo:
        print(json.dumps(dict(status="nothing_to_do", **receipt)))
        return receipt
    model, backend, config = load_model(args.model_id, args.protocol, args.inventory,
                                        adopt.OUTPUT_CAP)
    receipt.update(model=model, backend=config, loaded=time.time())
    write(receipt_path, receipt)
    for unit in todo:
        unit_methods, library = units[unit]
        out = root / unit / run_id / f"{unit}.jsonl"
        started, first = time.time(), len(backend.forwarded)
        report = benchmark(unit_methods, backend, tasks, seeds=seeds, library=(library,),
                           budget=adopt.BUDGET, fit_budget=fit_budget, fit_seeds=FIT_SEEDS,
                           heldout=heldout, out=out, model_id=model["model"],
                           execution_mode="model", study_manifest=manifest)
        receipt["units"][unit] = dict(
            episodes=len(report.rows), seconds=time.time() - started,
            terminal_sha256=sha(str(out) + ".terminal.json"), overflows=backend.overflows,
            first_prompts_forwarded=check_first_prompts(manifest, unit_methods,
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
    parser.add_argument("--stage", choices=("dev", "main"), required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pipeline", action="store_true")
    run(parser.parse_args())
