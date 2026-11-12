"""P-adopt runner (exploratory): one model, one part, inside a Slurm GPU allocation.

Design: docs/P_ADOPTION_STUDY_DESIGN_20261112.md. Parts are disjoint:

  check        Check arms L0 (no library) and L1, L2, L3 (library), 72 episodes
  playbook_L0  PlaybookLearning on L0 prompts: fit on the 12 dev tasks, then 18 episodes
  playbook_L1  PlaybookLearning on L1 prompts: fit on the 12 dev tasks, then 18 episodes
  all          the three parts above in one allocation (used by --dev)

Each unit writes under <output>/<model_id>/<unit>/job-<SLURM_JOB_ID>/. A unit that already has a
complete terminal receipt in an earlier job directory is skipped, so a preempted job can be
submitted again without repeating finished units; an interrupted directory is kept as it is.

--dev runs the same code on development tasks only (one test task, one seed, a small fit
budget) and checks the pipeline: every unit completes, the first prompt of every Check
episode reached the model unchanged, and the backend runs with the declared output cap. It
reads no grades, so it cannot be used to tune the design.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fin_skills.rsi import adopt  # noqa: E402
from fin_skills.rsi.meter import Budget, FitBudget  # noqa: E402
from fin_skills.rsi.run_record import digest  # noqa: E402

UNITS = {"check_L0": ((adopt.CheckL0,), False),
         "check_lib": ((adopt.CheckL1, adopt.CheckL2, adopt.CheckL3), True),
         "playbook_L0": ((adopt.PlaybookL0,), False),
         "playbook_L1": ((adopt.PlaybookL1,), True)}
PARTS = {"check": ("check_L0", "check_lib"), "playbook_L0": ("playbook_L0",),
         "playbook_L1": ("playbook_L1",), "all": tuple(UNITS)}
SEEDS = (101, 211, 307)
FIT_SEEDS = (11,)
DEV_TEST = ("macd_a",)
DEV_FIT = ("inverse_volatility_b", "e4_ma_crossover", "momentum_a", "e4_vol_target_momentum")
DEV_FIT_BUDGET = FitBudget(episodes=8, proposals=1, calls=60, generated_tokens=120_000)


def write(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(body, sort_keys=True, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def design(dev: bool):
    """Tasks, seeds and fit budget for a mode; the main mode is the declared design."""
    from fin_skills.tasks import suite
    s = suite()
    if dev:
        by_id = {t.id: t for t in s.dev}
        return (SimpleNamespace(dev=tuple(by_id[k] for k in DEV_FIT),
                                heldout=tuple(by_id[k] for k in DEV_TEST)),
                (SEEDS[0],), DEV_FIT_BUDGET)
    return SimpleNamespace(dev=tuple(s.dev), heldout=tuple(s.heldout)), SEEDS, FitBudget()


def complete(unit_dir):
    """Earlier job directories of this unit whose run finished completely."""
    done = []
    for terminal in sorted(unit_dir.glob("job-*/*.jsonl.terminal.json")):
        if json.loads(terminal.read_text(encoding="utf-8")).get("status") == "complete":
            done.append(str(terminal))
    return done


def load_model(model_id, protocol_path, inventory_path, output_cap):
    from benchmarks.agent_study.adopt_backend import ContextAsBudget
    from benchmarks.agent_study.rsi_backend import BoundedTransformersChat
    from benchmarks.agent_study.transformers_chat import TransformersChat
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    model = next(m for m in protocol["models"]["twenty_hf_models"] if m["id"] == model_id)
    inventory = json.loads(Path(inventory_path).read_text(encoding="utf-8"))
    cached = [m for m in inventory if m["id"] == model_id and m.get("cached")]
    if len(cached) != 1 or any(cached[0][k] != model[k] for k in ("model", "revision")):
        raise ValueError("complete cached weights at the frozen revision required")
    if any(os.environ.get(k) != "1" for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")):
        raise ValueError("offline model loading required")
    os.environ["HF_HUB_CACHE"] = cached[0]["cache_check"]["cache"]
    chat = TransformersChat(model["model"], model["revision"], max_tokens=output_cap, seed=11)
    backend = ContextAsBudget(BoundedTransformersChat(chat))
    config = backend.describe()
    if (config["max_new_tokens"], config["temperature"], config["top_p"]) != (output_cap, .1, 1.):
        raise ValueError("backend generation settings differ from the design")
    return model, backend, config


def run(args):
    from fin_skills.rsi import benchmark
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        raise RuntimeError("P-adopt execution requires a Slurm allocation")
    tasks, seeds, fit_budget = design(args.dev)
    manifest = adopt.manifest(tasks.heldout, seeds=seeds)
    manifest.update(mode="dev-pipeline-check" if args.dev else "main",
                    fit_budget=vars(fit_budget), fit_seeds=list(FIT_SEEDS))
    root = Path(args.output) / ("dev" if args.dev else "main") / args.model_id
    restart = int(os.environ.get("SLURM_RESTART_COUNT", "0") or 0)
    run_id = f"job-{job}" + (f"-r{restart}" if restart else "")
    receipt_path = root / f"{run_id}.json"
    receipt = dict(job=job, run_id=run_id, model_id=args.model_id, part=args.part, dev=args.dev,
                   manifest_sha256=digest(manifest), runner_sha256=sha(__file__),
                   adopt_sha256=sha(adopt.__file__), started=time.time(), units={})
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
            fits=[dict(method=f["method"], usage=f["usage"]) for f in report.fits],
            first_prompts_forwarded=check_first_prompts(manifest, methods, library,
                                                        backend.forwarded[first:]))
        write(receipt_path, receipt)
    receipt.update(status="complete", finished=time.time())
    write(receipt_path, receipt)
    print(json.dumps(dict(status="complete", units=list(receipt["units"]))))
    return receipt


def check_first_prompts(manifest, methods, library, forwarded):
    """For Check arms: every planned first prompt reached the model exactly as declared."""
    names = {m.name for m in methods if not m.learns}
    expected = [p["messages_sha256"] for p in manifest["packets"]
                if p["method"] in names and p["library"] is library]
    seen = set(forwarded)
    missing = [h for h in expected if h not in seen]
    if missing:
        raise ValueError(f"{len(missing)} declared first prompts never reached the model")
    return len(expected)


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
