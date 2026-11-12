"""Bounded production qualification of learning-stage accounting on public DEV tasks."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


class CountedBackend:
    """Observe the actual bounded provider boundary, avoiding nested-meter double counts."""
    def __init__(self, backend):
        self.backend, self.requests, self.responses = backend, 0, []

    def __call__(self, messages):
        raise RuntimeError("bounded generation required")

    def describe(self):
        return self.backend.describe()

    def generate_bounded(self, messages, *, max_new_tokens, seed):
        self.requests += 1
        response = self.backend.generate_bounded(messages, max_new_tokens=max_new_tokens, seed=seed)
        self.responses.append(dict(response["usage"]))
        return response


def verify_fit_usage(usage, observed, requests, budget):
    if len(observed) != requests or usage["calls"] != requests:
        raise ValueError("fit call meter differs from the actual backend boundary")
    if usage["generated_tokens"] != sum(r["completion_tokens"] for r in observed):
        raise ValueError("fit token meter differs from actual backend usage")
    if any(usage[k] > getattr(budget, k) for k in ("episodes", "calls", "generated_tokens")):
        raise ValueError("fit exceeded a declared budget")


def assert_qualified_runtime(source, entrypoint, modules=None):
    """Bind the entry point and loaded research modules to the CPU-tested snapshot."""
    source = source.resolve()
    manifest = json.loads((source / "source-manifest.json").read_text())
    paths = {"entrypoint": Path(entrypoint).resolve()}
    for name, module in (sys.modules if modules is None else modules).items():
        if name == "fin_skills" or name.startswith(("fin_skills.", "benchmarks.agent_study.")):
            path = getattr(module, "__file__", None)
            if path is not None:
                paths[name] = Path(path).resolve()
    for name, path in paths.items():
        if not path.is_relative_to(source):
            raise ValueError("executed module is outside qualified source: " + name)
        relative = path.relative_to(source).as_posix()
        if manifest.get(relative) != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("executed file is absent from or differs from CPU manifest: " + name)


def accounting_from_events(path):
    """Index durable provider-boundary events, including interrupted calls after a kill."""
    if not path.exists():
        return dict(available=False, reason="provider event ledger not created",
                    started_requests=None, known_generated_tokens=None, unknown_cost_requests=None)
    raw = path.read_bytes()
    rows = raw.splitlines(keepends=True)
    events, incomplete = [], False
    for index, row in enumerate(rows):
        if index == len(rows)-1 and not row.endswith(b"\n"):
            incomplete = True
            break
        events.append(json.loads(row))
    started = finished = known = tokens = 0
    current_fit = None
    for event in events:
        if event["kind"] == "fit_started":
            current_fit = {k: event[k] for k in ("method", "library", "seed")}
        if event.get("backend_mode") != "bounded":
            continue
        if event["kind"] == "call_started":
            started += 1
        elif event["kind"] in ("call_completed", "call_failed"):
            finished += 1
            cost = event.get("tokens") if event["kind"] == "call_completed" else event.get("token_cost")
            if cost is not None:
                known += 1
                tokens += cost
    if finished > started or known > started:
        raise ValueError("provider event ledger has more outcomes than requests")
    return dict(available=True, events_sha256=hashlib.sha256(raw).hexdigest(),
                started_requests=started, known_generated_tokens=tokens,
                unknown_cost_requests=started-known, unfinished_requests=started-finished,
                incomplete_final_event=incomplete, current_fit=current_fit,
                boundary="backend_mode=bounded; nested meter events excluded")


def worker(args):
    from fin_skills import rsi
    from fin_skills.tasks import frozen_suite
    from fin_skills.rsi.backend import describe_backend
    from fin_skills.rsi.meter import EVENT_SINK, BACKEND_BINDING, Budget, FitBudget
    from fin_skills.rsi.run_record import RunRecord, EXECUTION_BUNDLE, source_hashes, digest
    from benchmarks.agent_study.rsi_backend import BoundedTransformersChat
    from benchmarks.agent_study.rsi_qualify import verify_source, full_context_probe
    from benchmarks.agent_study.transformers_chat import TransformersChat
    source = args.qualified / "source"
    assert_qualified_runtime(source, __file__)
    source_sha = verify_source(source)
    cpu = json.loads(args.cpu_receipt.read_text())
    if cpu["status"] != "passed" or cpu["phase"] != "cpu" or cpu["source_manifest_sha256"] != source_sha:
        raise ValueError("new learning source requires its own complete CPU qualification")
    inventory = json.loads(args.inventory.read_text())
    row = next(item for item in inventory if item["id"] == "phi4-mini" and item.get("cached"))
    if (row["model"] != "microsoft/Phi-4-mini-instruct"
            or row["revision"] != "cfbefacb99257ffa30c83adab238a50856ac3083"):
        raise ValueError("pinned existing Phi-4 mini required")
    if any(os.environ.get(k) != "1" for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")):
        raise ValueError("offline cache required")
    dev = tuple(frozen_suite().dev)
    budget = FitBudget(episodes=6, proposals=2, calls=18, generated_tokens=20000)
    episode_budget = Budget(attempts=3, calls=6, generated_tokens=12000)
    plan = dict(model=row["model"], revision=row["revision"], seed=11, source_manifest_sha256=source_sha,
        cpu_receipt_sha256=hashlib.sha256(args.cpu_receipt.read_bytes()).hexdigest(),
        dev_tasks=[t.manifest() for t in dev], fit_budget=budget.as_dict(),
        episode_budget=episode_budget.as_dict(), per_call_output_cap=2048,
        fits=[dict(method=cls.name, library=lib) for cls in (rsi.PlaybookLearning, rsi.Evolve)
              for lib in (False, True)],
        scope="public development fit qualification; no test tasks, no scientific comparison",
        coverage="accounting, budget termination and state freezing; does not qualify accepted candidate updates",
        publication_ready=False)
    write(args.output / "fit-plan.json", plan)
    os.environ["HF_HUB_CACHE"] = row["cache_check"]["cache"]
    backend = CountedBackend(BoundedTransformersChat(TransformersChat(row["model"], row["revision"], max_tokens=2048, seed=11)))
    write(args.output / "loaded.json", dict(backend=backend.describe(), context=full_context_probe(backend.backend.chat)))
    fit_protocol = dict(plan=[], dev_tasks=plan["dev_tasks"], test_tasks=[], source_hashes=source_hashes(),
                        backend=describe_backend(backend, strict=True), qualification_plan=plan)
    record = RunRecord(args.output / "fit-events-owner.jsonl", fit_protocol)
    sink = EVENT_SINK.set(record.event)
    binding = BACKEND_BINDING.set((backend, describe_backend(backend, strict=True)))
    bundle = EXECUTION_BUNDLE.set(record.bundle)
    results = []
    current_fit = None
    try:
        for cls in (rsi.PlaybookLearning, rsi.Evolve):
            for library in (False, True):
                calls, responses = backend.requests, len(backend.responses)
                method = cls(library=library)
                current_fit = dict(method=method.name, library=library, seed=11)
                record.event(dict(kind="fit_started", method=method.name, library=library, seed=11))
                method.fit(backend, dev, budget=budget, seeds=(11,), episode_budget=episode_budget)
                verify_fit_usage(method.fit_usage, backend.responses[responses:], backend.requests-calls, budget)
                if not method.fitted or [t.manifest() for t in dev] != plan["dev_tasks"]:
                    raise ValueError("fit did not freeze state or mutated development tasks")
                state = method.state()
                if digest(state) != digest(method.state()):
                    raise ValueError("reading fitted state is not stable")
                result = dict(method=method.name, library=library, usage=method.fit_usage,
                    state_sha256=digest(state), state=state,
                    updates=getattr(method, "history", getattr(method, "decisions", [])))
                results.append(result)
                write(args.output / f"fit-{method.name}-{int(library)}.json", result)
                record.state("running", fits=results)
        if verify_source(source) != source_sha or source_hashes() != fit_protocol["source_hashes"]:
            raise ValueError("source changed during learning qualification")
        assert_qualified_runtime(source, __file__)
        record.state("complete", fits=results)
        write(args.output / "worker-terminal.json", dict(status="passed", fits=[{k:v for k,v in item.items()
            if k not in ("state", "updates")} for item in results], source_manifest_sha256=source_sha,
            actual_backend_requests=backend.requests, known_generated_tokens=sum(r["completion_tokens"] for r in backend.responses),
            unknown_cost_requests=backend.requests-len(backend.responses), publication_ready=False))
    except BaseException as exc:
        record.state("invalid", fits=results, error=f"{type(exc).__name__}: {exc}")
        write(args.output / "worker-terminal.json", dict(status="invalid", current_fit=current_fit,
            error=f"{type(exc).__name__}: {exc}", completed_fits=len(results),
            source_manifest_sha256=source_sha, actual_backend_requests=backend.requests,
            known_generated_tokens=sum(r["completion_tokens"] for r in backend.responses),
            unknown_cost_requests=backend.requests-len(backend.responses), publication_ready=False))
        raise
    finally:
        EVENT_SINK.reset(sink)
        BACKEND_BINDING.reset(binding)
        EXECUTION_BUNDLE.reset(bundle)


def main(args):
    # Reuse the CPU-qualified external supervisor without changing the numerical source.
    supervisor_receipt = args.supervisor / "result-3607880/terminal.json"
    receipt = json.loads(supervisor_receipt.read_text())
    helper = args.supervisor / "run_frozen_study.py"
    helper_sha = hashlib.sha256(helper.read_bytes()).hexdigest()
    if (receipt["status"] != "passed" or not receipt["source_binding"]
            or receipt["source_files"][helper.name] != helper_sha
            or helper_sha != "89db675b77ffcd87444b781bb69a715088984f297902b989c807dc617c377f64"):
        raise ValueError("the production-qualified supervisor is required")
    sys.path.insert(0, str(args.supervisor))
    import run_frozen_study
    if Path(run_frozen_study.__file__).resolve() != helper.resolve():
        raise ValueError("actual supervisor import differs from its qualified file")
    release = dict(process_seconds=4800,
                   resources=dict(cpu=2, mem_gib=32, gpu=1, walltime_seconds=5400))
    snapshot = subprocess.run(["scontrol", "show", "job", os.environ["SLURM_JOB_ID"], "-o"],
                              capture_output=True, text=True, timeout=15, check=True).stdout
    allocation = run_frozen_study.validate_allocation(release, snapshot)
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "supervisor-plan.json", dict(slurm_job_id=os.environ["SLURM_JOB_ID"],
        owner_chat="01a1bc9a-3383-7b41-800d-b1cb5bfefff9", process_seconds=4800,
        allocation=allocation, supervisor_sha256=helper_sha,
        supervisor_receipt_sha256=hashlib.sha256(supervisor_receipt.read_bytes()).hexdigest(),
        scope="bounded learning qualification, four fits, not full S-seven",
        started_utc=datetime.now(timezone.utc).isoformat(), publication_ready=False))
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker"]
    for name in ("qualified", "cpu_receipt", "inventory", "output", "supervisor"):
        command.extend(("--"+name.replace("_", "-"), str(getattr(args, name).resolve())))
    started = time.monotonic()
    supervision = run_frozen_study.supervised_command(command, seconds=4800, output=args.output)
    terminal = args.output / "worker-terminal.json"
    result = json.loads(terminal.read_text()) if terminal.exists() else dict(status="invalid")
    if supervision["timed_out"] or supervision["returncode"]:
        result["status"] = "invalid"
    result["provider_event_accounting"] = accounting_from_events(
        args.output / "fit-events-owner.jsonl.events.jsonl")
    result.update(supervision=supervision, seconds=time.monotonic()-started, publication_ready=False)
    write(args.output / "terminal.json", result)
    if result["status"] != "passed":
        raise RuntimeError("learning qualification failed; preserve partial fit records")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("qualified", "cpu-receipt", "inventory", "output", "supervisor"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("learning qualification requires Slurm")
    (worker if args.worker else main)(args)
