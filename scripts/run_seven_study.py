"""Freeze and run S-seven units in Slurm, with one fit per method/model/library."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

try:
    import run_frozen_study as common
    import seven_study as seven
except ModuleNotFoundError as exc:
    if exc.name not in ("run_frozen_study", "seven_study"):
        raise
    from scripts import run_frozen_study as common, seven_study as seven

read, write, sha = common.read, common.write, common.sha
PROTOCOL_SHA256 = "b430af714050ef1929f25affb51ace72fdcb423032cf9b4a461a2522d7ec121c"
BRIDGE_DECISION_SHA256 = "673b587c1821e14a3531eef50ae6014637b4e4e88ee4aaf151c7f54bcaa5f41c"
ANALYSIS_LEVELS_SHA256 = "5e898167f8596af2db178341e16ba9e1819a7f22f665277ac3e3291a932c55f7"
V3_SOURCE = "ca7a7fd230b196843daa335e08de657e3bd1f8a1173cdcdf2131ad472b8a22df"
V4_SOURCE = "6f4e7c4ccd36ad81e61ee8cb8996d04b520b51a47bf528301b003adb62d65785"
V2_SOURCE = "3b09f075a95a4d71102334e433a75463eac76fc822ca917b6d377152fc295d41"
BACKEND_FILES = ("fin_skills/rsi/backend.py", "fin_skills/rsi/meter.py",
    "benchmarks/agent_study/rsi_backend.py", "benchmarks/agent_study/transformers_chat.py",
    "benchmarks/agent_study/trading_schema_v8.py", "benchmarks/agent_study/trading_research_v9.py")


def checked_composition(path, qualified):
    """Require an explicit reviewed component bridge; never relabel old GPU receipts."""
    from benchmarks.agent_study.rsi_qualify import verify_source
    common.assert_runtime_root(qualified)
    composition = read(path)
    source_sha = verify_source(qualified / "source")
    if (composition["status"] != "reviewed_v4_component_acceptance"
            or composition["source_manifest_sha256"] != source_sha or source_sha != V4_SOURCE
            or composition["fit_source_manifest_sha256"] != V3_SOURCE
            or composition["backend_source_manifest_sha256"] != V2_SOURCE):
        raise ValueError("reviewed acceptance for this exact S runtime is required")
    evidence = composition["evidence"]
    if not {"cpu", "fit", "fit_plan", "fit_loaded", "fit_events", "delta_review", "composition_review"}.issubset(evidence):
        raise ValueError("CPU, fit, delta and composition review evidence are required")
    for item in evidence.values():
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("component evidence changed")
    if evidence["composition_review"]["sha256"] != BRIDGE_DECISION_SHA256:
        raise ValueError("only the exact independently reviewed v3/v4 bridge is permitted")
    decision = read(evidence["composition_review"]["path"])
    fit_root = Path(composition["fit_qualified_root"])
    if verify_source(fit_root / "source") != V3_SOURCE:
        raise ValueError("original fit source changed")
    before = read(fit_root / "source/source-manifest.json")
    after = read(qualified / "source/source-manifest.json")
    changed = sorted(name for name in set(before) | set(after) if before.get(name) != after.get(name))
    if changed != sorted(decision["changed_files"]):
        raise ValueError("runtime delta exceeds the reviewed bridge")
    for name, expected in decision["verified_unchanged_component_sha256"].items():
        if sha(fit_root / "source" / name) != expected or sha(qualified / "source" / name) != expected:
            raise ValueError("reviewed unchanged fit component differs: " + name)
    if sha(qualified / "source/fin_skills/rsi/methods.py") != decision["static_review"]["v4_methods_sha256"]:
        raise ValueError("methods differ from the exact reviewed delta")
    cpu = read(evidence["cpu"]["path"])
    if (cpu["status"] != "passed" or cpu["phase"] != "cpu"
            or cpu["source_manifest_sha256"] != source_sha):
        raise ValueError("S runtime needs its own successful production CPU receipt")
    counts = cpu["details"]["tests"]
    if (counts["tests"] != 146 or any(counts[k] for k in ("errors", "failures", "skipped"))
            or cpu["details"]["complete_planned_test_nodes"] != 146):
        raise ValueError("complete original v4 CPU plan required")
    fit = read(evidence["fit"]["path"])
    expected_fits = {(m, lib) for m in seven.LEARNERS for lib in (False, True)}
    if (fit["status"] != "passed" or fit["unknown_cost_requests"] != 0
            or len(fit["fits"]) != 4
            or {(f["method"], f["library"]) for f in fit["fits"]} != expected_fits
            or fit["source_manifest_sha256"] != composition["fit_source_manifest_sha256"]):
        raise ValueError("four complete bounded learning qualifications are required")
    fit_plan = read(evidence["fit_plan"]["path"])
    fit_loaded = read(evidence["fit_loaded"]["path"])
    if (fit_plan["source_manifest_sha256"] != V3_SOURCE
            or fit_plan["model"] != "microsoft/Phi-4-mini-instruct"
            or fit_loaded["backend"]["model_revision"] != fit_plan["revision"]
            or fit["provider_event_accounting"]["events_sha256"] != evidence["fit_events"]["sha256"]):
        raise ValueError("fit plan, model load and provider evidence differ")
    backend_root = Path(composition["backend_qualified_root"])
    if verify_source(backend_root / "source") != composition["backend_source_manifest_sha256"]:
        raise ValueError("inherited backend source changed")
    for name in BACKEND_FILES:
        if (sha(qualified / "source" / name) != sha(backend_root / "source" / name)
                or composition["inherited_backend_files"].get(name) != sha(qualified / "source" / name)):
            raise ValueError("inherited backend component is not identical: " + name)
    if not composition["coverage_limits"]:
        raise ValueError("component bridge must state its verification limits")
    return composition


def prepare(args):
    from fin_skills.tasks import suite
    if sha(args.protocol) != PROTOCOL_SHA256:
        raise ValueError("original frozen seven-method protocol required")
    if (sha(args.analysis_levels) != ANALYSIS_LEVELS_SHA256
            or read(args.analysis_levels)["base_protocol_sha256"] != PROTOCOL_SHA256):
        raise ValueError("prospective S comparison-level addendum required")
    composition = checked_composition(args.composition, args.qualified)
    coordinator = read(args.coordinator_receipt)
    if (coordinator["status"] != "passed" or not coordinator["source_binding"]
            or coordinator["source_manifest_sha256"] != V4_SOURCE
            or any(coordinator["source_files"].get(Path(p).name) != sha(p)
                   for p in (__file__, common.__file__, seven.__file__))):
        raise ValueError("new S coordinator requires its own actual-runtime CPU qualification")
    protocol = read(args.protocol)
    units = seven.build_units(protocol, suite())
    args.output.mkdir(parents=True, exist_ok=False)
    for name, source in (("protocol.json", args.protocol), ("composition.json", args.composition),
                         ("analysis-levels.json", args.analysis_levels)):
        with (args.output / name).open("xb") as stream:
            stream.write(source.read_bytes())
    write(args.output / "units.json", units)
    plan = dict(status="prepared_not_released", created_utc=datetime.now(timezone.utc).isoformat(),
        source_manifest_sha256=composition["source_manifest_sha256"],
        qualified_root=str(args.qualified.resolve()),
        coordinator_receipt=dict(path=str(args.coordinator_receipt.resolve()), sha256=sha(args.coordinator_receipt)),
        files={name:sha(args.output / name) for name in ("protocol.json", "composition.json", "units.json", "analysis-levels.json")},
        helpers={Path(p).name:sha(p) for p in (__file__, common.__file__, seven.__file__)},
        episodes=sum(len(u["cells"]) for u in units), learned_fits=sum(u["learned"] for u in units),
        units=len(units), model_calls=0, publication_ready=False,
        backend_gate="per-model exact original backend receipt plus runtime load/config/context verification",
        contrasts="six method-minus-no_improve comparisons averaged equally across two library conditions; seven library-minus-scratch comparisons; three algorithm blocks; descriptive only")
    write(args.output / "plan.json", plan)
    print(json.dumps(plan, sort_keys=True))


def verify_plan(root):
    plan = read(root / "plan.json")
    if sha(plan["coordinator_receipt"]["path"]) != plan["coordinator_receipt"]["sha256"]:
        raise ValueError("S coordinator qualification changed")
    for name, expected in plan["files"].items():
        if Path(name).name != name or sha(root / name) != expected:
            raise ValueError("frozen S plan changed")
    if sha(root / "protocol.json") != PROTOCOL_SHA256:
        raise ValueError("S protocol changed")
    if sha(root / "analysis-levels.json") != ANALYSIS_LEVELS_SHA256:
        raise ValueError("S comparison levels changed")
    for path in (__file__, common.__file__, seven.__file__):
        if plan["helpers"].get(Path(path).name) != sha(path):
            raise ValueError("S coordinator changed after freeze")
    checked_composition(root / "composition.json", Path(plan["qualified_root"]))
    from fin_skills.tasks import suite
    if seven.build_units(read(root / "protocol.json"), suite()) != read(root / "units.json"):
        raise ValueError("S units differ from original method/task scope")
    return plan


def selected_units(root, release):
    if release["status"] != "released" or release["plan_sha256"] != sha(root / "plan.json"):
        raise ValueError("matching explicit S release required")
    requested = release["unit_ids"]
    units = {u["id"]:u for u in read(root / "units.json")}
    if not requested or len(requested) != len(set(requested)) or set(requested)-set(units):
        raise ValueError("unique complete S execution units required")
    selected = [units[key] for key in requested]
    if len({u["model_id"] for u in selected}) != 1:
        raise ValueError("one model per S allocation")
    return selected


def qualification_for(model_id, protocol, composition, path):
    receipt = read(path)
    expected = next(m for m in protocol["models"]["twenty_hf_models"] if m["id"] == model_id)
    if (receipt["status"] != "passed" or receipt["phase"] != "model" or receipt["model_id"] != model_id
            or receipt["source_manifest_sha256"] != composition["backend_source_manifest_sha256"]):
        raise ValueError("matching original model qualification required")
    config = receipt["details"]["backend"]
    if config["model_id"] != expected["model"] or config["model_revision"] != expected["revision"]:
        raise ValueError("qualified model revision differs")
    return config


def run_unit(backend, tasks, unit, spec, output, model):
    from fin_skills import rsi
    from fin_skills.rsi.meter import Budget, FitBudget
    method = next(cls for cls in rsi.METHODS if cls.name == unit["method"])
    # Evolve defaults are scientific parameters, not tunable scheduler settings.
    if method.name == "evolve":
        check = method(library=unit["library"])
        frozen = spec["frozen_method_parameters"]["Evolve"]
        if check.generations != frozen["generations"] or check.minibatch != frozen["minibatch"]:
            raise ValueError("Evolve parameters differ from frozen protocol")
    report = rsi.benchmark([method], backend, tasks, library=(unit["library"],),
        seeds=tuple(spec["evaluation_seeds"]), fit_seeds=(spec["fit_seed"],),
        budget=Budget(**{k:spec["test_budget"][k] for k in ("attempts", "calls", "generated_tokens")}),
        fit_budget=FitBudget(**{k:spec["fit_budget_per_learned_method_model_library"][k]
                               for k in ("episodes", "proposals", "calls", "generated_tokens")}),
        out=output, model_id=model, execution_mode="model",
        study_manifest=dict(track="S-seven", unit_id=unit["id"], full_protocol_sha256=PROTOCOL_SHA256))
    costs = seven.phase_costs(Path(str(output)+".events.jsonl"))
    result = seven.validate_unit(report, report.fits, unit, costs)
    result["terminal_sha256"] = sha(str(output)+".terminal.json")
    return result


def accepted_unit(root, unit):
    """A marker alone is insufficient: verify its complete original report and costs."""
    try:
        from summarize_rsi_qualification import checked_report
    except ModuleNotFoundError as exc:
        if exc.name != "summarize_rsi_qualification":
            raise
        from scripts.summarize_rsi_qualification import checked_report
    from fin_skills.rsi.run_record import source_hashes
    marker = read(root / (unit["id"]+".accepted.json"))
    path = root / (unit["id"]+".jsonl")
    if marker["terminal_sha256"] != sha(str(path)+".terminal.json"):
        raise ValueError("accepted S terminal hash differs")
    report = checked_report(path, len(unit["cells"]))
    if (report.protocol["study_manifest"] != dict(track="S-seven", unit_id=unit["id"],
            full_protocol_sha256=PROTOCOL_SHA256) or report.protocol["source_hashes"] != source_hashes()):
        raise ValueError("accepted S runtime or study identity differs")
    fit = read(str(path)+".terminal.json")["fits"]
    result = seven.validate_unit(report, fit, unit, seven.phase_costs(Path(str(path)+".events.jsonl")))
    result["terminal_sha256"] = sha(str(path)+".terminal.json")
    if marker != result:
        raise ValueError("accepted S marker differs from verified complete unit")
    return report, result


def worker(args):
    from fin_skills.tasks import suite
    from benchmarks.agent_study.rsi_backend import BoundedTransformersChat
    from benchmarks.agent_study.transformers_chat import TransformersChat
    from benchmarks.agent_study.rsi_qualify import full_context_probe
    plan = verify_plan(args.plan)
    release = read(args.release)
    selected = selected_units(args.plan, release)
    protocol, composition = read(args.plan / "protocol.json"), read(args.plan / "composition.json")
    model_id = selected[0]["model_id"]
    config = qualification_for(model_id, protocol, composition, args.qualification)
    supervisor = read(args.output / "allocation.json")
    if (supervisor["release_sha256"] != sha(args.release)
            or supervisor["qualification_sha256"] != sha(args.qualification)
            or supervisor["inventory_sha256"] != sha(args.inventory)):
        raise ValueError("supervisor input identity differs")
    matching = [m for m in read(args.inventory) if m["id"] == model_id and m.get("cached")]
    expected = next(m for m in protocol["models"]["twenty_hf_models"] if m["id"] == model_id)
    if len(matching) != 1 or any(matching[0][k] != expected[k] for k in ("model", "revision")):
        raise ValueError("complete matching cached model required")
    if any(os.environ.get(k) != "1" for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")):
        raise ValueError("offline model loading required")
    claims = args.plan / "claims"
    claims.mkdir(exist_ok=True)
    completed = []
    started = time.monotonic()
    try:
        for unit in selected:
            write(claims / (unit["id"]+".json"), dict(slurm_job_id=os.environ["SLURM_JOB_ID"],
                output=str(args.output.resolve()), release_sha256=sha(args.release)))
        os.environ["HF_HUB_CACHE"] = matching[0]["cache_check"]["cache"]
        chat = TransformersChat(expected["model"], expected["revision"], max_tokens=2048, seed=11)
        backend = BoundedTransformersChat(chat)
        current = backend.describe()
        if {k:v for k,v in current.items() if k != "hardware"} != {k:v for k,v in config.items() if k != "hardware"}:
            raise ValueError("S backend differs from qualified components")
        write(args.output / "loaded.json", dict(backend=current, context=full_context_probe(chat),
            source_manifest_sha256=plan["source_manifest_sha256"],
            qualification_scope="inherited model components; actual S runtime load and context probe"))
        for unit in selected:
            if time.monotonic()-started >= release["process_seconds"]:
                raise TimeoutError("allocation budget exhausted before next complete S unit")
            output = args.output / (unit["id"]+".jsonl")
            result = run_unit(backend, suite(), unit, protocol["seven_methods"], output, expected["model"])
            write(args.output / (unit["id"]+".accepted.json"), result)
            completed.append(result)
        verify_plan(args.plan)
        write(args.output / "worker-terminal.json", dict(status="complete_released_units", completed=completed))
    except BaseException as exc:
        write(args.output / "worker-terminal.json", dict(status="invalid", completed=completed,
            error=f"{type(exc).__name__}: {exc}"))
        raise


def run(args):
    plan = verify_plan(args.plan)
    release = read(args.release)
    selected = selected_units(args.plan, release)
    qualification_for(selected[0]["model_id"], read(args.plan / "protocol.json"),
                      read(args.plan / "composition.json"), args.qualification)
    snapshot = subprocess.run(["scontrol", "show", "job", os.environ["SLURM_JOB_ID"], "-o"],
                              capture_output=True, text=True, timeout=15, check=True).stdout
    allocation = common.validate_allocation(release, snapshot)
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "allocation.json", dict(plan_sha256=sha(args.plan / "plan.json"),
        release=release, release_sha256=sha(args.release), qualification_sha256=sha(args.qualification),
        inventory_sha256=sha(args.inventory), allocation=allocation,
        slurm_job_id=os.environ["SLURM_JOB_ID"], source_manifest_sha256=plan["source_manifest_sha256"]))
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "run", "--worker"]
    for name in ("plan", "release", "inventory", "qualification", "output"):
        command.extend(("--"+name, str(getattr(args, name).resolve())))
    try:
        supervision = common.supervised_command(command, seconds=release["process_seconds"], output=args.output)
    except BaseException as exc:
        supervision = dict(timed_out=False, returncode=None,
                           supervisor_error=f"{type(exc).__name__}: {exc}")
    child = args.output / "worker-terminal.json"
    errors = []
    try:
        result = read(child)
        if not isinstance(result, dict) or result.get("status") not in ("invalid", "complete_released_units"):
            raise ValueError("worker terminal has no recognized status")
    except (ValueError, OSError) as exc:
        errors.append(dict(stage="worker_terminal", path=str(child), error=f"{type(exc).__name__}: {exc}"))
        result = dict(status="invalid", completed=[])
    if supervision["timed_out"] or supervision["returncode"] != 0:
        result["status"] = "invalid"
    # Recover only already accepted complete units; never infer missing rows or replay fit.
    result["completed"] = []
    for unit in selected:
        marker = args.output / (unit["id"]+".accepted.json")
        if not marker.exists():
            continue
        try:
            _, accepted = accepted_unit(args.output, unit)
            result["completed"].append(accepted)
        except Exception as exc:
            result["status"] = "invalid"
            errors.append(dict(stage="accepted_unit", path=str(marker), error=f"{type(exc).__name__}: {exc}"))
    result["provider_event_accounting"] = {}
    for unit in selected:
        try:
            cost = seven.partial_accounting(args.output / (unit["id"]+".jsonl.events.jsonl"))
        except (ValueError, OSError) as exc:
            result["status"] = "invalid"
            cost = dict(available=False, unknown_cost_requests=None,
                        error=f"{type(exc).__name__}: {exc}")
        result["provider_event_accounting"][unit["id"]] = cost
    if len(result["completed"]) != len(selected):
        result["status"] = "invalid"
    result.update(supervision=supervision, read_errors=errors, publication_ready=False)
    write(args.output / "terminal.json", result)
    if result["status"] != "complete_released_units" or len(result["completed"]) != len(selected):
        raise RuntimeError("S allocation invalid; retain claims, partial fit, rows and unknown costs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    prep = sub.add_parser("prepare")
    for name in ("qualified", "composition", "protocol", "analysis-levels", "coordinator-receipt", "output"):
        prep.add_argument("--"+name, type=Path, required=True)
    exe = sub.add_parser("run")
    for name in ("plan", "release", "inventory", "qualification", "output"):
        exe.add_argument("--"+name, type=Path, required=True)
    exe.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("S preparation and execution require Slurm")
    (prepare if args.phase == "prepare" else worker if args.worker else run)(args)
