"""Prepare and execute immutable, explicitly released chunks of the P/E study in Slurm.

The numerical runtime is the separately qualified immutable snapshot. This coordinator
does not change that runtime, retry existing outputs, fit methods, or infer new scope.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

SENSITIVITY_AMENDMENT_SHA256 = "c6c5cd4a7eb79fe3a9317e77d5281edb34918d3b5a9373e7e2c2ee54f7f45a1f"
CONTEXT_AMENDMENT_SHA256 = "42cf5910f2479b0caa3e88c4251d2c90e604311ebf69a935805b8f316ecc2170"
EXECUTION_AMENDMENT_SHA256 = "2ba7083155fb6c9f9ab852d00a4e19d74835e528c01b9b7ac1f5bd1003ca841b"


def check_execution_amendment(path, protocol_sha):
    if (sha(path) != EXECUTION_AMENDMENT_SHA256
            or read(path)['base_protocol_sha256'] != protocol_sha):
        raise ValueError('the authorized prospective execution-order amendment is required')


def check_context_amendment(path, protocol_sha):
    if sha(path) != CONTEXT_AMENDMENT_SHA256 or read(path)['base_protocol_sha256'] != protocol_sha:
        raise ValueError('the authorized prospective context amendment is required')


def context_module():
    try:
        import study_context
    except ModuleNotFoundError as exc:
        if exc.name != 'study_context':
            raise
        from scripts import study_context
    return study_context


def profile_builder():
    try:
        import prepare_context_profiles
    except ModuleNotFoundError as exc:
        if exc.name != 'prepare_context_profiles':
            raise
        from scripts import prepare_context_profiles
    return prepare_context_profiles


def allocation_context(root, plan, protocol, release, model_id):
    if 'context_amendment_sha256' not in plan:
        return None  # Old exact32K plans remain readable; new verify_plan requires the amendment.
    if plan['context_amendment_sha256'] != CONTEXT_AMENDMENT_SHA256:
        raise ValueError('context amendment differs from frozen plan')
    model = next(m for m in protocol['models']['twenty_hf_models'] if m['id']==model_id)
    cells = [c for c in read(root/'cells.json') if c['model_id']==model_id]
    return context_module().validate_profile(release.get('context_profile'), model_id=model_id,
        model=model, protocol_sha=plan['protocol_source_sha256'],
        enrollment_sha=plan['enrollment_terminal_sha256'], amendment_sha=CONTEXT_AMENDMENT_SHA256,
        source_sha=plan['source_manifest_sha256'], planned_cells=cells,
        helper_hashes={p.name:sha(p) for p in (Path(context_module().__file__),Path(profile_builder().__file__))})


def check_analysis_amendment(path, protocol_sha):
    if sha(path) != SENSITIVITY_AMENDMENT_SHA256:
        raise ValueError("the prespecified sensitivity addendum is required")
    if read(path)["base_protocol_sha256"] != protocol_sha:
        raise ValueError("sensitivity addendum belongs to a different protocol")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, body):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(body, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def helpers():
    try:
        import frozen_plan
        return frozen_plan
    except ModuleNotFoundError as exc:
        if exc.name != "frozen_plan":
            raise
        from benchmarks.agent_study import frozen_plan
        return frozen_plan


def assert_runtime_root(qualified):
    """Bind the modules actually executing to the qualified snapshot, not a nearby tree."""
    import importlib
    source = (qualified / "source").resolve()
    required = ("fin_skills", "fin_skills.tasks.execution", "fin_skills.rsi.run_record",
                "fin_skills.rsi.study", "benchmarks.agent_study.rsi_backend",
                "benchmarks.agent_study.external_study", "benchmarks.agent_study.rsi_qualify",
                "benchmarks.agent_study.transformers_chat")
    for name in required:
        importlib.import_module(name)
    for name, module in tuple(sys.modules.items()):
        if name == "fin_skills" or name.startswith(("fin_skills.", "benchmarks.agent_study.")):
            path = getattr(module, "__file__", None)
            if path is not None and not Path(path).resolve().is_relative_to(source):
                raise ValueError("executed module is outside qualified source: " + name)
    from fin_skills.rsi.run_record import source_hashes
    manifest = read(source / "source-manifest.json")
    if any(manifest.get(name) != value for name, value in source_hashes().items()):
        raise ValueError("executed runtime source hashes differ from qualified manifest")


def load_tasks(enrollment):
    from benchmarks.agent_study.external_study import ScalarTask
    terminal = read(enrollment / "terminal.json")
    if terminal["status"] != "enrollment_frozen":
        raise ValueError("complete frozen external enrollment required")
    for name, expected in terminal["files"].items():
        if Path(name).name != name or sha(enrollment / name) != expected:
            raise ValueError("enrollment file identity changed")
    public = [json.loads(line) for line in (enrollment / "public.jsonl").read_text().splitlines()]
    scoring = [json.loads(line) for line in (enrollment / "scoring.jsonl").read_text().splitlines()]
    if len(public) != len(scoring) or len(public) != terminal["selected_questions"]:
        raise ValueError("enrollment packet counts differ")
    tasks = [ScalarTask(p, s) for p, s in zip(public, scoring)]
    if len({t.id for t in tasks}) != len(tasks):
        raise ValueError("duplicate external task identities")
    return tasks


def checked_model_qualification(path, protocol, source_sha, model_id, context_profile=None):
    expected = {row["id"]: row for row in protocol["models"]["twenty_hf_models"]}
    receipt = read(path)
    if model_id not in expected or receipt.get("model_id") != model_id:
        raise ValueError("undeclared or mismatched qualified model")
    if (receipt.get("status") != "passed" or receipt.get("phase") != "model"
            or receipt.get("source_manifest_sha256") != source_sha):
        raise ValueError("matching successful production qualification required")
    config = receipt["details"]["backend"]
    if (config["model_id"] != expected[model_id]["model"]
            or config["model_revision"] != expected[model_id]["revision"]):
        raise ValueError("qualified model identity differs from protocol")
    generation = protocol["models"]["common_generation"]
    declared = dict(precision="torch.bfloat16", context_tokens=context_module().expected_context(protocol,context_profile),
                    max_new_tokens=generation["per_call_output_cap"],
                    temperature=generation["temperature"], top_p=generation["top_p"])
    if any(config.get(key) != value for key, value in declared.items()):
        raise ValueError("qualified backend differs from the frozen generation/context contract")
    return dict(path=str(Path(path).resolve()), sha256=sha(path), backend=config)


def checked_qualifications(paths, protocol, source_sha):
    expected = {row["id"]: row for row in protocol["models"]["twenty_hf_models"]}
    result = {}
    for path in paths:
        receipt = read(path)
        model_id = receipt["model_id"]
        if model_id in result or model_id not in expected:
            raise ValueError("duplicate or undeclared qualified model")
        result[model_id] = checked_model_qualification(path, protocol, source_sha, model_id)
    if not result:
        raise ValueError("at least one successful declared-model qualification is required")
    return result


def allocation_qualification(plan, protocol, release, model_id, context_profile=None):
    """Admit any predeclared model without mutating the population or earlier receipts."""
    frozen = plan["qualifications"].get(model_id)
    if frozen is not None:
        if "qualification" in release:
            raise ValueError("cannot override a qualification frozen in the study plan")
        ref = frozen
    else:
        ref = release.get("qualification")
        if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
            raise ValueError("model requires an explicit path/hash qualification release")
    if sha(ref["path"]) != ref["sha256"]:
        raise ValueError("qualification receipt changed")
    checked = checked_model_qualification(ref["path"], protocol,
                                           plan["source_manifest_sha256"], model_id, context_profile)
    if frozen is not None and checked != frozen:
        raise ValueError("qualification differs from frozen plan")
    return checked


def selected_chunks(root, release, protocol):
    if release["plan_sha256"] != sha(root / "plan.json") or release["status"] != "released":
        raise ValueError("matching concrete release required")
    requested = release["chunk_ids"]
    by_id = {c["id"]: c for c in read(root / "chunks.json")}
    if not requested or len(requested) != len(set(requested)) or set(requested)-set(by_id):
        raise ValueError("release requires unique planned chunks")
    selected = [by_id[key] for key in requested]
    model_ids = {c["model_id"] for c in selected}
    if len(model_ids) != 1:
        raise ValueError("one qualified model per allocation required")
    model_id = next(iter(model_ids))
    expected = next((m for m in protocol["models"]["twenty_hf_models"] if m["id"] == model_id), None)
    stage = "core" if model_id in protocol["models"]["core_ids"] else "extension"
    if expected is None or any(not chunk["cells"] or any(
            cell["stage"] != stage or cell["model_id"] != model_id
            or cell["model"] != expected["model"] or cell["model_revision"] != expected["revision"]
            for cell in chunk["cells"]) for chunk in selected):
        raise ValueError("chunk model/stage differs from fixed population")
    return selected


def accounting_module():
    try:
        import seven_study
    except ModuleNotFoundError as exc:
        if exc.name != "seven_study":
            raise
        from scripts import seven_study
    return seven_study


def report_module():
    try:
        import summarize_rsi_qualification
    except ModuleNotFoundError as exc:
        if exc.name != "summarize_rsi_qualification":
            raise
        from scripts import summarize_rsi_qualification
    return summarize_rsi_qualification


def coordinator_files():
    return (Path(__file__), Path(helpers().__file__), Path(accounting_module().__file__),
            Path(report_module().__file__), Path(context_module().__file__),Path(profile_builder().__file__))


def accepted_chunk(root, chunk, qualification, context_profile=None):
    """A marker is only an index: verify its complete report, runtime and provider ledger."""
    from fin_skills.rsi.run_record import source_hashes
    path = root / (chunk["id"] + ".jsonl")
    marker = read(root / (chunk["id"] + ".accepted.json"))
    expected = dict(chunk_id=chunk["id"], episodes=len(chunk["cells"]),
                    terminal_sha256=sha(str(path) + ".terminal.json"))
    if marker != expected:
        raise ValueError("accepted chunk marker differs from complete terminal")
    report = report_module().checked_report(path, len(chunk["cells"]))
    helpers().validate_report(report, chunk)
    if report.protocol["source_hashes"] != source_hashes():
        raise ValueError("accepted chunk runtime differs from qualified source")
    config = report.protocol["backend"]["configuration"]
    if {k:v for k,v in config.items() if k != "hardware"} != {
            k:v for k,v in qualification["backend"].items() if k != "hardware"}:
        raise ValueError("accepted chunk backend differs from admitted qualification")
    cost = accounting_module().partial_accounting(Path(str(path) + ".events.jsonl"))
    if (not cost["available"] or cost["unknown_cost_requests"] != 0
            or cost["unfinished_requests"] != 0 or cost["incomplete_final_event"]
            or cost["started_requests"] != sum(row["usage"]["calls"] for row in report.rows)
            or cost["known_generated_tokens"] != sum(row["usage"]["generated_tokens"] for row in report.rows)):
        raise ValueError("accepted chunk provider ledger is incomplete or differs")
    if context_profile is not None:
        guard = context_module().guard_accounting(Path(str(path)+'.events.jsonl'),
            effective_tokens=context_profile['native']['effective_tokens'])
        if (report.protocol['backend']['adapter_sha256'] != sha(context_module().__file__)
                or not guard['available'] or guard['checked_requests'] != cost['started_requests']
                or guard['proven_zero_generation_requests'] or guard['missing_checks']
                or guard['unfinished_request'] or guard['incomplete_final_event']):
            raise ValueError('accepted chunk lacks complete exact context-guard evidence')
    return report, marker


def allocation_terminal(root, selected, supervision, qualification, context_profile=None):
    errors = []
    try:
        result = read(root / "worker-terminal.json")
        if not isinstance(result, dict) or result.get("status") not in ("invalid", "complete_released_chunks"):
            raise ValueError("worker terminal has no recognized status")
    except (ValueError, OSError) as exc:
        errors.append(dict(stage="worker_terminal", error=f"{type(exc).__name__}: {exc}"))
        result = dict(status="invalid")
    if supervision["timed_out"] or supervision["returncode"] != 0:
        result["status"] = "invalid"
    result["completed"], result["provider_event_accounting"] = [], {}
    for chunk in selected:
        if (root / (chunk["id"] + ".accepted.json")).exists():
            try:
                _, marker = accepted_chunk(root, chunk, qualification, context_profile)
                result["completed"].append(marker)
            except Exception as exc:
                result["status"] = "invalid"
                errors.append(dict(stage="accepted_chunk", chunk_id=chunk["id"],
                                   error=f"{type(exc).__name__}: {exc}"))
        try:
            cost = accounting_module().partial_accounting(root / (chunk["id"] + ".jsonl.events.jsonl"))
        except (ValueError, OSError) as exc:
            result["status"] = "invalid"
            cost = dict(available=False, unknown_cost_requests=None, error=f"{type(exc).__name__}: {exc}")
        result["provider_event_accounting"][chunk["id"]] = cost
        if context_profile is not None:
            try:
                guard = context_module().guard_accounting(root/(chunk['id']+'.jsonl.events.jsonl'),
                    effective_tokens=context_profile['native']['effective_tokens'])
                zero = guard['proven_zero_generation_requests']
                unknown = cost.get('unknown_cost_requests')
                guard['remaining_unknown_generation_requests'] = (
                    unknown-zero if unknown is not None and zero is not None and unknown>=zero else None)
            except (ValueError, OSError, KeyError, TypeError) as exc:
                result['status']='invalid'
                guard=dict(available=False,error=f'{type(exc).__name__}: {exc}')
            result.setdefault('context_guard_accounting',{})[chunk['id']]=guard
    if len(result["completed"]) != len(selected):
        result["status"] = "invalid"
    result.update(supervision=supervision, read_errors=errors, publication_ready=False)
    return result


def prepare(args):
    assert_runtime_root(args.qualified)
    from fin_skills.tasks import suite
    from fin_skills.rsi import study
    from fin_skills.rsi.meter import Budget
    from benchmarks.agent_study.rsi_qualify import verify_source
    protocol = read(args.protocol)
    check_analysis_amendment(args.analysis_amendment, sha(args.protocol))
    check_context_amendment(args.context_amendment, sha(args.protocol))
    check_execution_amendment(args.execution_amendment, sha(args.protocol))
    source_sha = verify_source(args.qualified / "source")
    qualifications = checked_qualifications(args.qualification, protocol, source_sha)
    external = load_tasks(args.enrollment)
    enrollment = read(args.enrollment / "terminal.json")
    if (enrollment["qualified_source_manifest_sha256"] != source_sha
            or enrollment["protocol_sha256"] != sha(args.protocol)):
        raise ValueError("enrollment and study protocol/runtime differ")
    positions = suite().heldout
    if sorted((t.manifest() for t in positions), key=lambda t:t["id"]) != sorted(
            protocol["positions"]["evaluation_tasks"], key=lambda t:t["id"]):
        raise ValueError("position tasks differ from the frozen public suite")
    seeds = protocol["positions"]["data_and_generation_seed_ids"]
    budget = Budget(**{k:protocol["positions"]["budget"][k]
                       for k in ("attempts", "calls", "generated_tokens")})
    position_plan = study.manifest(positions, seeds=seeds, budget=budget)
    external_by_id = {t.id:t for t in external}
    cells = read(args.enrollment / "episode-plan.json")
    for cell in cells:
        cell["task_sha256"] = external_by_id[cell["task_id"]].sha256
    core = set(protocol["models"]["core_ids"])
    for model in protocol["models"]["twenty_hf_models"]:
        for packet in position_plan["packets"]:
            cells.append(dict(track="P", stage="core" if model["id"] in core else "extension",
                model_id=model["id"], model=model["model"], model_revision=model["revision"],
                **{key:packet[key] for key in ("method", "library", "task_id", "task_sha256", "seed",
                                               "messages_sha256")}))
    partition = helpers().chunks(cells, tasks_per_chunk=args.tasks_per_chunk)
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "protocol.json", protocol)
    with (args.output / "analysis-amendment.json").open("xb") as stream:
        stream.write(args.analysis_amendment.read_bytes())
    with (args.output / 'context-amendment.json').open('xb') as stream:
        stream.write(args.context_amendment.read_bytes())
    with (args.output / 'execution-amendment.json').open('xb') as stream:
        stream.write(args.execution_amendment.read_bytes())
    write(args.output / "position-prompts.json", position_plan)
    write(args.output / "cells.json", cells)
    write(args.output / "chunks.json", partition)
    plan = dict(status="prepared_not_released", created_utc=datetime.now(timezone.utc).isoformat(),
        slurm_job_id=os.environ["SLURM_JOB_ID"], source_manifest_sha256=source_sha,
        qualified_root=str(args.qualified.resolve()), enrollment_root=str(args.enrollment.resolve()),
        enrollment_terminal_sha256=sha(args.enrollment / "terminal.json"),
        protocol_source_sha256=sha(args.protocol), qualifications=qualifications,
        context_amendment_sha256=sha(args.context_amendment),
        execution_amendment_sha256=sha(args.execution_amendment),
        helpers={path.name:sha(path) for path in coordinator_files()},
        files={name:sha(args.output / name) for name in
               ("protocol.json", "position-prompts.json", "cells.json", "chunks.json",
                "analysis-amendment.json", 'context-amendment.json', 'execution-amendment.json')},
        cell_counts={track:dict(Counter(c["stage"] for c in cells if c["track"] == track))
                     for track in ("P", "E")},
        chunk_count=len(partition), tasks_per_chunk=args.tasks_per_chunk,
        model_calls=0, publication_ready=False,
        release_requirement="separate frozen allocation resource/time budget for exact chunk IDs")
    write(args.output / "plan.json", plan)
    print(json.dumps({k:plan[k] for k in ("status", "cell_counts", "chunk_count", "model_calls")}))


def verify_plan(root):
    plan = read(root / "plan.json")
    check_analysis_amendment(root / "analysis-amendment.json", plan["protocol_source_sha256"])
    check_context_amendment(root/'context-amendment.json', plan['protocol_source_sha256'])
    check_execution_amendment(root/'execution-amendment.json', plan['protocol_source_sha256'])
    if plan['execution_amendment_sha256'] != EXECUTION_AMENDMENT_SHA256:
        raise ValueError('plan execution-order amendment binding differs')
    if plan['context_amendment_sha256'] != CONTEXT_AMENDMENT_SHA256:
        raise ValueError('plan context amendment binding differs')
    assert_runtime_root(Path(plan["qualified_root"]))
    from benchmarks.agent_study.rsi_qualify import verify_source
    for name, expected in plan["files"].items():
        if Path(name).name != name or sha(root / name) != expected:
            raise ValueError("frozen plan file changed")
    if any(sha(path) != plan["helpers"][path.name]
           for path in coordinator_files()):
        raise ValueError("coordinator changed after plan freeze")
    if verify_source(Path(plan["qualified_root"]) / "source") != plan["source_manifest_sha256"]:
        raise ValueError("qualified runtime changed")
    if sha(Path(plan["enrollment_root"]) / "terminal.json") != plan["enrollment_terminal_sha256"]:
        raise ValueError("enrollment receipt changed")
    return plan


def supervised_command(command, *, seconds, output, grace_seconds=2):
    """Bound a newly owned process group even if Python cleanup ignores an exception."""
    if os.name != "posix":
        raise RuntimeError("formal supervision requires the qualified Linux host")
    started = time.monotonic()
    with (output / "worker.stdout").open("xb") as stdout, (output / "worker.stderr").open("xb") as stderr:
        proc = subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True)
        timed_out = False
        try:
            returncode = proc.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                returncode = proc.wait(timeout=grace_seconds)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                returncode = proc.wait(timeout=grace_seconds)
        finally:
            # A surviving owned descendant must not keep running after its parent exits.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    return dict(returncode=returncode, timed_out=timed_out, seconds=time.monotonic()-started,
                process_group=proc.pid, deadline_seconds=seconds, cleanup_grace_seconds=grace_seconds)


def validate_allocation(release, slurm_text):
    resources = release["resources"]
    fields = dict(part.split("=", 1) for part in slurm_text.split() if "=" in part)
    duration = fields["TimeLimit"]
    days, _, clock = duration.rpartition("-")
    parts = [int(value) for value in clock.split(":")]
    if len(parts) != 3:
        raise ValueError("explicit Slurm hours:minutes:seconds walltime required")
    wall = int(days or 0)*86400 + parts[0]*3600 + parts[1]*60 + parts[2]
    tres = dict(part.split("=", 1) for part in fields["ReqTRES"].split(","))
    expected_mem = {str(resources["mem_gib"])+"G", str(resources["mem_gib"]*1024)+"M"}
    if (int(tres["cpu"]) != resources["cpu"] or tres["mem"] not in expected_mem
            or int(tres["gres/gpu"]) != resources["gpu"] or wall != resources["walltime_seconds"]
            or fields["Requeue"] != "0"):
        raise ValueError("Slurm allocation request differs from the concrete release")
    if not 0 < release["process_seconds"] <= wall-60:
        raise ValueError("process limit must leave at least 60 seconds to close records")
    return dict(job_id=fields["JobId"], requested_tres=tres,
                allocated_tres=fields.get("AllocTRES"), walltime_seconds=wall,
                gpu_hours_cap=resources["gpu"]*wall/3600)


def run(args):
    plan = verify_plan(args.plan)
    release = read(args.release)
    protocol = read(args.plan / "protocol.json")
    selected = selected_chunks(args.plan, release, protocol)
    context_profile = allocation_context(args.plan,plan,protocol,release,selected[0]['model_id'])
    qualification = allocation_qualification(plan, protocol, release, selected[0]["model_id"],context_profile)
    seconds = release["process_seconds"]
    if type(seconds) is not int or seconds <= 0:
        raise ValueError("positive bounded process time required")
    snapshot = subprocess.run(["scontrol", "show", "job", os.environ["SLURM_JOB_ID"], "-o"],
                              capture_output=True, text=True, timeout=15, check=True).stdout
    allocation = validate_allocation(release, snapshot)
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "plan.json", dict(release=release, release_sha256=sha(args.release),
        inventory_sha256=sha(args.inventory), qualification=qualification,context_profile=context_profile,
        slurm_job_id=os.environ["SLURM_JOB_ID"],
        created_utc=datetime.now(timezone.utc).isoformat(), allocation=allocation,
        supervisor="owned-process-group-wallclock"))
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "run", "--worker"]
    for name in ("plan", "release", "inventory", "output"):
        command.extend(("--"+name, str(getattr(args, name).resolve())))
    try:
        supervision = supervised_command(command, seconds=seconds, output=args.output)
    except BaseException as exc:
        supervision = dict(timed_out=False, returncode=None,
                           supervisor_error=f"{type(exc).__name__}: {exc}")
    result = allocation_terminal(args.output, selected, supervision, qualification,context_profile)
    write(args.output / "terminal.json", result)
    if result["status"] != "complete_released_chunks":
        raise RuntimeError(result.get("error", "worker did not complete released chunks"))
    print(json.dumps(result, sort_keys=True))


def run_worker(args):
    from types import SimpleNamespace
    from fin_skills.tasks import suite
    from fin_skills.rsi import study
    from fin_skills.rsi.meter import Budget
    from benchmarks.agent_study.rsi_backend import BoundedTransformersChat
    from benchmarks.agent_study.transformers_chat import TransformersChat
    from benchmarks.agent_study.rsi_qualify import full_context_probe
    from benchmarks.agent_study.external_study import run as external_run, interface_text
    plan = verify_plan(args.plan)
    release, protocol = read(args.release), read(args.plan / "protocol.json")
    selected = selected_chunks(args.plan, release, protocol)
    model_id = selected[0]["model_id"]
    context_profile = allocation_context(args.plan,plan,protocol,release,model_id)
    qualification = allocation_qualification(plan, protocol, release, model_id,context_profile)
    expected_model = next(m for m in protocol["models"]["twenty_hf_models"] if m["id"] == model_id)
    inventory = read(args.inventory)
    matching = [m for m in inventory if m["id"] == model_id and m.get("cached")]
    if len(matching) != 1 or any(matching[0][key] != expected_model[key] for key in ("model", "revision")):
        raise ValueError("matching complete cached weights required")
    if any(os.environ.get(k) != "1" for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")):
        raise ValueError("offline model loading required")
    if type(release["process_seconds"]) is not int or release["process_seconds"] <= 0:
        raise ValueError("positive bounded process time required")
    # Exclusive claims precede loading. An interrupted claim requires an explicit
    # recovery decision; another allocation cannot silently regenerate those cells.
    claims = args.plan / "claims"
    claims.mkdir(exist_ok=True)
    supervisor = read(args.output / "plan.json")
    if (supervisor["release_sha256"] != sha(args.release)
            or supervisor["inventory_sha256"] != sha(args.inventory)
            or supervisor["qualification"] != qualification
            or supervisor.get('context_profile') != context_profile
            or supervisor.get("supervisor") != "owned-process-group-wallclock"):
        raise ValueError("supervisor receipt required")
    completed = []
    started = time.monotonic()
    try:
        for chunk in selected:
            write(claims / (chunk["id"] + ".json"), dict(slurm_job_id=os.environ["SLURM_JOB_ID"],
                release_sha256=sha(args.release), output=str(args.output.resolve())))
        os.environ["HF_HUB_CACHE"] = matching[0]["cache_check"]["cache"]
        chat = TransformersChat(expected_model["model"], expected_model["revision"], max_tokens=2048, seed=11)
        backend = BoundedTransformersChat(chat)
        config = backend.describe()
        if {k:v for k,v in config.items() if k != "hardware"} != {
                k:v for k,v in qualification["backend"].items() if k != "hardware"}:
            raise ValueError("runtime/backend configuration differs from qualified model")
        if context_profile is not None:
            backend = context_module().ContextGuardedBackend(backend,context_profile)
        write(args.output / "loaded.json", dict(backend=config, context=full_context_probe(chat),
            context_profile_sha256=release.get('context_profile',{}).get('sha256'),
            context_guard_sha256=sha(context_module().__file__) if context_profile is not None else None))
        external = {t.id:t for t in load_tasks(Path(plan["enrollment_root"]))}
        frozen_interface = read(Path(plan["enrollment_root"]) / "interface.json")
        if interface_text() != frozen_interface["text"]:
            raise ValueError("library interface differs from enrollment prompts")
        positions = {t.id:t for t in suite().heldout}
        for chunk in selected:
            if time.monotonic() - started >= release["process_seconds"]:
                raise TimeoutError("released process budget exhausted before next chunk")
            output = args.output / (chunk["id"] + ".jsonl")
            task_map = positions if chunk["track"] == "P" else external
            if any(task_map[cell["task_id"]].sha256 != cell["task_sha256"] for cell in chunk["cells"]):
                raise ValueError("task changed before generation")
            if chunk["track"] == "P":
                tasks = tuple(positions[key] for key in chunk["task_ids"])
                budget = Budget(**{k:protocol["positions"]["budget"][k]
                                   for k in ("attempts", "calls", "generated_tokens")})
                report = study.run(backend, SimpleNamespace(dev=(), heldout=tasks), out=output,
                    seeds=protocol["positions"]["data_and_generation_seed_ids"], budget=budget,
                    model_id=expected_model["model"], execution_mode="model")
            else:
                report = external_run(backend, tuple(external[key] for key in chunk["task_ids"]),
                    seeds=protocol["external"]["experiment"]["seeds"], out=output,
                    model_id=expected_model["model"], execution_mode="model")
            helpers().validate_report(report, chunk)
            completed.append(dict(chunk_id=chunk["id"], episodes=len(report.rows),
                terminal_sha256=sha(str(output) + ".terminal.json")))
            write(args.output / (chunk["id"] + ".accepted.json"), completed[-1])
        verify_plan(args.plan)
        result = dict(status="complete_released_chunks", completed=completed,
                      seconds=time.monotonic()-started, publication_ready=False)
    except BaseException as exc:
        write(args.output / "worker-terminal.json", dict(status="invalid", completed=completed,
            error=f"{type(exc).__name__}: {exc}", seconds=time.monotonic()-started, publication_ready=False))
        raise
    write(args.output / "worker-terminal.json", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    prep = sub.add_parser("prepare")
    for name in ("qualified", "enrollment", "protocol", "output", "analysis-amendment",
                 'context-amendment', 'execution-amendment'):
        prep.add_argument("--" + name, type=Path, required=True)
    prep.add_argument("--qualification", type=Path, action="append", required=True)
    prep.add_argument("--tasks-per-chunk", type=int, default=25)
    exe = sub.add_parser("run")
    exe.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    for name in ("plan", "release", "inventory", "output"):
        exe.add_argument("--" + name, type=Path, required=True)
    arguments = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("study preparation and execution require Slurm")
    (prepare if arguments.phase == "prepare" else (run_worker if arguments.worker else run))(arguments)
