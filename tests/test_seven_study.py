import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fin_skills.rsi.run_record import digest
from fin_skills.tasks import suite
from scripts.seven_study import build_units, complete_summary, partial_accounting, phase_costs, validate_unit


def protocol():
    tasks = suite()
    names = ["no_improve", "self_debug", "check_feedback", "reflexion", "playbook", "team", "evolve"]
    models = [dict(id=f"model-{i}", model=f"provider/{i}", revision="revision") for i in range(5)]
    validation = set(sorted({t.algorithm for t in tasks.dev})[1::2])
    return dict(models=dict(core_ids=[m["id"] for m in models], twenty_hf_models=models),
        positions=dict(dev_tasks=[t.manifest() for t in tasks.dev],
                       evaluation_tasks=[t.manifest() for t in tasks.heldout]),
        seven_methods=dict(names=names, learning_methods=["playbook", "evolve"],
            models=[m["id"] for m in models], library_conditions=[False, True],
            evaluation_seeds=[101, 211, 307], fit_seed=11,
            frozen_method_parameters=dict(Evolve=dict(actual_fit_batch_ids=
                [t.id for t in sorted(tasks.dev, key=lambda t:t.id)[:4]]),
                PlaybookLearning=dict(development_ids=[t.id for t in tasks.dev if t.algorithm not in validation],
                                      validation_ids=[t.id for t in tasks.dev if t.algorithm in validation]))))


def test_plan_keeps_all_tasks_seeds_in_one_fit_and_complete_population():
    units = build_units(protocol(), suite())
    assert len(units) == 70
    assert sum(u["learned"] for u in units) == 20
    assert sum(len(u["cells"]) for u in units) == 1260
    assert all(len(u["cells"]) == 18 and len({c["task_id"] for c in u["cells"]}) == 6 for u in units)
    changed = protocol()
    changed["positions"]["dev_tasks"][0]["definition"] = "altered"
    with pytest.raises(ValueError, match="task manifests"):
        build_units(changed, suite())
    changed = protocol()
    changed["seven_methods"]["frozen_method_parameters"]["Evolve"]["actual_fit_batch_ids"].reverse()
    with pytest.raises(ValueError, match="minibatch"):
        build_units(changed, suite())


def events():
    return [dict(kind="fit_started", method="playbook", library=True),
        dict(kind="call_started", backend_mode="nested"),
        dict(kind="call_started", backend_mode="bounded"),
        dict(kind="response_received", response=dict(usage=dict(prompt_tokens=100, completion_tokens=7))),
        dict(kind="call_completed", backend_mode="bounded", tokens=7),
        dict(kind="response_received", response=dict(usage=dict(prompt_tokens=100, completion_tokens=7))),
        dict(kind="call_completed", backend_mode="nested", tokens=7),
        dict(kind="episode_started", method="playbook", library=True, task_id="t", seed=101),
        dict(kind="call_started", backend_mode="bounded"),
        dict(kind="response_received", response=dict(usage=dict(prompt_tokens=20, completion_tokens=2))),
        dict(kind="call_failed", backend_mode="bounded", token_cost=2)]


def write_events(tmp_path, rows):
    path = tmp_path / "events.jsonl"
    path.write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")
    return path


def test_nested_fit_costs_count_actual_provider_once_and_known_refusal():
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as directory:
        costs = phase_costs(write_events(Path(directory), events()))
    assert costs[("fit", "playbook", True)] == dict(calls=1, generated_tokens=7, prompt_tokens=100)
    assert costs[("test", "playbook", True, "t", 101)] == dict(calls=1, generated_tokens=2, prompt_tokens=20)


@pytest.mark.parametrize("change", ["unknown", "unfinished", "double_response", "phase_switch"])
def test_broken_provider_ledger_cannot_produce_zero_costs(tmp_path, change):
    rows = events()
    if change == "unknown":
        rows[-1]["token_cost"] = None
    elif change == "unfinished":
        rows.pop()
    elif change == "double_response":
        rows.insert(4, copy.deepcopy(rows[3]))
    else:
        rows.insert(3, dict(kind="fit_started", method="team", library=False))
    with pytest.raises(ValueError):
        phase_costs(write_events(tmp_path, rows))


def test_unit_requires_one_fit_exact_state_and_all_cells():
    unit = next(u for u in build_units(protocol(), suite()) if u["method"] == "playbook")
    rows = [dict(c, model_id=c["model"], state_sha256=digest({}), usage=dict(calls=0, generated_tokens=0))
            for c in unit["cells"]]
    report = SimpleNamespace(rows=rows, _validate=lambda:None)
    fit = dict(method="playbook", library=False, model_id=rows[0]["model_id"], state={},
               state_sha256=digest({}), usage=dict(calls=1, generated_tokens=7))
    costs = {("fit", "playbook", False):dict(calls=1, generated_tokens=7, prompt_tokens=10)}
    result = validate_unit(report, [fit], unit, costs)
    assert result["amortized_fit_cost_per_test_episode"]["generated_tokens"] == 7/18
    with pytest.raises(ValueError, match="exactly one fit"):
        validate_unit(report, [fit, fit], unit, costs)
    rows[-1]["state_sha256"] = digest({"leak":"test update"})
    with pytest.raises(ValueError, match="fitted state"):
        validate_unit(report, [fit], unit, costs)
    rows.pop()
    with pytest.raises(ValueError, match="every frozen"):
        validate_unit(report, [fit], unit, costs)


def test_whole_population_summary_refuses_missing_cells_and_retains_all_contrasts():
    units = build_units(protocol(), suite())
    rows = [dict(c, model_id=c["model"], valid=False, status="model_refusal") for u in units for c in u["cells"]]
    receipts = [dict(unit_id=u["id"]) for u in units]
    result = complete_summary(rows, units, receipts)
    assert result["episodes"] == 1260 and result["learned_fits"] == 20
    assert len(result["groups"]) == 14 and len(result["contrasts"]) == 13
    assert all(c["effect"] == 0 and c["algorithm_blocks"] == 3 for c in result["contrasts"])
    with pytest.raises(ValueError, match="complete seven-method"):
        complete_summary(rows[:-1], units, receipts)
    with pytest.raises(ValueError, match="one-fit-per-unit"):
        complete_summary(rows, units, receipts[:-1])


def test_partial_costs_preserve_unknown_and_absent_usage(tmp_path):
    absent = partial_accounting(tmp_path / "missing")
    assert absent["unknown_cost_requests"] is None
    rows = events()[:-2]
    path = write_events(tmp_path, rows)
    with path.open("ab") as stream:
        stream.write(b'{"incomplete"')
    result = partial_accounting(path)
    assert result["started_requests"] == 2 and result["known_generated_tokens"] == 7
    assert result["unknown_cost_requests"] == result["unfinished_requests"] == 1
    assert result["incomplete_final_event"]


def test_release_keeps_complete_units_and_refuses_duplicate_or_cross_model(tmp_path):
    from scripts.run_seven_study import selected_units, sha
    units = build_units(protocol(), suite())
    (tmp_path / "plan.json").write_text("{}")
    (tmp_path / "units.json").write_text(json.dumps(units))
    release = dict(status="released", plan_sha256=sha(tmp_path / "plan.json"), unit_ids=[units[0]["id"]])
    assert len(selected_units(tmp_path, release)[0]["cells"]) == 18
    release["unit_ids"] *= 2
    with pytest.raises(ValueError, match="unique complete"):
        selected_units(tmp_path, release)
    release["unit_ids"] = [units[0]["id"], units[14]["id"]]
    with pytest.raises(ValueError, match="one model"):
        selected_units(tmp_path, release)


def test_original_model_qualification_cannot_be_relabelled_as_new_runtime(tmp_path):
    from scripts.run_seven_study import qualification_for
    p = protocol()
    receipt = dict(status="passed", phase="model", model_id="model-0", source_manifest_sha256="v2",
        details=dict(backend=dict(model_id="provider/0", model_revision="revision")))
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps(receipt))
    composition = dict(backend_source_manifest_sha256="v2")
    assert qualification_for("model-0", p, composition, path)["model_id"] == "provider/0"
    receipt["source_manifest_sha256"] = "v4"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="original model"):
        qualification_for("model-0", p, composition, path)


def test_runner_calls_benchmark_once_with_original_fit_budget_and_all_test_cells(tmp_path, monkeypatch):
    from fin_skills import rsi
    from scripts import run_seven_study as runner
    tasks = suite()
    unit = next(u for u in build_units(protocol(), tasks) if u["method"] == "playbook")
    captured = []
    def benchmark(methods, backend, passed_tasks, **kwargs):
        captured.append((methods, passed_tasks, kwargs))
        return SimpleNamespace(fits=[])
    monkeypatch.setattr(rsi, "benchmark", benchmark)
    monkeypatch.setattr(runner.seven, "phase_costs", lambda path:{})
    monkeypatch.setattr(runner.seven, "validate_unit", lambda *args:{})
    output = tmp_path / "unit.jsonl"
    Path(str(output)+".terminal.json").write_text("{}")
    spec = dict(evaluation_seeds=[101,211,307], fit_seed=11,
        test_budget=dict(attempts=3, calls=6, generated_tokens=12000),
        fit_budget_per_learned_method_model_library=dict(episodes=48, proposals=6, calls=600, generated_tokens=1200000))
    runner.run_unit(object(), tasks, unit, spec, output, "provider/0")
    assert len(captured) == 1
    methods, passed_tasks, kwargs = captured[0]
    assert methods == [rsi.PlaybookLearning] and len(passed_tasks.heldout) == 6
    assert len(passed_tasks.dev) == 12 and kwargs["seeds"] == (101,211,307)
    assert kwargs["library"] == (False,) and kwargs["fit_seeds"] == (11,)
    assert kwargs["fit_budget"].calls == 600 and kwargs["fit_budget"].episodes == 48


@pytest.mark.parametrize("broken", ["worker", "marker"])
def test_interrupted_json_still_writes_parent_invalid_terminal_and_unknown_costs(tmp_path, monkeypatch, broken):
    from scripts import run_seven_study as runner
    plan = tmp_path / "plan"
    plan.mkdir()
    for name in ("plan.json", "protocol.json", "composition.json"):
        (plan / name).write_text("{}")
    release = tmp_path / "release.json"
    release.write_text(json.dumps(dict(process_seconds=10)))
    qualification, inventory = tmp_path / "qualification.json", tmp_path / "inventory.json"
    qualification.write_text("{}"); inventory.write_text("[]")
    unit = dict(id="S-model-playbook-0", model_id="model", cells=[])
    monkeypatch.setattr(runner, "verify_plan", lambda _:dict(source_manifest_sha256="v4"))
    monkeypatch.setattr(runner, "selected_units", lambda *args:[unit])
    monkeypatch.setattr(runner, "qualification_for", lambda *args:{})
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs:SimpleNamespace(stdout=""))
    monkeypatch.setattr(runner.common, "validate_allocation", lambda *args:{})
    monkeypatch.setenv("SLURM_JOB_ID", "synthetic")
    def interrupt(command, *, seconds, output):
        (output / "worker-terminal.json").write_text('{"status":' if broken == "worker" else
            json.dumps(dict(status="complete_released_units", completed=[])))
        if broken == "marker":
            (output / (unit["id"]+".accepted.json")).write_text('{"unit_id":')
        (output / (unit["id"]+".jsonl.events.jsonl")).write_text(
            json.dumps(dict(kind="fit_started", method="playbook", library=False, data_seed=11))+"\n"+
            json.dumps(dict(kind="call_started", backend_mode="bounded"))+"\n")
        return dict(timed_out=True, returncode=-15)
    monkeypatch.setattr(runner.common, "supervised_command", interrupt)
    args = SimpleNamespace(plan=plan, release=release, qualification=qualification,
                           inventory=inventory, output=tmp_path / "result")
    with pytest.raises(RuntimeError, match="S allocation invalid"):
        runner.run(args)
    terminal = json.loads((args.output / "terminal.json").read_text())
    assert terminal["status"] == "invalid" and terminal["completed"] == []
    assert terminal["read_errors"][0]["stage"] == ("worker_terminal" if broken == "worker" else "accepted_unit")
    assert terminal["provider_event_accounting"][unit["id"]]["unknown_cost_requests"] == 1


def test_existing_partial_claim_stops_before_model_load_or_fit(tmp_path, monkeypatch):
    from scripts import run_seven_study as runner
    from benchmarks.agent_study import transformers_chat
    plan, output = tmp_path / "plan", tmp_path / "output"
    plan.mkdir(); output.mkdir()
    (plan / "claims").mkdir()
    unit = dict(id="S-model-playbook-0", model_id="model", cells=[])
    claim = plan / "claims" / (unit["id"]+".json")
    claim.write_text('{"interrupted":')
    protocol = dict(models=dict(twenty_hf_models=[dict(id="model", model="provider/model", revision="rev")]))
    (plan / "protocol.json").write_text(json.dumps(protocol))
    (plan / "composition.json").write_text("{}")
    release, qualification, inventory = (tmp_path / name for name in ("release.json", "qualification.json", "inventory.json"))
    release.write_text("{}"); qualification.write_text("{}")
    inventory.write_text(json.dumps([dict(id="model", model="provider/model", revision="rev", cached=True)]))
    (output / "allocation.json").write_text(json.dumps(dict(release_sha256=runner.sha(release),
        qualification_sha256=runner.sha(qualification), inventory_sha256=runner.sha(inventory))))
    monkeypatch.setattr(runner, "verify_plan", lambda _:dict(source_manifest_sha256="v4"))
    monkeypatch.setattr(runner, "selected_units", lambda *args:[unit])
    monkeypatch.setattr(runner, "qualification_for", lambda *args:{})
    monkeypatch.setenv("SLURM_JOB_ID", "synthetic")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1"); monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    loaded = []
    monkeypatch.setattr(transformers_chat, "TransformersChat", lambda *args, **kwargs:loaded.append(True))
    with pytest.raises(FileExistsError):
        runner.worker(SimpleNamespace(plan=plan, output=output, release=release,
                                      qualification=qualification, inventory=inventory))
    assert not loaded and claim.read_text() == '{"interrupted":'
    assert json.loads((output / "worker-terminal.json").read_text())["status"] == "invalid"
