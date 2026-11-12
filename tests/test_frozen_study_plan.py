from types import SimpleNamespace
import json

import pytest

from benchmarks.agent_study.frozen_plan import chunks, validate_report


def cells():
    return [dict(track="E", model_id=model, model="org/" + model,
                 method="external_code", library=lib, task_id=task, task_sha256=task + "-hash", seed=seed)
            for model in ("a", "b") for task in ("one", "two", "three")
            for lib in (False, True) for seed in (101, 211, 307)]


def test_partition_preserves_every_pair_and_every_model():
    partition = chunks(cells(), tasks_per_chunk=2)
    assert len(partition) == 4
    assert sum(len(chunk["cells"]) for chunk in partition) == 36
    assert sorted(len(chunk["cells"]) for chunk in partition) == [6, 6, 12, 12]
    assert partition == chunks(list(reversed(cells())), tasks_per_chunk=2)


def test_duplicates_and_wrong_identity_never_pass():
    with pytest.raises(ValueError, match="unique"):
        chunks(cells() + [cells()[0]], tasks_per_chunk=2)
    chunk = chunks(cells(), tasks_per_chunk=2)[0]
    rows = [dict(cell, model_id=cell["model"]) for cell in chunk["cells"]]
    report = SimpleNamespace(rows=rows, _validate=lambda: None)
    validate_report(report, chunk)
    report.rows[0]["task_sha256"] = "changed"
    with pytest.raises(ValueError, match="task or model"):
        validate_report(report, chunk)
    report.rows = rows[1:]
    with pytest.raises(ValueError, match="frozen study cells"):
        validate_report(report, chunk)


def test_per_model_release_preserves_fixed_population_and_earlier_qualifications(tmp_path):
    from scripts.run_frozen_study import checked_qualifications, allocation_qualification, sha
    models = [dict(id=f"m{i}", model=f"org/m{i}", revision=str(i) * 40) for i in range(5)]
    protocol = dict(models=dict(core_ids=[m["id"] for m in models], twenty_hf_models=models,
        common_generation=dict(context_tokens=32768, per_call_output_cap=2048, temperature=.1, top_p=1.0)))
    paths = []
    for model in models:
        path = tmp_path / (model["id"] + ".json")
        path.write_text(json.dumps(dict(model_id=model["id"], status="passed", phase="model",
            source_manifest_sha256="source", details=dict(backend=dict(
                model_id=model["model"], model_revision=model["revision"], precision="torch.bfloat16",
                context_tokens=32768, max_new_tokens=2048, temperature=.1, top_p=1.0)))), encoding="utf-8")
        paths.append(path)
    assert len(checked_qualifications(paths, protocol, "source")) == 5
    initial = checked_qualifications(paths[:-1], protocol, "source")
    assert len(initial) == 4
    plan = dict(qualifications=initial, source_manifest_sha256="source")
    original = json.dumps(dict(plan=plan, protocol=protocol), sort_keys=True)
    late = dict(qualification=dict(path=str(paths[-1]), sha256=sha(paths[-1])))
    assert allocation_qualification(plan, protocol, late, "m4")["sha256"] == sha(paths[-1])
    assert json.dumps(dict(plan=plan, protocol=protocol), sort_keys=True) == original
    with pytest.raises(ValueError, match="explicit path/hash"):
        allocation_qualification(plan, protocol, {}, "m4")
    with pytest.raises(ValueError, match="cannot override"):
        allocation_qualification(plan, protocol, late, "m0")
    with pytest.raises(ValueError, match="at least one"):
        checked_qualifications([], protocol, "source")
    with pytest.raises(ValueError, match="matching successful"):
        checked_qualifications(paths, protocol, "different-source")
    receipt = json.loads(paths[0].read_text())
    receipt["status"] = "failed"
    paths[0].write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="matching successful"):
        checked_qualifications(paths, protocol, "source")


def test_loading_enrollment_rejects_changed_data_before_generation(tmp_path):
    from benchmarks.agent_study.external_packets import ScalarPolicy, packets
    from scripts.run_frozen_study import load_tasks, sha
    row = dict(task="CodeFinQA", question="How many?", context="There are 3.",
               context_type="string", options=[], answer="3")
    public, scoring = packets(row, revision="a" * 40, split="test", row_index=0,
        source_cluster="unresolved", policy=ScalarPolicy("Use original units.", 0, 0, "synthetic"))
    for name, body in (("public.jsonl", public), ("scoring.jsonl", scoring)):
        (tmp_path / name).write_text(json.dumps(body) + "\n", encoding="utf-8")
    terminal = dict(status="enrollment_frozen", selected_questions=1,
        files={name:sha(tmp_path / name) for name in ("public.jsonl", "scoring.jsonl")})
    (tmp_path / "terminal.json").write_text(json.dumps(terminal))
    assert load_tasks(tmp_path)[0].id == public["id"]
    public["question"] = "An altered question"
    (tmp_path / "public.jsonl").write_text(json.dumps(public) + "\n")
    with pytest.raises(ValueError, match="file identity changed"):
        load_tasks(tmp_path)


def test_declared_resources_and_finishing_margin_are_enforced():
    from scripts.run_frozen_study import validate_allocation
    release = dict(process_seconds=3500, resources=dict(cpu=2, mem_gib=32, gpu=1, walltime_seconds=3600))
    snapshot = "JobId=123 TimeLimit=01:00:00 Requeue=0 ReqTRES=cpu=2,mem=32G,gres/gpu=1 AllocTRES=cpu=2,mem=32G,gres/gpu=1"
    assert validate_allocation(release, snapshot)["gpu_hours_cap"] == 1
    with pytest.raises(ValueError, match="differs"):
        validate_allocation(release, snapshot.replace("Requeue=0", "Requeue=1"))
    with pytest.raises(ValueError, match="at least 60"):
        validate_allocation(dict(release, process_seconds=3599), snapshot)


def test_unexecuted_tree_cannot_qualify_the_imported_runtime(tmp_path):
    from scripts.run_frozen_study import assert_runtime_root
    (tmp_path / "source").mkdir()
    with pytest.raises(ValueError, match="outside qualified source"):
        assert_runtime_root(tmp_path)


def test_analysis_addendum_is_bound_before_generation(tmp_path, monkeypatch):
    import scripts.run_frozen_study as runner
    path = tmp_path / "addendum.json"
    path.write_text(json.dumps(dict(base_protocol_sha256="frozen-protocol")))
    monkeypatch.setattr(runner, "SENSITIVITY_AMENDMENT_SHA256", runner.sha(path))
    runner.check_analysis_amendment(path, "frozen-protocol")
    with pytest.raises(ValueError, match="different protocol"):
        runner.check_analysis_amendment(path, "another-protocol")
    path.write_text(json.dumps(dict(base_protocol_sha256="changed")))
    with pytest.raises(ValueError, match="prespecified"):
        runner.check_analysis_amendment(path, "changed")


def test_execution_order_amendment_requires_exact_bytes_and_original_protocol(tmp_path, monkeypatch):
    import scripts.run_frozen_study as runner
    path = tmp_path / "execution-amendment.json"
    path.write_text(json.dumps(dict(base_protocol_sha256="frozen-protocol")))
    monkeypatch.setattr(runner, "EXECUTION_AMENDMENT_SHA256", runner.sha(path))
    runner.check_execution_amendment(path, "frozen-protocol")
    with pytest.raises(ValueError, match="execution-order"):
        runner.check_execution_amendment(path, "different-protocol")
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="execution-order"):
        runner.check_execution_amendment(path, "frozen-protocol")
