"""Cross-layer evaluation invariants, using trusted scripted programs and fake models."""
import json
from dataclasses import replace

import pytest

from fin_skills import rsi
from fin_skills.tasks import Judge, Task, suite
from fin_skills.tasks.judge import EvaluationInfrastructureError
from fin_skills.rsi.bench import Report


def test_task_identity_covers_instructions_library_options_and_origin():
    task = suite().dev[0]
    assert Task.from_manifest(task.manifest()) == task
    for changes in (dict(library_options={"window": 31}), dict(traits=frozenset()),
                    dict(family="other"), dict(origin="independent")):
        assert replace(task, **changes).sha256 != task.sha256
    with pytest.raises(ValueError, match="complete"):
        Task.from_manifest({"id": task.id})


def test_worker_executes_complete_custom_task_not_builtin_id():
    task = replace(suite().get("e4_ma_crossover"), id="custom-contract",
                   rows=120, factors="ratio", parameters={"fast": 8, "slow": 30})
    code = '''from fin_skills.algorithms import positions
def solve(prices, split_factors):
    assert len(prices) == 120
    return positions("ma_crossover", prices, split_factors, factors="ratio",
                     parameters={"fast": 8, "slow": 30})
'''
    judge = Judge(task, seed=991, library=True)
    assert judge.check(code).accepted
    assert judge.grade(code)


def test_infrastructure_failure_never_becomes_repair_feedback(monkeypatch):
    import fin_skills.tasks.judge as module
    monkeypatch.setattr(module, "_run_worker", lambda *a: dict(
        executed=False, infrastructure_error=True,
        error={"type": "WorkerCrash", "message": "dependency missing"}))
    calls = []
    def generate(messages):
        calls.append(messages)
        return "def solve(prices, split_factors): return prices"
    with pytest.raises(EvaluationInfrastructureError, match="dependency missing"):
        rsi.CheckFeedback(library=True).solve(generate, suite().heldout[0], seed=11)
    assert len(calls) == 1


def test_failed_backend_dispatch_is_charged_and_not_retried():
    def broken(messages):
        raise OSError("provider unavailable")
    meter = rsi.Meter(broken, rsi.Budget())
    with pytest.raises(OSError):
        meter([])
    assert meter.calls == 1 and meter.log[-1]["error"] == "OSError"


def test_over_budget_response_cannot_become_an_accepted_program():
    response = {"choices": [{"message": {"content": "do not use"}}],
                "usage": {"completion_tokens": 11}}
    meter = rsi.Meter(lambda _: response, rsi.Budget(generated_tokens=10))
    with pytest.raises(rsi.BudgetExceeded, match="generated_tokens"):
        meter([])
    assert meter.calls == 1 and meter.generated_tokens == 11


def test_nested_meter_preserves_estimated_token_provenance():
    outer = rsi.Meter(lambda _: "text", rsi.FitBudget())
    inner = rsi.Meter(outer.respond, rsi.Budget())
    inner([])
    assert outer.estimated_tokens and inner.estimated_tokens


def test_invalid_run_retains_plan_and_refuses_restart(tmp_path):
    out = tmp_path / "episodes.jsonl"
    def broken(messages):
        raise OSError("provider unavailable")
    args = dict(seeds=(11,), library=(True,), heldout=[suite().heldout[0]], out=out)
    with pytest.raises(OSError):
        rsi.benchmark([rsi.NoImprove], broken, suite(), **args)
    state = json.loads(out.with_name(out.name + ".state.json").read_text())
    assert state["status"] == "invalid" and state["expected_episodes"] == 1
    assert state["completed_episodes"] == 0
    events = [json.loads(line) for line in out.with_name(
        out.name + ".events.jsonl").read_text().splitlines()]
    assert any(e["kind"] == "call_started" and e["messages"] for e in events)
    assert any(e["kind"] == "call_failed" and not e["cost_known"] for e in events)
    terminal = json.loads(out.with_name(out.name + ".terminal.json").read_text())
    assert terminal["status"] == "invalid" and terminal["events_sha256"]
    with pytest.raises(FileExistsError):
        rsi.benchmark([rsi.NoImprove], broken, suite(), **args)


def test_renamed_development_algorithm_is_still_not_test_data():
    renamed = replace(suite().dev[0], id="new-name")
    with pytest.raises(ValueError, match="algorithms"):
        rsi.benchmark([rsi.NoImprove], lambda _: "unused", suite(), heldout=[renamed])


def test_incomplete_report_refuses_a_success_rate():
    with pytest.raises(ValueError, match="incomplete"):
        Report().summary()


def test_bounded_backend_receives_remaining_cap_and_stable_role_seeds():
    class Model:
        def __init__(self):
            self.calls = []
        def __call__(self, messages):
            raise AssertionError("unbounded dispatch")
        def generate_bounded(self, messages, *, max_new_tokens, seed):
            self.calls.append((max_new_tokens, seed))
            return {"choices": [{"message": {"content": "ok"}}],
                    "usage": {"completion_tokens": 4}}
    a, b = Model(), Model()
    for backend in (a, b):
        outer = rsi.Meter(backend, rsi.FitBudget(generated_tokens=10), seed=999)
        inner = rsi.Meter(outer.respond, rsi.Budget(generated_tokens=9), seed=11)
        inner([], role="writer")
        inner([], role="writer")
        assert inner.generated_tokens == outer.generated_tokens == 8
    assert a.calls == b.calls
    assert [cap for cap, _ in a.calls] == [9, 5]
    assert a.calls[0][1] != a.calls[1][1]


def test_nested_predispatch_failure_does_not_charge_previous_tokens():
    reply = {"choices": [{"message": {"content": "ok"}}],
             "usage": {"completion_tokens": 4}}
    outer = rsi.Meter(lambda _: reply, rsi.FitBudget(calls=1))
    inner = rsi.Meter(outer.respond, rsi.Budget())
    inner([])
    with pytest.raises(rsi.BudgetExceeded):
        inner([])
    assert inner.generated_tokens == 4
    assert inner.log[-1]["tokens"] is None


def test_fit_seed_reset_is_independent_of_arm_order(monkeypatch):
    import fin_skills.tasks.judge as module
    monkeypatch.setattr(module, "_run_worker", lambda *a: dict(
        executed=False, error={"type": "Fixture", "message": "no program"}))
    observed = []
    class SeedModel:
        seed = 999
        def __call__(self, messages):
            return "def solve(prices, split_factors): return prices"
    class Probe(rsi.NoImprove):
        def fit(self, generate, dev_tasks, **kwargs):
            observed.append(generate.seed)
            return super().fit(generate, dev_tasks, **kwargs)
    rsi.benchmark([Probe], SeedModel(), suite(), seeds=(23, 37), library=(True, False),
                  fit_seeds=(11,), heldout=[suite().heldout[0]])
    assert observed == [11, 11]
    with pytest.raises(ValueError, match="exactly one"):
        rsi.benchmark([Probe], SeedModel(), suite(), fit_seeds=(11, 23))


def test_terminal_run_cannot_be_extended(tmp_path):
    from fin_skills.rsi.run_record import RunRecord
    record = RunRecord(tmp_path / "run.jsonl", {"plan": []})
    record.state("complete")
    for action in (lambda: record.append({}), lambda: record.event({}),
                   lambda: record.state("invalid")):
        with pytest.raises(ValueError, match="terminal"):
            action()


def test_real_model_mode_fails_closed_before_dispatch(tmp_path, monkeypatch):
    monkeypatch.delenv("FIN_TASKS_CONFINE", raising=False)
    def unused(messages):
        raise AssertionError("no model dispatch permitted")
    with pytest.raises(ValueError, match="confinement"):
        rsi.benchmark([rsi.NoImprove], unused, suite(), out=tmp_path / "run.jsonl",
                      model_id="fixture", execution_mode="model")
    assert not list(tmp_path.iterdir())


def test_contrasts_pair_seeds_and_cluster_algorithm_variants():
    rows = []
    for method, result in (("a", True), ("b", False), ("c", True)):
        for alg in ("one", "two", "three"):
            for variant in ("x", "y"):
                for seed in (11, 23):
                    rows.append(dict(method=method, library=False, task_id=alg+variant,
                                     task_sha256=alg+variant, task_algorithm=alg,
                                     seed=seed, valid=result))
    report = Report(rows=rows, complete=True, protocol={
        "plan": [[r[k] for k in ("method", "library", "task_id", "seed")] for r in rows],
        "contrasts": [[["a", False], ["b", False]], [["a", False], ["c", False]]]})
    effects = report.contrasts(draws=100)
    assert effects[0]["algorithm_clusters"] == 3 and effects[0]["pairs"] == 12
    assert effects[0]["effect"] == 1 and effects[0]["p_sign_flip"] == .25
    assert effects[0]["p_holm"] == .5 and effects[1]["effect"] == 0
    report.rows[-1]["task_sha256"] = "mismatch"
    with pytest.raises(ValueError, match="paired"):
        report.contrasts(draws=100)


@pytest.mark.parametrize("phase", ["fit", "solve"])
def test_task_mutation_cannot_produce_a_complete_receipt(tmp_path, phase):
    from fin_skills.rsi.methods import Episode
    tasks = suite()
    class Mutator(rsi.NoImprove):
        def fit(self, generate, dev_tasks, **kwargs):
            if phase == "fit":
                dev_tasks[0].parameters["nested"] = {"changed": True}
            return self
        def solve(self, generate, task, *, seed, budget):
            task.parameters["nested"] = {"changed": True}
            return Episode(self.name, task.id, seed, self.library)
    out = tmp_path / "mutant.jsonl"
    with pytest.raises(ValueError, match="changed"):
        rsi.benchmark([Mutator], lambda _: "unused", tasks, seeds=(11,), library=(True,),
                      heldout=[tasks.heldout[0]], out=out)
    assert json.loads(out.with_name(out.name + ".terminal.json").read_text())["status"] == "invalid"
    assert out.read_text() == ""


@pytest.mark.parametrize("nested", [False, True])
def test_invalid_content_retains_known_cost_and_original_receipt(nested):
    from fin_skills.rsi.meter import EVENT_SINK
    events = []
    token = EVENT_SINK.set(events.append)
    response = {"choices": [{"message": {"content": None}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 7, "prompt_tokens": 12}}
    outer = rsi.Meter(lambda _: response, rsi.FitBudget())
    meter = rsi.Meter(outer.respond, rsi.Budget()) if nested else outer
    try:
        with pytest.raises(TypeError):
            meter([])
    finally:
        EVENT_SINK.reset(token)
    assert meter.generated_tokens == outer.generated_tokens == 7
    assert any(e["kind"] == "response_received" and e["response"] == response for e in events)
    assert all(e["token_cost"] == 7 for e in events if e["kind"] == "call_failed")


def test_explicit_refusal_is_recorded_as_a_model_outcome():
    response = {"choices": [{"message": {"content": None, "refusal": "fixture refusal"},
                             "finish_reason": "stop"}], "usage": {"completion_tokens": 7}}
    episode = rsi.NoImprove().solve(lambda _: response, suite().heldout[0], seed=11)
    assert episode.status == "model_refusal" and episode.usage["generated_tokens"] == 7
    assert not episode.accepted


@pytest.mark.parametrize("usage,cost", [(None, 0), ({"completion_tokens": 7}, 7),
    ({"completion_tokens": 2, "completion_tokens_estimated": True}, 2)])
def test_bounded_refusal_cannot_skip_backend_contract(usage, cost):
    class Backend:
        def __call__(self, messages):
            raise AssertionError("unbounded path")
        def generate_bounded(self, messages, **kwargs):
            return {"choices": [{"message": {"content": None, "refusal": "fixture"}}],
                    "usage": usage}
    meter = rsi.Meter(Backend(), rsi.Budget(generated_tokens=3))
    with pytest.raises(RuntimeError, match="exact usage/output limit"):
        meter([])
    assert meter.calls == 1 and meter.generated_tokens == cost


def test_backend_configuration_change_is_rejected_before_dispatch():
    from fin_skills.rsi.backend import describe_backend
    from fin_skills.rsi.meter import BACKEND_BINDING
    class Model:
        revision = "fixed-revision"
        calls = 0
        def describe(self):
            return dict(model_id="fixture", model_revision=self.revision,
                        tokenizer_revision="fixed-tokenizer", precision="bf16",
                        context_tokens=4096, max_new_tokens=128, temperature=0., top_p=1.)
        def __call__(self, messages):
            self.calls += 1
            return "unused"
    model = Model()
    original = describe_backend(model, strict=True)
    token = BACKEND_BINDING.set((model, original))
    try:
        model.revision = "changed"
        with pytest.raises(ValueError, match="descriptor changed"):
            rsi.Meter(model, rsi.Budget())([])
    finally:
        BACKEND_BINDING.reset(token)
    assert model.calls == 0


def test_runtime_bundle_binds_hook_and_public_resources(tmp_path):
    from fin_skills.rsi.run_record import RunRecord, source_hashes
    hashes = source_hashes()
    record = RunRecord(tmp_path / "frozen.jsonl", {"plan": [], "source_hashes": hashes})
    import hashlib
    for name in ("benchmarks/agent_study/linux_sandbox.py", "fin_skills/tasks/worker.py"):
        assert hashlib.sha256((record.bundle / name).read_bytes()).hexdigest() == hashes[name]
    record.state("complete")


def test_repeated_sampling_never_receives_previous_failure(monkeypatch):
    import fin_skills.tasks.judge as module
    monkeypatch.setattr(module, "_run_worker", lambda *a: dict(
        executed=False, error={"type": "Fixture", "message": "failure must stay hidden"}))
    prompts = []
    def generate(messages):
        prompts.append(messages)
        return "def solve(prices, split_factors): return prices"
    episode = rsi.RepeatedSampling().solve(generate, suite().heldout[0], seed=11,
                                          budget=rsi.Budget(attempts=2))
    assert episode.usage["attempts"] == 2 and len(prompts) == 2
    assert prompts[0] == prompts[1]
    assert "failure must stay hidden" not in str(prompts)
