"""P-adopt arms: what each arm adds to the frozen first prompt, with scripted models only."""
import json

import pytest

from fin_skills import rsi
from fin_skills.rsi import adopt, study
from fin_skills.rsi.meter import Budget
from fin_skills.tasks import frozen_suite
from fin_skills.tasks.judge import Visible

TASKS = frozen_suite()
HELDOUT = TASKS.heldout


def test_retrieval_results_are_fixed_before_any_run():
    table = {r["task_id"]: r for r in adopt.retrieval_table(HELDOUT)}
    assert len(adopt.retrieval_pool()) == 12
    assert {k for k, r in table.items() if r["rank"] == 1} == {
        "rsi_reversion_a", "rsi_reversion_b", "donchian_breakout_a", "donchian_breakout_b"}
    assert all(table[k]["top3"][0] == "equal_weight" for k in
               ("bollinger_reversion_a", "bollinger_reversion_b"))
    assert adopt.retrieval_table(HELDOUT) == adopt.retrieval_table(HELDOUT)


@pytest.mark.parametrize("task", HELDOUT, ids=lambda t: t.id)
def test_arms_differ_only_after_the_task_text(task):
    m = {arm: adopt.adopt_messages(task, seed=101, arm=arm) for arm in adopt.ARMS}
    assert m["L0"] == study.initial_messages(task, seed=101, library=False)
    assert m["L1"] == study.initial_messages(task, seed=101, library=True)
    assert m["L1"][0] == m["L2"][0] == m["L3"][0]
    assert all(m[a][1]["content"].startswith(m["L0"][1]["content"]) for a in adopt.ARMS)
    assert m["L2"][1]["content"].endswith(f'positions("{task.algorithm}", ...).')
    top = adopt.retrieved(task)["id"]
    assert f"\n  {top}: " in m["L3"][1]["content"]
    assert "fin_skills interface:" not in m["L0"][0]["content"]
    for arm in ("L0", "L1"):
        assert "implements this rule" not in m[arm][1]["content"]
        assert "retrieved for this rule" not in m[arm][1]["content"]


def test_arms_refuse_the_other_library_condition():
    with pytest.raises(ValueError):
        adopt.CheckL0(library=True)
    for cls in (adopt.CheckL1, adopt.CheckL2, adopt.CheckL3, adopt.PlaybookL1):
        with pytest.raises(ValueError):
            cls(library=False)
    with pytest.raises(ValueError):
        adopt.PlaybookL0(library=True)


def test_every_arm_gets_failed_check_names_only():
    seen = Visible("program", True, checks={"execution_lag": False})
    notes = {cls.name: cls(library=cls.adopt_arm != "L0")._check_feedback(seen)
             for cls in adopt.SCRATCH + adopt.LIBRARY}
    assert len(set(notes.values())) == 1
    assert notes["check_L0"] == study.Check()._check_feedback(seen)
    assert "Rule: [" not in notes["check_L0"] and "fin_skills" not in notes["check_L0"]


def test_manifest_binds_72_packets_and_contrasts():
    m = adopt.manifest(HELDOUT)
    assert len(m["packets"]) == 4 * 6 * 3 and m["exploratory"] is True
    assert m["budget"]["attempts"] == 3 and m["budget"]["generated_tokens"] == 12288
    assert [r["rank"] == 1 for r in m["retrieval"]].count(True) == 4
    assert json.dumps(m, sort_keys=True) == json.dumps(adopt.manifest(HELDOUT), sort_keys=True)


def test_check_arms_run_through_the_benchmark(tmp_path, monkeypatch):
    import fin_skills.tasks.judge as judge_module
    monkeypatch.setattr(judge_module, "_run_worker", lambda *a: dict(
        executed=False, error={"type": "Fixture", "message": "synthetic failure"}))
    prompts = []
    def generate(messages):
        prompts.append(messages)
        return "def solve(prices, split_factors): return prices"
    reports = [rsi.benchmark(methods, generate, TASKS, seeds=(101,), library=(library,),
                             budget=Budget(attempts=1), heldout=[HELDOUT[0]],
                             out=tmp_path / f"{int(library)}.jsonl")
               for methods, library in (((adopt.CheckL0,), False),
                                        ((adopt.CheckL1, adopt.CheckL2, adopt.CheckL3), True))]
    rows = [r for report in reports for r in report.rows]
    assert sorted(r["method"] for r in rows) == ["check_L0", "check_L1", "check_L2", "check_L3"]
    assert len({json.dumps(p) for p in prompts}) == 4


class _Lesson:
    """Writes the right library call only once a lesson is in its system prompt."""
    LESSON = "LESSON: never shift positions upward; the library already lags them."

    def __call__(self, messages):
        first = messages[0]["content"]
        if first.startswith("Propose ONE small playbook entry"):
            return json.dumps(dict(key="lag", surface="experience", text=self.LESSON,
                                   rationale="execution_lag failures in the traces"))
        user = next(m["content"] for m in messages if m["role"] == "user")
        task = next(t for t in TASKS.tasks if f"Rule:\n{t.definition}\n" in user)
        options = dict(task.library_options)
        params = {k: v for k, v in task.parameters.items() if k not in options}
        tail = "" if self.LESSON in first else ".shift(-1).fillna(0.0)"
        return ("```python\nfrom fin_skills.algorithms import positions\n"
                "def solve(prices, split_factors):\n"
                f"    w = positions({task.algorithm!r}, prices, split_factors, "
                f"factors={task.factors!r}, parameters={params!r}, **{options!r})\n"
                f"    return w{tail}\n```")


def test_playbook_learns_on_adopt_prompts_with_unhinted_traces(monkeypatch):
    from fin_skills.rsi import wiring
    seen_tools, real = [], wiring.observe
    def spy(*args, tools=True, **kwargs):
        seen_tools.append(tools)
        return real(*args, tools=tools, **kwargs)
    monkeypatch.setattr(wiring, "observe", spy)
    dev = [TASKS.get(i) for i in ("e4_ma_crossover", "momentum_a", "e4_vol_target_momentum",
                                  "macd_a")]
    model = _Lesson()
    method = adopt.PlaybookL1(library=True).fit(
        model, dev, budget=rsi.FitBudget(episodes=24, proposals=2))
    assert seen_tools and set(seen_tools) == {False}
    assert model.LESSON in method.addendum
    messages = method._messages(HELDOUT[0], 101, method.addendum)
    assert messages[0]["content"].endswith(method.addendum)
    assert messages[1] == adopt.adopt_messages(HELDOUT[0], seed=101, arm="L1")[1]
    episode = method.solve(model, HELDOUT[0], seed=101, budget=adopt.BUDGET)
    assert episode.status == "accepted" and len(episode.steps) == 1
