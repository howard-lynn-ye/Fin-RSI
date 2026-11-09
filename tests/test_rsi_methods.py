"""fin_skills.rsi methods, meter and benchmark, with scripted models (no real LLM)."""
from __future__ import annotations

import json

import pytest

from fin_skills import rsi
from fin_skills.tasks import suite

TASKS = suite()
LESSON = "LESSON: never shift positions upward; the library already lags them."


def _program(task, *, bug: bool) -> str:
    options = dict(task.library_options)
    params = {k: v for k, v in task.parameters.items() if k not in options}
    tail = ".shift(-1).fillna(0.0)" if bug else ""       # look-ahead: fails execution_lag
    return ("```python\nfrom fin_skills.algorithms import positions\n"
            "def solve(prices, split_factors):\n"
            f"    w = positions({task.algorithm!r}, prices, split_factors, "
            f"factors={task.factors!r}, parameters={params!r}, **{options!r})\n"
            f"    return w{tail}\n```")


def _task_of(messages):
    user = next(m["content"] for m in messages if m["role"] == "user")
    return next(t for t in TASKS.tasks if f"Rule:\n{t.definition}\n" in user)


class LessonModel:
    """Writes the correct library call only when LESSON is in its system prompt (or, with
    learns_from_feedback, once it has been told what failed). Proposers return LESSON."""

    def __init__(self, *, uses_lessons=True, learns_from_feedback=False):
        self.uses_lessons, self.learns_from_feedback = uses_lessons, learns_from_feedback
        self.roles = []

    def __call__(self, messages):
        first = messages[0]["content"]
        if first.startswith("Propose ONE small playbook entry"):
            self.roles.append("proposer")
            return json.dumps(dict(key="lag", surface="experience", text=LESSON,
                                   rationale="execution_lag failures in the traces"))
        if "improving the standing instructions" in first:
            self.roles.append("mutate")
            return LESSON
        if first.startswith(("You wrote the program below", "You review")):
            self.roles.append("helper")
            return "The position reads a later close; remove the upward shift."
        self.roles.append("writer")
        task = _task_of(messages)
        fixed = (self.uses_lessons and LESSON in first) or (
            self.learns_from_feedback and len(messages) > 2)
        return _program(task, bug=not fixed)


HELDOUT = TASKS.heldout[0]
DEV = [TASKS.get(i) for i in ("e4_ma_crossover", "momentum_a", "e4_vol_target_momentum",
                              "macd_a")]


def test_meter_counts_and_refuses_beyond_budget():
    m = rsi.Meter(lambda msgs: "x" * 40, rsi.Budget(attempts=1, calls=2, generated_tokens=99))
    m([{"role": "user", "content": "a"}])
    m([{"role": "user", "content": "b"}], role="reflection")
    with pytest.raises(rsi.BudgetExceeded) as err:
        m([{"role": "user", "content": "c"}])
    assert err.value.limit == "calls" and m.usage()["roles"] == ["writer", "reflection"]
    assert m.generated_tokens == 20 and m.estimated_tokens
    m.charge_attempt()
    with pytest.raises(rsi.BudgetExceeded):
        m.charge_attempt()


def test_meter_uses_backend_token_counts_and_nests():
    reply = {"choices": [{"message": {"content": "ok"}}], "usage": {"completion_tokens": 7}}
    outer = rsi.Meter(lambda msgs: reply, rsi.FitBudget(calls=5))
    inner = rsi.Meter(outer.respond, rsi.Budget(calls=3))
    assert inner([{"role": "user", "content": "q"}]) == "ok"
    assert inner.generated_tokens == outer.generated_tokens == 7 and not inner.estimated_tokens


@pytest.mark.parametrize("cls,status,steps,valid", [
    (rsi.NoImprove, "ran", 1, False),
    (rsi.SelfDebug, "ran", 1, False),          # it runs, so self-debug stops: checks unseen
    (rsi.CheckFeedback, "accepted", 2, True),
    (rsi.Reflexion, "accepted", 2, True),
    (rsi.Team, "accepted", 2, True),
])
def test_methods_on_a_lookahead_bug(cls, status, steps, valid):
    model = LessonModel(learns_from_feedback=True)
    method = cls(library=True)
    episode = method.solve(model, HELDOUT, seed=11, budget=rsi.Budget())
    assert episode.status == status and len(episode.steps) == steps
    assert [step.attempt_index for step in episode.steps] == list(range(steps))
    assert episode.__dict__["_judge"].grade(episode.code) is valid
    assert episode.__dict__["_judge"].grade_attempt(episode.steps[-1].attempt_index) is valid
    assert episode.valid is None                        # methods never fill in the grade
    if cls is rsi.Reflexion:
        assert episode.usage["roles"] == ["writer", "reflection", "writer"]
        assert "Your reflections so far" in episode.steps[0].feedback
    if cls is rsi.Team:
        assert "reviewer" in episode.usage["roles"]


def test_budget_stops_an_episode_and_is_recorded():
    model = LessonModel()                               # never fixes the bug
    episode = rsi.Reflexion(library=True).solve(model, HELDOUT, seed=11,
                                                budget=rsi.Budget(attempts=3, calls=2))
    assert episode.status == "budget:calls" and episode.usage["calls"] == 2
    assert len(episode.steps) == 1


def test_feedback_without_library_has_no_library_calls():
    model = LessonModel(learns_from_feedback=True)
    episode = rsi.CheckFeedback(library=False).solve(model, HELDOUT, seed=11)
    # The library call is refused in the no-library arm, so every attempt fails to run.
    assert not episode.steps[0].executed and "not available" in episode.steps[0].error
    assert "fin_skills.algorithms" not in episode.steps[0].feedback


@pytest.mark.parametrize("cls", [rsi.PlaybookLearning, rsi.Evolve])
def test_learners_keep_a_lesson_that_helps(cls):
    model = LessonModel()
    method = cls(library=True)
    method.fit(model, DEV, budget=rsi.FitBudget(episodes=24, proposals=2))
    assert LESSON in method.addendum and method.fit_usage["episodes"] > 0
    episode = method.solve(model, HELDOUT, seed=11)
    assert episode.status == "accepted" and len(episode.steps) == 1
    assert episode.steps[0].attempt_index == 0
    assert episode.__dict__["_judge"].grade(episode.code)
    assert episode.__dict__["_judge"].grade_attempt(episode.steps[0].attempt_index)
    assert method.state() and episode.state_sha256


@pytest.mark.parametrize("cls", [rsi.PlaybookLearning, rsi.Evolve])
def test_learners_reject_a_lesson_that_does_not_help(cls):
    model = LessonModel(uses_lessons=False)
    method = cls(library=True).fit(model, DEV, budget=rsi.FitBudget(episodes=24, proposals=2))
    assert LESSON not in method.addendum


class ZeroModel(LessonModel):
    """Programs that run and pass every visible check but are wrong (all-zero output)."""

    def __call__(self, messages):
        if "improving the standing instructions" in messages[0]["content"]:
            return super().__call__(messages)
        self.roles.append("writer")
        return "```python\ndef solve(prices, split_factors):\n    return prices * 0.0\n```"


def test_evolve_never_shows_the_proposer_a_hidden_grade():
    model = ZeroModel()
    method = rsi.Evolve(library=False).fit(model, DEV, budget=rsi.FitBudget(proposals=2))
    # Every dev episode is accepted but invalid: nothing visible failed, so no proposal.
    assert "mutate" not in model.roles
    assert [h["generation"] for h in method.history] == [0]
    assert method.history[0]["score"][0] == 0 and method.addendum == ""


def test_learner_must_be_fitted_and_fit_budget_is_enforced():
    with pytest.raises(RuntimeError, match="fit"):
        rsi.Evolve(library=True).solve(LessonModel(), HELDOUT, seed=11)
    method = rsi.Evolve(library=True).fit(LessonModel(), DEV,
                                          budget=rsi.FitBudget(episodes=2, proposals=2))
    assert method.fit_usage["episodes"] == 2
    assert any(h.get("note") == "fit budget exhausted" for h in method.history)


def test_benchmark_grades_after_solving_and_reports(tmp_path):
    out = tmp_path / "episodes.jsonl"
    model = LessonModel(learns_from_feedback=True)
    report = rsi.benchmark([rsi.NoImprove, rsi.CheckFeedback, rsi.Team], model,
                           TASKS, seeds=(11,), library=(True,), heldout=[HELDOUT], out=out)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["method"] for r in rows] == ["no_improve", "check_feedback", "team"]
    assert [r["valid"] for r in rows] == [False, True, True]
    assert rows[-1]["first_valid"] is False
    assert [s["attempt_index"] for s in rows[-1]["steps"]] == [0, 1]
    summary = {s["method"]: s for s in report.summary(draws=200)}
    assert summary["check_feedback"]["valid"] == 1.0 and summary["no_improve"]["valid"] == 0.0
    assert "check_feedback" in report.table()
    with pytest.raises(ValueError, match="development"):
        rsi.benchmark([rsi.NoImprove], LessonModel(), TASKS, heldout=[DEV[0]])
