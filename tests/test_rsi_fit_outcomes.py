"""Fit outcomes retain the model and cost; transport faults still invalidate the run."""
import pytest

from fin_skills import rsi
from fin_skills.rsi import methods
from fin_skills.tasks import frozen_suite


def reply(content=None, refusal=None):
    return {"choices": [{"message": {"content": content, "refusal": refusal}}],
            "usage": {"completion_tokens": 7}}


@pytest.mark.parametrize("cls", [rsi.PlaybookLearning, rsi.Evolve])
@pytest.mark.parametrize("failure", ["writer_refusal", "proposer_refusal", "invalid_proposal"])
def test_fit_model_outcomes_freeze_state_and_charge_every_call(cls, failure, monkeypatch):
    calls = []
    def backend(messages):
        calls.append(messages)
        proposer = any("Propose ONE small" in m["content"] or
                       "improving the standing instructions" in m["content"] for m in messages)
        if failure == "writer_refusal" or (proposer and failure == "proposer_refusal"):
            return reply(refusal="declined")
        return reply("" if proposer else "writer response")
    # Exercise the real fit controller and meters without running irrelevant numerical code.
    def failed_episode(self, meter, judge, task, seed, episode, **kwargs):
        meter([dict(role="user", content="writer")])
        episode.status = "exhausted"
    monkeypatch.setattr(methods._Learner, "_loop", failed_episode)
    method = cls(library=True).fit(backend, frozen_suite().dev,
        budget=rsi.FitBudget(episodes=24, proposals=2))
    assert method.fitted and method.addendum == ""
    assert method.fit_usage["calls"] == len(calls)
    assert method.fit_usage["generated_tokens"] == 7 * len(calls)
    assert method.fit_usage["stopped_reason"] == (
        "invalid_proposal" if failure == "invalid_proposal" else "model_refusal")
    if failure == "writer_refusal":
        assert method.fit_usage["episode_statuses"]["model_refusal"] > 0


@pytest.mark.parametrize("cls", [rsi.PlaybookLearning, rsi.Evolve])
def test_fit_transport_failure_is_not_converted_into_a_model_outcome(cls):
    def offline(_):
        raise ConnectionError("synthetic transport fault")
    with pytest.raises(ConnectionError, match="transport fault"):
        cls(library=True).fit(offline, frozen_suite().dev)


@pytest.mark.parametrize("cls", [rsi.PlaybookLearning, rsi.Evolve])
def test_fit_refusal_preserves_an_earlier_accepted_update(cls, monkeypatch):
    import json
    selected = [frozen_suite().get(name) for name in
                ("e4_ma_crossover", "momentum_a", "e4_vol_target_momentum", "macd_a")]
    always_wrong = sorted(selected, key=lambda task: task.id)[0].id
    proposal_count, calls = 0, []
    def backend(messages):
        nonlocal proposal_count
        calls.append(messages)
        if any("Propose ONE small" in m["content"] or
               "improving the standing instructions" in m["content"] for m in messages):
            proposal_count += 1
            if proposal_count == 2:
                return reply(refusal="declined second proposal")
            return reply(json.dumps(dict(key="lesson", surface="experience",
                text="legal improvement", rationale="synthetic development evidence"))
                if cls is rsi.PlaybookLearning else "legal improvement")
        return reply("writer response")
    class Grader:
        def __init__(self, *a, **k):
            pass
        def check(self, code):
            raise AssertionError("synthetic learning-state fixture must not execute a program")
        def grade(self, code):
            return code == "fixed"
    def candidate(self, meter, judge, task, seed, episode, *, extra=""):
        meter([dict(role="user", content="writer")])
        passed = "legal improvement" in extra and task.id != always_wrong
        episode.steps.append(methods.Step(1, "fixed" if passed else "wrong",
            True, passed, {"synthetic_check": passed}, feedback="failed" if not passed else ""))
        episode.status = "accepted" if passed else "exhausted"
    monkeypatch.setattr(methods, "Judge", Grader)
    monkeypatch.setattr(methods._Learner, "_loop", candidate)
    method = cls(library=True).fit(backend, selected,
        budget=rsi.FitBudget(episodes=24, proposals=2))
    assert proposal_count == 2
    assert "legal improvement" in method.addendum and method.fitted
    assert method.fit_usage["stopped_reason"] == "model_refusal"
    assert method.fit_usage["calls"] == len(calls)
    assert method.fit_usage["generated_tokens"] == 7 * len(calls)


@pytest.mark.parametrize("cls", [rsi.PlaybookLearning, rsi.Evolve])
def test_fit_budget_freezes_without_extra_calls(cls, monkeypatch):
    calls = []
    def backend(messages):
        calls.append(messages)
        return reply("writer response")
    def failed_episode(self, meter, judge, task, seed, episode, **kwargs):
        meter([dict(role="user", content="writer")])
        episode.status = "exhausted"
    monkeypatch.setattr(methods._Learner, "_loop", failed_episode)
    method = cls(library=True).fit(backend, frozen_suite().dev,
        budget=rsi.FitBudget(episodes=1, calls=1, proposals=2))
    assert method.fitted and method.addendum == ""
    assert method.fit_usage["stopped_reason"] == "budget_exhausted"
    assert method.fit_usage["episodes"] == method.fit_usage["calls"] == len(calls) == 1
    assert method.fit_usage["generated_tokens"] == 7


def test_evolve_cannot_improve_by_refusing_a_low_scoring_task():
    def episode(task, checks=None):
        item = methods.Episode("evolve", task, 11, True, valid=False,
                               status="model_refusal" if checks is None else "exhausted")
        if checks is not None:
            item.steps.append(methods.Step(1, "candidate", True, False, checks))
        return item
    partial = episode("A", {"finite": True, "split": True, "lag": False})
    wrong = episode("B", {"finite": False, "split": False, "lag": False})
    refusal = episode("B")
    method = rsi.Evolve()
    baseline = method._score([partial, wrong])
    candidate = method._score([partial, refusal])
    assert baseline == candidate == (0, 1 / 3)
    assert not candidate > baseline
    partial.steps[-1].executed = False
    assert method._score([partial, refusal]) == (0, 0)
