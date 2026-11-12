"""Check intervention differences with synthetic models, not capability scores."""
import json

import pytest

from fin_skills.tasks import frozen_suite
from fin_skills.tasks.judge import Visible
from fin_skills.rsi import study
from fin_skills.rsi.meter import Budget


def test_initial_prompts_vary_only_by_library_and_have_no_teaching_cards():
    task = frozen_suite().heldout[0]
    for library in (False, True):
        packets = [cls(library=library)._messages(task, 11) for cls in study.METHODS]
        assert packets[0] == packets[1] == packets[2]
        assert "Rule: [" not in packets[0][0]["content"]
        assert ("fin_skills interface:" in packets[0][0]["content"]) is library
    assert study.initial_messages(task, seed=11, library=False)[1] == study.initial_messages(
        task, seed=11, library=True)[1]


def test_explanation_is_same_in_both_library_arms_and_check_arm_has_only_names():
    seen = Visible("program", True, checks={"execution_lag": False})
    explained = [study.Explain(library=lib)._check_feedback(seen) for lib in (False, True)]
    checked = [study.Check(library=lib)._check_feedback(seen) for lib in (False, True)]
    assert explained[0] == explained[1] and checked[0] == checked[1]
    assert "execution_lag" in checked[0] and "Rule: [" not in checked[0]
    assert "Rule: [" in explained[0] and "fin_skills" not in explained[0]


@pytest.mark.parametrize("cls", study.METHODS)
def test_all_arms_use_first_visible_acceptance_and_same_attempt_ceiling(cls, monkeypatch):
    import fin_skills.tasks.judge as judge_module
    replies = iter([dict(executed=True, checks={"execution_lag": False},
                         positions=[[0.]], library_calls=0),
                    dict(executed=True, checks={"execution_lag": True},
                         positions=[[0.]], library_calls=0)])
    monkeypatch.setattr(judge_module, "_run_worker", lambda *a: next(replies))
    prompts = []
    def generate(messages):
        prompts.append(messages)
        return "def solve(prices, split_factors): return prices * 0"
    ep = cls().solve(generate, frozen_suite().heldout[0], seed=11, budget=Budget(attempts=3))
    assert ep.accepted and len(ep.steps) == ep.usage["calls"] == 2
    assert (prompts[0] == prompts[1]) is (cls is study.Repeat)
    assert ep.valid is None  # no reference score influenced the stop decision


def test_six_arm_manifest_and_first_attempt_are_bound_before_calls(tmp_path, monkeypatch):
    import fin_skills.tasks.judge as judge_module
    monkeypatch.setattr(judge_module, "_run_worker", lambda *a: dict(
        executed=False, error={"type": "Fixture", "message": "synthetic failure"}))
    out = tmp_path / "six.jsonl"
    def generate(messages):
        frozen = json.loads(out.with_name(out.name + ".protocol.json").read_text())
        assert len(frozen["study_manifest"]["packets"]) == 6
        return "def solve(prices, split_factors): return prices"
    report = study.run(generate, frozen_suite(), heldout=[frozen_suite().heldout[0]], seeds=(11,),
                       budget=Budget(attempts=1), out=out)
    assert report.complete and len(report.rows) == 6
    assert all(r["first_valid"] is False for r in report.rows)
    assert len(report.protocol["contrasts"]) == 3


def test_repeated_code_does_not_overwrite_first_attempt_grade(monkeypatch):
    import numpy as np
    import fin_skills.tasks.judge as judge_module
    from fin_skills.rsi import benchmark
    responses = iter([dict(executed=True, checks={"execution_lag": False},
                           positions=[[1.]], library_calls=0),
                      dict(executed=True, checks={"execution_lag": True},
                           positions=[[0.]], library_calls=0)])
    monkeypatch.setattr(judge_module, "_run_worker", lambda *a: next(responses))
    monkeypatch.setattr(judge_module, "reference", lambda *a: np.array([[0.]]))
    report = benchmark([study.Repeat], lambda _: "same code", frozen_suite(), seeds=(11,),
                       library=(False,), heldout=[frozen_suite().heldout[0]])
    row = report.rows[0]
    assert row["valid"] is True and row["first_valid"] is False
    assert [s["attempt_index"] for s in row["steps"]] == [0, 1]
