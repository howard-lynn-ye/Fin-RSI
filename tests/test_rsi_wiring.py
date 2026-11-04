"""fin_skills.rsi.wiring - an RSI loop driven by property-check results."""
from __future__ import annotations

import json

import pytest

from fin_skills.rsi import Policy
from fin_skills.rsi.wiring import Attempt, check_evaluator, loop_from_checks, observe, traces

T0, T1 = "2026-11-01T00:00:00Z", "2026-11-02T00:00:00Z"
DEV = [f"dev-{i}" for i in range(4)]
VAL = [f"val-{i}" for i in range(4)]
LESSON = "split_factors are already cumulative; multiply once and never cumprod them"


def fake_model(task_id: str, playbook: str) -> Attempt:
    """A deterministic stand-in for model + sandbox + checks + hidden grader.

    Without the lesson every task re-cumulates split factors (split check fails). With the
    lesson in the playbook, three tasks in four become valid; the fourth still fails on a
    timing check, so some task always fails - the case the all_pass rule cannot handle.
    """
    knows = LESSON in playbook
    hard = task_id.endswith("-3")
    checks = {"split_representation": knows, "execution_lag": not hard}
    valid = knows and not hard
    return Attempt(task_id=task_id, code="def solve(p, f): ...", executed=True,
                   checks=checks, valid=valid)


def proposer(prompt: str) -> str:
    assert "val-" not in prompt                 # validation tasks never reach the proposer
    assert '"valid"' not in prompt              # nor does the hidden grader's verdict
    return json.dumps({"key": "split_factors", "surface": "experience", "text": LESSON,
                       "rationale": "development traces fail the split check"})


def make(tmp_path, policy):
    loop = loop_from_checks(tmp_path / "rsi", proposer, fake_model,
                            {"development": DEV, "validation": VAL}, model_id="fake@1",
                            protocol_id="wiring-test", available_at=T0, policy=policy,
                            evidence_dir=tmp_path / "evidence")
    observe(loop, [fake_model(t, "") for t in DEV], available_at=T0, source_version="test")
    return loop


def test_no_regression_rule_accepts_a_lesson_that_fixes_most_tasks(tmp_path):
    loop = make(tmp_path, Policy(enabled=True, guard_rule="no_regression"))
    decision = loop.improve(as_of=T1)
    assert decision.status == "accepted", decision.reason
    assert LESSON in loop.instructions(as_of=T1)
    # every receipt hash names an evidence file that exists
    shas = {p.stem for p in (tmp_path / "evidence").glob("*.json")}
    for event in loop.audit():
        if event["kind"] == "evaluation":
            assert {c["evidence_sha256"] for c in event["data"]["receipt"]["cases"]} <= shas


def test_all_pass_rule_vetoes_while_any_task_still_fails(tmp_path):
    loop = make(tmp_path, Policy(enabled=True))
    decision = loop.improve(as_of=T1)
    assert decision.status == "rejected"
    assert decision.reason == "candidate failed execution or guards"


def test_breaking_a_task_that_passed_every_check_is_rejected(tmp_path):
    from fin_skills.rsi import Entry, Playbook
    seeded = "seed lesson: " + LESSON

    def flaky(task_id, playbook):
        a = fake_model(task_id, playbook)
        if "extra advice" in playbook and task_id == "val-0":
            return Attempt(task_id=task_id, code=a.code, executed=False, error="TypeError: x")
        return a

    def extra(prompt):
        return json.dumps({"key": "extra", "surface": "experience", "text": "extra advice",
                           "rationale": "test"})

    loop = loop_from_checks(tmp_path / "rsi", extra, flaky,
                            {"development": DEV, "validation": VAL}, model_id="fake@1",
                            protocol_id="wiring-test", available_at=T0,
                            policy=Policy(enabled=True, guard_rule="no_regression"),
                            seed=Playbook((Entry("split_factors", "experience", seeded),)))
    observe(loop, [flaky(t, seeded) for t in DEV], available_at=T0, source_version="test")
    decision = loop.improve(as_of=T1)
    assert decision.status == "rejected"
    assert decision.reason == "candidate broke a case that passed its checks"


def test_traces_carry_cause_and_rule_but_not_the_grader_verdict():
    bad = fake_model("dev-0", "")
    (tr,) = traces(bad, available_at=T0, source_version="test")
    assert tr.kind == "analysis"
    assert "Check split_representation failed" in tr.summary and "[split_once]" in tr.summary
    assert "valid" not in tr.result
    crash = Attempt(task_id="dev-1", code="x", executed=False, error="NameError: returns")
    (tr2,) = traces(crash, available_at=T0, source_version="test")
    assert tr2.kind == "tool_contract" and "NameError" in tr2.summary
    ok = Attempt(task_id="dev-2", code="x", executed=True, checks={"execution_lag": True},
                 valid=True)
    assert traces(ok, available_at=T0, source_version="test")[0].kind == "success"


def test_traces_follow_the_run_tools_and_rules():
    bad = fake_model("dev-0", "")
    with_tools = traces(bad, available_at=T0, source_version="test")[0].summary
    assert "fin_skills.algorithms.positions" in with_tools
    no_tools = traces(bad, available_at=T0, source_version="test", tools=False)[0].summary
    assert "[split_once]" in no_tools and "fin_skills" not in no_tools
    causes = traces(bad, available_at=T0, source_version="test", rules=False)[0].summary
    assert "Check split_representation failed" in causes and "Rule:" not in causes


def test_attempt_and_evaluator_validation():
    with pytest.raises(ValueError, match="valid attempt"):
        Attempt(task_id="t", code="", executed=False, valid=True)
    with pytest.raises(ValueError, match="disjoint"):
        check_evaluator(fake_model, {"development": ["a"], "validation": ["a"]},
                        protocol_id="p")
    with pytest.raises(ValueError, match="score"):
        check_evaluator(fake_model, {"development": ["a"], "validation": ["b"]},
                        protocol_id="p", score="profit")
    with pytest.raises(ValueError, match="guard_rule"):
        Policy(enabled=True, guard_rule="lenient")


def test_default_policy_header_is_unchanged_for_existing_archives(tmp_path):
    from dataclasses import asdict
    from fin_skills.rsi.core import _policy_header
    assert "guard_rule" not in _policy_header(Policy())
    assert _policy_header(Policy(guard_rule="no_regression"))["guard_rule"] == "no_regression"
    assert set(_policy_header(Policy())) == set(asdict(Policy())) - {"guard_rule"}
