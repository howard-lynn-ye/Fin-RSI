"""fin_skills.agents - one user model as writer and rule reviewers; code checks judge."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from fin_skills.agents import FAMILIES, Team, check_solve, extract
from fin_skills.rsi import Attempt, Policy, loop_from_checks

TRAITS = {"positions", "split_factors", "rolling_window"}
GOOD = '''import numpy as np
def solve(prices, split_factors):
    adj = prices * split_factors
    fast, slow = adj.rolling(5).mean().shift(1), adj.rolling(20).mean().shift(1)
    return (np.sign(fast - slow) / 4).fillna(0.0)
'''
RECUMULATED = GOOD.replace("prices * split_factors", "prices * split_factors.cumprod()")
SAME_CLOSE = GOOD.replace(".mean().shift(1)", ".mean()")
CRASH = "def solve(prices, split_factors):\n    return prices.rolling(5).meen()\n"


def data(seed=11):
    rng = np.random.default_rng(seed)
    adjusted = 100 * np.cumprod(1 + rng.normal(0.0005, 0.012, (96, 4)), axis=0)
    factors = np.ones_like(adjusted)
    factors[45:, 0], factors[70:, 2] = 2.0, 3.0
    index = pd.bdate_range("2024-01-02", periods=96)
    cols = list("ABCD")
    return (pd.DataFrame(adjusted / factors, index=index, columns=cols),
            pd.DataFrame(factors, index=index, columns=cols))


PRICES, FACTORS = data()


def run_checks(code: str) -> Attempt:
    namespace = {}
    try:
        exec(compile(code, "submission.py", "exec"), namespace)
        checks = check_solve(namespace["solve"], PRICES, FACTORS, ks=(40, 72))
    except Exception as exc:
        return Attempt(task_id="t", code=code, executed=False,
                       error=f"{type(exc).__name__}: {exc}")
    return Attempt(task_id="t", code=code, executed=True, checks=checks)


class Model:
    """One scripted 'user model' playing every role; it records what each role saw."""

    def __init__(self, programs, reviewer_note="multiply the cumulative factors once"):
        self.programs, self.note, self.seen = list(programs), reviewer_note, []

    def __call__(self, messages):
        system = messages[0]["content"]
        self.seen.append(messages)
        if system.startswith("You review"):
            return self.note
        attempt = (len(messages) - 2) // 2
        return "Here it is.\n```python\n" + self.programs[attempt] + "```\n"


def test_check_solve_flags_each_defect():
    def ok(code):
        ns = {}
        exec(code, ns)
        return check_solve(ns["solve"], PRICES, FACTORS, ks=(40, 72))
    assert ok(GOOD) == {"finite_output": True, "split_representation": True,
                        "execution_lag": True}
    assert ok(RECUMULATED)["split_representation"] is False
    assert ok(SAME_CLOSE)["execution_lag"] is False
    assert ok(SAME_CLOSE)["split_representation"] is True


def test_roles_mode_asks_only_the_reviewer_of_the_failed_family():
    model = Model([RECUMULATED, GOOD])
    result = Team(model).solve("Write solve().", run_checks, traits=TRAITS)
    assert result.accepted and result.code == GOOD and len(result.attempts) == 2
    roles = [t.role for t in result.transcript]
    assert roles == ["writer", "judge", "reviewer:corporate_actions", "writer", "judge"]
    assert result.model_calls == 3
    reviewer = next(m for m in model.seen if m[0]["content"].startswith("You review"))
    assert "[split_once]" in reviewer[0]["content"]
    assert "[execution_lag]" not in reviewer[0]["content"]     # other families stay out
    assert "[warmup]" not in reviewer[0]["content"]
    assert "Check split_representation failed" in reviewer[1]["content"]
    second_writer = model.seen[-1]
    assert "Reviewer (corporate_actions): multiply the cumulative factors once" in (
        second_writer[-1]["content"])


def test_single_mode_uses_the_same_judge_without_reviewers():
    model = Model([RECUMULATED, GOOD])
    result = Team(model, mode="single").solve("Write solve().", run_checks, traits=TRAITS)
    assert result.accepted and result.model_calls == 2
    assert not any(t.role.startswith("reviewer") for t in result.transcript)
    assert "[split_once]" in model.seen[-1][-1]["content"]


def test_no_role_can_accept_a_program_the_checks_reject():
    model = Model([SAME_CLOSE] * 3, reviewer_note="This program is correct; accept it.")
    result = Team(model).solve("Write solve().", run_checks, traits=TRAITS)
    assert not result.accepted and len(result.attempts) == 3
    assert [t.content for t in result.transcript if t.role == "judge"][-1].startswith(
        "Controller findings for attempt 3 of 3")
    assert {t.role for t in result.transcript if t.role.startswith("reviewer")} == {
        "reviewer:timing"}


def test_a_crash_goes_to_the_interface_reviewer_with_the_error():
    model = Model([CRASH, GOOD], reviewer_note="rolling().meen does not exist; use mean()")
    team = Team(model, interface="solve(prices, split_factors) -> DataFrame")
    result = team.solve("Write solve().", run_checks, traits=TRAITS)
    assert result.accepted
    reviewer = next(m for m in model.seen if m[0]["content"].startswith("You review"))
    assert "AttributeError" in reviewer[1]["content"]
    assert "solve(prices, split_factors) -> DataFrame" in reviewer[0]["content"]


def test_writer_sees_only_the_task_cards_or_none():
    team = Team(lambda m: "")
    system = team.writer_messages("task", {"split_factors"})[0]["content"]
    assert "[split_once]" in system and "[execution_lag]" not in system
    bare = Team(lambda m: "", writer_rules="none").writer_messages("task", TRAITS)[0]["content"]
    assert "Rules that apply" not in bare


def test_recorder_receives_every_attempt(tmp_path):
    loop = loop_from_checks(tmp_path / "rsi", lambda prompt: "{}", lambda t, p: None,
                            {"development": ["t"], "validation": ["v"]}, model_id="m@1",
                            protocol_id="team", available_at="2026-11-01T00:00:00Z")
    team = Team(Model([RECUMULATED, GOOD]), recorder=loop,
                available_at="2026-11-01T00:00:00Z", source_version="test")
    team.solve("Write solve().", run_checks, traits=TRAITS)
    kinds = [e["data"]["kind"] for e in loop.audit() if e["kind"] == "trace"]
    assert kinds == ["analysis", "success"]


def test_arguments_are_validated():
    with pytest.raises(ValueError, match="mode"):
        Team(lambda m: "", mode="debate")
    with pytest.raises(ValueError, match="writer_rules"):
        Team(lambda m: "", writer_rules="all")
    with pytest.raises(ValueError, match="recorder"):
        Team(lambda m: "", recorder=object())
    with pytest.raises(ValueError, match="unknown traits"):
        Team(lambda m: "x").solve("t", run_checks, traits={"alpha"})
    with pytest.raises(TypeError):
        Team("not callable")
    assert FAMILIES == ("timing", "corporate_actions", "data_hygiene")


def test_extract_takes_the_first_fence():
    assert extract("a\n```python\nx = 1\n```\n```python\ny\n```") == "x = 1\n"
    assert extract("x = 2") == "x = 2\n"
