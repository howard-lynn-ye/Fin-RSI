"""P-fix arms: the frozen P track on the current library interface (scripted models only)."""
import hashlib
import json

import pytest

from fin_skills.rsi import adopt, pfix, study
from fin_skills.rsi.run_record import digest
from fin_skills.tasks import frozen_suite
from fin_skills.tasks.prompts import library_text

TASKS = frozen_suite()
HELDOUT = TASKS.heldout


@pytest.mark.parametrize("task", HELDOUT, ids=lambda t: t.id)
def test_arms_are_the_frozen_prompt_with_the_current_interface(task):
    m = {arm: pfix.pfix_messages(task, seed=211, arm=arm) for arm in pfix.ARMS}
    assert m["F0"] == adopt.adopt_messages(task, seed=211, arm="L0")
    assert m["F0"] == study.initial_messages(task, seed=211, library=False)
    frozen = study.initial_messages(task, seed=211, library=True)[0]["content"]
    head = frozen.split("\n\nfin_skills interface:\n")[0]
    assert m["F1"][0]["content"] == head + "\n\nfin_skills interface:\n" + pfix.interface_text()
    assert m["F1"][0] == m["F2"][0] and m["F1"][1] == m["F0"][1]
    assert m["F2"][1]["content"] == m["F0"][1]["content"] + "\n\n" + adopt.POINTER.format(
        algorithm=task.algorithm)
    assert m["F3"][1] == m["F0"][1]  # nothing task-specific
    assert m["F3"][0]["content"] == (head + "\n\nfin_skills interface:\n"
                                     + pfix.interface_text(documented=True))


def test_interface_is_the_current_positions_only_text():
    text = pfix.interface_text()
    assert text == library_text(rolling=False, frozen=False) != library_text(rolling=False)
    assert "signal_positions(" in text and "Do not shift" in text
    assert "rolling(" not in text
    with pytest.raises(ValueError):
        library_text(rolling=True, frozen=True)


def test_documented_interface_only_adds_the_rules():
    from fin_skills.tasks.prompts import DEFAULTS, RULES
    plain, documented = pfix.interface_text(), pfix.interface_text(documented=True)
    assert hashlib.sha256(plain.encode()).hexdigest().startswith("04fbb48cc1a5")  # v4 text
    assert "factors='auto'" in plain
    assert set(RULES) == set(DEFAULTS)
    added = {"      " + rule for rule in RULES.values()}
    kept = [line for line in documented.splitlines()
            if line not in added and not line.startswith("Algorithms usable")]
    assert kept == [line for line in plain.splitlines() if not line.startswith("Algorithms usable")]
    for rule in RULES.values():
        assert "      " + rule in documented
    for task in HELDOUT:  # the interface never names a task's own parameters or variant
        assert task.id not in documented
    with pytest.raises(ValueError):
        library_text(rolling=True, documented=True)


def test_arms_refuse_the_other_library_condition():
    with pytest.raises(ValueError):
        pfix.CheckF0(library=True)
    for cls in pfix.LIBRARY + pfix.DOCUMENTED + pfix.ROUTED:
        with pytest.raises(ValueError):
            cls(library=False)


def test_manifest_binds_90_packets():
    m = pfix.manifest(HELDOUT)
    assert len(m["packets"]) == 5 * 6 * 3 and m["exploratory"] is True
    assert m["version"] == "p-fix-v4" and len(m["visible_checks"]) == 6
    routed = [x for x in m["packets"] if x["arm"] == "F4"]
    assert routed[0]["messages_sha256"] == digest(pfix.route_messages(HELDOUT[0], seed=101))
    assert m["interface_sha256"] != m["documented_interface_sha256"]
    assert m["budget"]["attempts"] == 3 and m["output_cap"] == 4096
    assert json.dumps(m, sort_keys=True) == json.dumps(pfix.manifest(HELDOUT), sort_keys=True)


def test_runner_units_partition_the_arms():
    from scripts import run_p_fix
    names = [m.name for methods, _ in run_p_fix.UNITS.values() for m in methods]
    assert sorted(names) == ["check_F0", "check_F1", "check_F2", "check_F3", "check_F4"]
    assert all(lib == (u != "check_F0") for u, (_, lib) in run_p_fix.UNITS.items())


@pytest.mark.parametrize("method", ["CheckF1", "CheckF3"])
def test_a_correct_library_program_is_valid(method):
    """End to end through the real judge: the call P-fix hopes to see is graded valid."""
    from fin_skills import rsi
    from fin_skills.rsi.meter import Budget
    task = next(t for t in HELDOUT if t.id == "rsi_reversion_a")
    program = ("from fin_skills.algorithms import positions\n"
               "def solve(prices, split_factors):\n"
               f"    return positions('{task.algorithm}', prices, split_factors)\n")
    report = rsi.benchmark((getattr(pfix, method),), lambda messages: f"```python\n{program}```",
                           TASKS, seeds=(101,), library=(True,), budget=Budget(attempts=1),
                           heldout=[task])
    row = report.rows[0]
    assert row["method"] == getattr(pfix, method).name and row["library_used"] and row["valid"]


def _program(task, tail=""):
    return ("from fin_skills.algorithms import positions\n"
            "def solve(prices, split_factors):\n"
            f"    return positions('{task.algorithm}', prices, split_factors){tail}\n")


def _routed_run(task, route_reply, program, attempts=3):
    """F4 through the real judge with a scripted model: the routing reply, then programs."""
    from fin_skills import rsi
    from fin_skills.rsi.meter import Budget
    seen = []

    def generate(messages):
        seen.append(messages)
        if messages[0]["content"] == pfix.routing.ROUTE_SYSTEM:
            return route_reply
        return f"```python\n{program}```"
    report = rsi.benchmark((pfix.CheckF4,), generate, TASKS, seeds=(101,), library=(True,),
                           budget=Budget(attempts=attempts), heldout=[task])
    return report.rows[0], seen


def test_f4_adds_the_sentence_for_the_rule_the_model_named():
    task = next(t for t in HELDOUT if t.id == "rsi_reversion_a")
    row, seen = _routed_run(task, f"<think>RSI with simple means</think>\n{task.algorithm}",
                            _program(task))
    assert seen[0] == pfix.route_messages(task, seed=101)          # the declared first prompt
    assert task.algorithm not in seen[0][1]["content"].split("Catalogue")[0]  # no label leaks
    assert seen[1] == pfix.pfix_messages(task, seed=101, arm="F4", routed=task.algorithm)
    assert seen[1] == pfix.pfix_messages(task, seed=101, arm="F2")  # right route = F2's prompt
    assert row["extra"]["routed"] == task.algorithm and row["valid"]
    assert row["usage"]["roles"] == ["router", "writer"] and row["usage"]["calls"] == 2


def test_f4_without_a_route_is_f1_and_a_wrong_route_is_recorded():
    task = next(t for t in HELDOUT if t.id == "rsi_reversion_a")
    row, seen = _routed_run(task, "NONE", _program(task))
    assert seen[1] == pfix.pfix_messages(task, seed=101, arm="F1")
    assert row["extra"]["routed"] is None and row["valid"]
    row, seen = _routed_run(task, "macd", _program(task))
    assert 'positions("macd", ...)' in seen[1][1]["content"] and row["extra"]["routed"] == "macd"
    with pytest.raises(ValueError, match="routed is F4's"):
        pfix.pfix_messages(task, seed=101, arm="F1", routed="macd")


def test_every_arm_is_judged_with_the_declared_property_checks():
    """A library result divided by the number of assets again passed the three old checks."""
    task = next(t for t in HELDOUT if t.id == "rsi_reversion_a")
    from fin_skills import rsi
    from fin_skills.rsi.meter import Budget
    twice = _program(task, " / prices.shape[1]")
    report = rsi.benchmark((pfix.CheckF1,), lambda messages: f"```python\n{twice}```", TASKS,
                           seeds=(101,), library=(True,), budget=Budget(attempts=2),
                           heldout=[task])
    row = report.rows[0]
    assert set(row["steps"][0]["checks"]) == set(pfix.VISIBLE_CHECKS)
    assert not row["accepted"] and not row["valid"]
    assert row["steps"][0]["feedback"].startswith("Failed checks: position_scale.")
