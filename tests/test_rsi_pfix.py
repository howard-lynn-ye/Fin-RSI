"""P-fix arms: the frozen P track on the current library interface (scripted models only)."""
import hashlib
import json

import pytest

from fin_skills.rsi import adopt, pfix, study
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
    assert hashlib.sha256(plain.encode()).hexdigest().startswith("120c321f67cd")  # F1/F2 as run
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
    for cls in pfix.LIBRARY + pfix.DOCUMENTED:
        with pytest.raises(ValueError):
            cls(library=False)


def test_manifest_binds_72_packets():
    m = pfix.manifest(HELDOUT)
    assert len(m["packets"]) == 4 * 6 * 3 and m["exploratory"] is True
    assert m["interface_sha256"] != m["documented_interface_sha256"]
    assert m["budget"]["attempts"] == 3 and m["output_cap"] == 4096
    assert json.dumps(m, sort_keys=True) == json.dumps(pfix.manifest(HELDOUT), sort_keys=True)


def test_runner_units_partition_the_arms():
    from scripts import run_p_fix
    names = [m.name for methods, _ in run_p_fix.UNITS.values() for m in methods]
    assert sorted(names) == ["check_F0", "check_F1", "check_F2", "check_F3"]
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
