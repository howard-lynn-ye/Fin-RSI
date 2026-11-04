"""fin_skills.tasks - task definitions, independent references, judge and hidden grader."""
from __future__ import annotations

import numpy as np
import pytest

from fin_skills.algorithms import positions
from fin_skills.tasks import Judge, library_positions, library_text, messages, panel, reference
from fin_skills.tasks import suite
from fin_skills.algorithms import rolling
from fin_skills.tasks.prompts import DEFAULTS, ROLLING_DEFAULTS
from fin_skills.tasks.spec import FORCED_DEV, assign_splits

SEEDS = (11, 23, 37)
TASKS = suite()


@pytest.mark.parametrize("task", TASKS.tasks, ids=lambda t: t.id)
def test_reference_matches_library_on_every_seed(task):
    """Qualification 1: the independent equations and the library compute the same rule."""
    for seed in SEEDS:
        raw, factors, _ = panel(seed, rows=task.rows, factors=task.factors, order=task.order)
        lib = library_positions(task, raw, factors).to_numpy()
        # Closed-form rules must agree to rounding; a numerical optimizer (min_variance, graded
        # at 1e-5) must land well inside the tolerance the task is graded at.
        atol = 1e-12 if task.tolerance <= 1e-8 else task.tolerance / 5
        np.testing.assert_allclose(lib, reference(task, seed), atol=atol, rtol=0)


def test_e4_inputs_are_the_original_inputs():
    from benchmarks.agent_study.codegen_utility import dataset
    for seed in SEEDS:
        raw, factors, _ = panel(seed, rows=96)
        d = dataset(seed)
        np.testing.assert_array_equal(raw.to_numpy(), np.array(d["prices"]))
        np.testing.assert_array_equal(factors.to_numpy(), np.array(d["factors"]))


def test_ratio_factors_compose_to_the_cumulative_ones():
    raw_c, cum, adj_c = panel(11, factors="cumulative")
    raw_r, ratio, adj_r = panel(11, factors="ratio")
    np.testing.assert_array_equal(raw_c.to_numpy(), raw_r.to_numpy())
    np.testing.assert_allclose(ratio.cumprod().to_numpy(), cum.to_numpy(), rtol=0, atol=1e-15)
    assert (ratio.to_numpy() != 1).sum() == 4                   # four split events


def test_split_is_by_algorithm_and_e4_is_always_dev():
    by_algorithm = {}
    for t in TASKS.tasks:
        by_algorithm.setdefault(t.algorithm, set()).add(TASKS.splits[t.algorithm])
    assert all(len(s) == 1 for s in by_algorithm.values())
    assert all(TASKS.splits[a] == "dev" for a in FORCED_DEV)      # E4, qualification
    assert {t.family for t in TASKS.heldout} == {"portfolio", "signal", "volatility",
                                                "risk", "forecast"}
    assert TASKS.heldout and not {t.id for t in TASKS.dev} & {t.id for t in TASKS.heldout}
    assert assign_splits(by_algorithm) == assign_splits(sorted(by_algorithm))   # deterministic


def test_listed_defaults_are_the_defaults_the_algorithms_use():
    raw, factors, _ = panel(23)
    for algorithm_id, defaults in DEFAULTS.items():
        if algorithm_id in ("hrp", "min_variance", "equal_weight", "inverse_volatility"):
            continue                                   # no numeric defaults to compare
        implicit = positions(algorithm_id, raw, factors)
        explicit = positions(algorithm_id, raw, factors, parameters=dict(defaults))
        np.testing.assert_array_equal(implicit.to_numpy(), explicit.to_numpy())


def test_listed_rolling_defaults_are_the_defaults_the_algorithms_use():
    raw, factors, _ = panel(23)
    for algorithm_id, defaults in ROLLING_DEFAULTS.items():
        implicit = rolling(algorithm_id, raw, factors, window=30)
        explicit = rolling(algorithm_id, raw, factors, window=30, parameters=dict(defaults))
        np.testing.assert_array_equal(implicit.to_numpy(), explicit.to_numpy())


def test_library_text_is_generated_from_signatures():
    text = library_text()
    assert "positions(algorithm_id, prices, split_factors=None, *" in text
    assert "run(algorithm_id" in text and "data_dictionary" not in text
    assert "rolling(algorithm_id, prices, split_factors=None, *, window" in text
    assert all(a in text for a in DEFAULTS) and all(a in text for a in ROLLING_DEFAULTS)


def test_prompts_without_library_carry_no_library_calls():
    task = TASKS.get("macd_b")
    raw = messages(task, seed=11, library=False)[0]["content"]
    lib = messages(task, seed=11, library=True)[0]["content"]
    assert "fin_skills.algorithms" not in raw and "Do not import fin_skills" in raw
    assert "fin_skills.algorithms" in lib and "EVENT RATIOS" in raw


def test_cards_switch_removes_only_the_rule_cards():
    task = TASKS.get("macd_b")
    for library in (False, True):
        with_cards = messages(task, seed=11, library=library)[0]["content"]
        bare = messages(task, seed=11, library=library, cards=False)[0]["content"]
        assert "Rules that apply" in with_cards and "[split_once]" in with_cards
        assert "Rules that apply" not in bare and "[split_once]" not in bare
        assert "EVENT RATIOS" in bare and "closes through row t-1 only" in bare
        assert ("fin_skills.algorithms" in bare) is library
    seen = Judge(task, seed=11, library=False).check(BUGS["no_adjustment"][0])
    assert seen.failed == ["split_representation"]
    assert "[split_once]" in seen.check_text(tools=False)
    causes = seen.check_text(tools=False, rules=False)
    assert causes.startswith("Check split_representation failed: likely")
    assert "Rule:" not in causes and "[split_once]" not in causes


def test_table_tasks_say_values_not_positions():
    text = messages(TASKS.get("drift_a"), seed=11, library=False)[0]["content"]
    assert "DataFrame of values" in text and "known at close t-1" in text
    assert "DataFrame of positions" in messages(TASKS.get("macd_b"), seed=11,
                                                library=False)[0]["content"]


SCRATCH = '''
import numpy as np, pandas as pd
def solve(prices, split_factors):
    adj = prices * split_factors.cumprod()
    out = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for c in adj.columns:
        x = adj[c]
        line = (x.ewm(span=8, adjust=False, min_periods=8).mean()
                - x.ewm(span=21, adjust=False, min_periods=21).mean())
        sig = line.ewm(span=5, adjust=False, min_periods=5).mean()
        out[c] = (np.sign(line - sig) / 4).shift(1).fillna(0.0)
    return out
'''
LIBRARY = '''
from fin_skills.algorithms import positions
def solve(prices, split_factors):
    return positions("macd", prices, split_factors, factors="ratio",
                     parameters={"fast": 8, "slow": 21, "signal_span": 5})
'''
BUGS = {   # planted defect -> the check that must reject it
    "same_session": (SCRATCH.replace(".shift(1).fillna(0.0)", ".fillna(0.0)"), "execution_lag"),
    "no_adjustment": (SCRATCH.replace("prices * split_factors.cumprod()", "prices"),
                      "split_representation"),
    "cumulated_twice": (SCRATCH.replace("split_factors.cumprod()",
                                        "split_factors.cumprod().cumprod()"),
                        "split_representation"),
    "nan_warmup": (SCRATCH.replace(".shift(1).fillna(0.0)", ".shift(1)"), "finite_output"),
}


def test_correct_programs_are_accepted_and_graded_valid():
    task = TASKS.get("macd_b")
    for code, library in ((SCRATCH, False), (LIBRARY, True)):
        judge = Judge(task, seed=11, library=library)
        seen = judge.check(code)
        assert seen.accepted and judge.grade(code)
        assert not hasattr(seen, "valid")              # the grade is not part of what is seen
    assert Judge(task, seed=11, library=True).check(LIBRARY).library_calls == 1


def _library_program(task) -> str:
    options = dict(task.library_options)
    params = {k: v for k, v in task.parameters.items() if k not in options}
    call = "rolling" if task.kind == "table" else "positions"
    return (f"from fin_skills.algorithms import {call}\n"
            "def solve(prices, split_factors):\n"
            f"    return {call}({task.algorithm!r}, prices, split_factors, "
            f"factors={task.factors!r}, parameters={params!r}, **{options!r})\n")


@pytest.mark.parametrize("task", TASKS.tasks, ids=lambda t: t.id)
def test_a_correct_program_passes_every_check_on_every_task(task):
    """Qualification 2: no check rejects a correct program (no false alarm on any task)."""
    code = _library_program(task)
    judge = Judge(task, seed=11, library=True)
    seen = judge.check(code)
    assert seen.accepted, (seen.error, seen.checks)
    assert set(seen.checks) == {"finite_output", "split_representation", "execution_lag"}
    assert judge.grade(code)


@pytest.mark.parametrize("bug", sorted(BUGS))
def test_planted_defects_fail_their_check_and_the_grade(bug):
    """Qualification 3: every kind of trap is caught by the check meant for it."""
    code, check = BUGS[bug]
    judge = Judge(TASKS.get("macd_b"), seed=11, library=False)
    seen = judge.check(code)
    assert seen.executed and not seen.accepted and seen.checks[check] is False
    assert check in seen.check_text(tools=False) and "fin_skills" not in seen.check_text(
        tools=False)
    assert judge.grade(code) is False


def test_library_is_refused_without_library_access():
    seen = Judge(TASKS.get("macd_b"), seed=11, library=False).check(LIBRARY)
    assert not seen.executed and "not available" in seen.error


def test_wrong_shape_and_crash_are_results_not_errors():
    judge = Judge(TASKS.get("macd_b"), seed=11, library=False)
    shape = judge.check("def solve(prices, split_factors):\n    return prices.iloc[1:]\n")
    crash = judge.check("def solve(prices, split_factors):\n    raise KeyError('x')\n")
    assert not shape.executed and "index and columns" in shape.error
    assert not crash.executed and crash.error.startswith("KeyError")


VOL = """
import numpy as np, pandas as pd
def solve(prices, split_factors):
    r = (prices * split_factors).pct_change()
    return (r.rolling(20).std().shift(1) * np.sqrt(252)).fillna(0.0)
"""
TABLE_BUGS = {
    "same_session": (VOL.replace(".shift(1)", ""), "execution_lag"),
    "no_adjustment": (VOL.replace("(prices * split_factors)", "prices"), "split_representation"),
    "nan_warmup": (VOL.replace(".fillna(0.0)", ""), "finite_output"),
}


def test_table_task_accepts_a_correct_program():
    judge = Judge(TASKS.get("historical_volatility_a"), seed=11, library=False)
    assert judge.check(VOL).accepted and judge.grade(VOL)


@pytest.mark.parametrize("bug", sorted(TABLE_BUGS))
def test_table_task_defects_fail_their_check(bug):
    code, check = TABLE_BUGS[bug]
    judge = Judge(TASKS.get("historical_volatility_a"), seed=11, library=False)
    seen = judge.check(code)
    assert seen.executed and seen.checks[check] is False and not judge.grade(code)
