"""fin_skills.tasks - task definitions, independent references, judge and hidden grader."""
from __future__ import annotations

import numpy as np
import pytest

from fin_skills.algorithms import positions
from fin_skills.tasks import Judge, library_positions, library_text, messages, panel, reference
from fin_skills.tasks import suite
from fin_skills.tasks.prompts import DEFAULTS
from fin_skills.tasks.spec import E4_ALGORITHMS, assign_splits

SEEDS = (11, 23, 37)
TASKS = suite()


@pytest.mark.parametrize("task", TASKS.tasks, ids=lambda t: t.id)
def test_reference_matches_library_on_every_seed(task):
    """Qualification 1: the independent equations and the library compute the same rule."""
    for seed in SEEDS:
        raw, factors, _ = panel(seed, rows=task.rows, factors=task.factors, order=task.order)
        lib = library_positions(task, raw, factors).to_numpy()
        np.testing.assert_allclose(lib, reference(task, seed), atol=1e-12, rtol=0)


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
    assert all(TASKS.splits[a] == "dev" for a in E4_ALGORITHMS)
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


def test_library_text_is_generated_from_signatures():
    text = library_text()
    assert "positions(algorithm_id, prices, split_factors=None, *" in text
    assert "run(algorithm_id" in text and "data_dictionary" not in text
    assert all(a in text for a in DEFAULTS)


def test_prompts_without_library_carry_no_library_calls():
    task = TASKS.get("macd_b")
    raw = messages(task, seed=11, library=False)[0]["content"]
    lib = messages(task, seed=11, library=True)[0]["content"]
    assert "fin_skills.algorithms" not in raw and "Do not import fin_skills" in raw
    assert "fin_skills.algorithms" in lib and "EVENT RATIOS" in raw


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
