"""Qualification of the 24 confirmatory tasks (fin_skills/tasks/confirm.py)."""
import numpy as np
import pytest

from fin_skills.algorithms import positions, routing
from fin_skills.tasks import Judge, library_positions, panel, reference, suite
from fin_skills.tasks.confirm import CONFIRM_SEEDS, DRAW_SEED, SECOND, confirm_tasks
from fin_skills.tasks.properties import checks, first_position_row

TASKS = confirm_tasks()
CATALOGUED = [t for t in TASKS if t.algorithm in routing.RULES]
OTHER = [t for t in TASKS if t.algorithm not in routing.RULES]
SEEDS = CONFIRM_SEEDS + (601, 733)

HEAD = '''import numpy as np
import pandas as pd


def adjusted(prices, split_factors):
    f = split_factors.cumprod() if RATIO else split_factors
    return prices * f


def walk(indicator, lower, upper, centre):
    out = pd.DataFrame(np.nan, index=indicator.index, columns=indicator.columns)
    for name in indicator.columns:
        state, col = 0.0, []
        for value in indicator[name]:
            if pd.isna(value):
                col.append(np.nan)
                continue
            if value < lower:
                state = 1.0
            elif value > upper:
                state = -1.0
            elif (state == 1.0 and value >= centre) or (state == -1.0 and value <= centre):
                state = 0.0
            col.append(state)
        out[name] = col
    return out


def weights(values):
    return values.shift(1).fillna(0.0) / values.shape[1]

'''
# One hand-written program per rule the library lacks, in pandas, written apart from the numpy
# references. P is the task's parameter dict; RATIO says whether the factors are event ratios.
SOLUTIONS = {
    "wilder_rsi_reversion": '''
def solve(prices, split_factors):
    p, n = adjusted(prices, split_factors), P["lookback"]
    index = pd.DataFrame(np.nan, index=p.index, columns=p.columns)
    for name in p.columns:
        d = p[name].diff().to_numpy()
        u = v = None
        for t in range(n, len(p)):
            if u is None:
                u, v = np.maximum(d[1:n + 1], 0).mean(), np.maximum(-d[1:n + 1], 0).mean()
            else:
                u, v = (u * (n - 1) + max(d[t], 0)) / n, (v * (n - 1) + max(-d[t], 0)) / n
            index.iloc[t, p.columns.get_loc(name)] = 50.0 if u + v == 0 else 100 * u / (u + v)
    return weights(walk(index, P["lower"], P["upper"], 50.0))
''',
    "band_breakout": '''
def solve(prices, split_factors):
    p, n = adjusted(prices, split_factors), P["lookback"]
    std = p.rolling(n).std(ddof=1)
    z = ((p - p.rolling(n).mean()) / std).where(std != 0, 0.0).where(std.notna())
    out = pd.DataFrame(np.nan, index=p.index, columns=p.columns)
    for name in p.columns:
        state, col = 0.0, []
        for value in z[name]:
            if pd.isna(value):
                col.append(np.nan)
                continue
            if value > P["entry_z"]:
                state = 1.0
            elif value < -P["entry_z"]:
                state = -1.0
            elif (state == 1.0 and value < P["exit_z"]) or (state == -1.0 and value > -P["exit_z"]):
                state = 0.0
            col.append(state)
        out[name] = col
    return weights(out)
''',
    "ema_crossover": '''
def solve(prices, split_factors):
    p = adjusted(prices, split_factors)
    fast = p.ewm(span=P["fast"], adjust=False).mean()
    slow = p.ewm(span=P["slow"], adjust=False).mean()
    value = np.sign(fast - slow)
    value.iloc[:P["slow"] - 1] = np.nan
    return weights(value)
''',
    "threshold_momentum": '''
def solve(prices, split_factors):
    p = adjusted(prices, split_factors)
    m = p / p.shift(P["lookback"]) - 1
    value = (m > P["threshold"]).astype(float) - (m < -P["threshold"]).astype(float)
    return weights(value.where(m.notna()))
''',
    "range_position_reversion": '''
def solve(prices, split_factors):
    p, n = adjusted(prices, split_factors), P["lookback"]
    high, low = p.rolling(n).max(), p.rolling(n).min()
    k = ((p - low) / (high - low)).where(high != low, 0.5).where(high.notna())
    return weights(walk(k, P["lower"], P["upper"], 0.5))
''',
    "ma_envelope": '''
def solve(prices, split_factors):
    p = adjusted(prices, split_factors)
    m = p.rolling(P["lookback"]).mean()
    value = (p > m * (1 + P["band"])).astype(float) - (p < m * (1 - P["band"])).astype(float)
    return weights(value.where(m.notna()))
''',
    "long_short_momentum": '''
def solve(prices, split_factors):
    p, n = adjusted(prices, split_factors), P["lookback"]
    score = (1 + p.pct_change()).rolling(n).apply(np.prod, raw=True) - 1
    value = pd.DataFrame(np.nan, index=p.index, columns=p.columns)
    for t in range(len(p)):
        s = score.iloc[t].to_numpy()
        if np.isnan(s).any():
            continue
        order = np.argsort(-s, kind="stable")
        row = np.zeros(len(s))
        if s[order[0]] > s[order[-1]]:
            row[order[0]], row[order[-1]] = 1.0, -1.0
        value.iloc[t] = row
    return weights(value)
''',
    "inverse_variance": '''
def solve(prices, split_factors):
    p = adjusted(prices, split_factors)
    inverse = 1 / p.pct_change().rolling(P["window"]).var(ddof=1)
    w = inverse.div(inverse.sum(axis=1), axis=0)
    return w.shift(1).fillna(0.0)
''',
}
# The catalogued rule each of those is a near miss of, called with the nearest parameters.
NEAR = {
    "wilder_rsi_reversion": ("rsi_reversion", lambda p: dict(p)),
    "band_breakout": ("bollinger_reversion", lambda p: dict(lookback=p["lookback"],
                                                           entry_z=p["entry_z"])),
    "ema_crossover": ("ma_crossover", lambda p: dict(p)),
    "threshold_momentum": ("momentum", lambda p: dict(lookback=p["lookback"])),
    "range_position_reversion": ("donchian_breakout", lambda p: dict(lookback=p["lookback"])),
    "ma_envelope": ("ma_crossover", lambda p: dict(fast=2, slow=p["lookback"])),
    "long_short_momentum": ("cross_sectional_momentum", lambda p: dict(lookback=p["lookback"],
                                                                       top_k=1)),
    "inverse_variance": ("inverse_volatility", lambda p: {}),
}


def program(task):
    return (f"RATIO = {task.factors == 'ratio'}\nP = {task.parameters!r}\n" + HEAD
            + SOLUTIONS[task.algorithm])


def test_the_set_is_what_the_design_says():
    assert len(TASKS) == 24 and len(CATALOGUED) == 16 and len(OTHER) == 8
    assert len({t.id for t in TASKS}) == 24 and {t.origin for t in TASKS} == {"confirm"}
    rules = sorted({t.algorithm for t in CATALOGUED})
    assert rules == sorted(set(routing.RULES) - {"equal_weight"})
    drawn = sorted(np.random.default_rng(DRAW_SEED).choice(rules, 5, replace=False).tolist())
    assert drawn == sorted(SECOND) == sorted(a for a in rules if sum(
        t.algorithm == a for t in CATALOGUED) == 2)
    assert {t.factors for t in TASKS} == {"cumulative", "ratio"} and len({t.order for t in TASKS}) == 4
    earlier = {(t.algorithm, tuple(sorted(t.parameters.items()))) for t in suite().tasks}
    for t in CATALOGUED:                    # no earlier task used these parameter values
        assert (t.algorithm, tuple(sorted(t.parameters.items()))) not in earlier, t.id
    assert not set(CONFIRM_SEEDS) & {11, 23, 37, 101, 211, 307}


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.id)
def test_the_rule_is_described_and_never_named(task):
    text = task.definition.lower()
    names = set(routing.RULES) | {t.algorithm for t in OTHER} | {
        "bollinger", "rsi", "donchian", "macd", "momentum", "wilder", "stochastic", "envelope",
        "crossover", "breakout", "reversion", "risk parity", "hierarchical risk parity"}
    import re
    assert not [name for name in names if re.search(rf"\b{re.escape(name)}\b", text)], task.id
    assert "positions(" not in text and "fin_skills" not in text


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.id)
def test_the_reference_passes_every_check_and_takes_positions(task):
    for seed in SEEDS:
        r = reference(task, seed)
        result = checks(task, r)
        assert len(result) >= 2 and all(result.values()), (seed, result)
        assert (np.abs(r) > 1e-12).mean() > 0.05
        first = first_position_row(task)
        if first is not None:
            assert not np.abs(r[:first]).any()


@pytest.mark.parametrize("task", CATALOGUED, ids=lambda t: t.id)
def test_the_library_computes_every_catalogued_task(task):
    for seed in SEEDS:
        raw, factors, _ = panel(seed, rows=task.rows, factors=task.factors, order=task.order)
        got = library_positions(task, raw, factors)
        assert np.abs(got.to_numpy() - reference(task, seed)).max() <= task.tolerance
    # and one call written from the hint, with the task's values, is graded valid
    args = "".join(f", {k}={v!r}" for k, v in {**task.parameters, **task.library_options}.items())
    code = ("from fin_skills.algorithms import positions\n"
            "def solve(prices, split_factors):\n"
            f"    return positions('{task.algorithm}', prices, split_factors{args})\n")
    judge = Judge(task, seed=CONFIRM_SEEDS[0], library=True, properties=True)
    assert judge.check(code).accepted and judge.grade(code)


@pytest.mark.parametrize("task", OTHER, ids=lambda t: t.id)
def test_a_hand_written_program_solves_every_other_task(task):
    for seed in CONFIRM_SEEDS:
        judge = Judge(task, seed=seed, library=False, properties=True)
        seen = judge.check(program(task))
        assert seen.executed, seen.error[-400:]
        assert seen.accepted, seen.failed
        assert judge.grade(program(task))


@pytest.mark.parametrize("task", OTHER, ids=lambda t: t.id)
def test_no_catalogued_rule_computes_the_other_tasks(task):
    """The near miss, called with the nearest parameters, is wrong on every seed, and so is
    its negative: the right answer to route() is none."""
    name, parameters = NEAR[task.algorithm]
    for seed in SEEDS:
        raw, factors, _ = panel(seed, rows=task.rows, factors=task.factors, order=task.order)
        options = dict(window=task.parameters["window"]) if name == "inverse_volatility" else {}
        near = positions(name, raw, factors, parameters=parameters(task.parameters) or None,
                         **options).to_numpy()
        want = reference(task, seed)
        assert np.abs(near - want).max() > 1e-3 and np.abs(-near - want).max() > 1e-3
        assert not checks(task, want) == {}      # and the task is still checked like the others
