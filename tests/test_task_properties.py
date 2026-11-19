"""Declared-property checks (tasks/properties.py) and the rule router (algorithms/routing.py)."""
import numpy as np
import pytest

from fin_skills.algorithms import routing
from fin_skills.tasks import Judge, frozen_suite, library_positions, panel, reference, suite
from fin_skills.tasks.properties import CAUSES, CHECKS, checks, first_position_row

POSITION_TASKS = [t for t in suite().tasks if t.kind == "positions"]
SEEDS = (11, 101, 211, 307, 1, 2, 3)


@pytest.mark.parametrize("task", POSITION_TASKS, ids=lambda t: t.id)
def test_every_reference_passes_every_property_check(task):
    """A program that matches the reference can never be rejected by a property check."""
    for seed in SEEDS:
        result = checks(task, reference(task, seed))
        assert result and all(result.values()), (seed, result)
        assert set(result) <= set(CHECKS)


@pytest.mark.parametrize("task", [t for t in POSITION_TASKS if first_position_row(t) is not None],
                         ids=lambda t: t.id)
def test_declared_warmup_is_the_librarys(task):
    for seed in SEEDS[:3]:
        raw, factors, _ = panel(seed, rows=task.rows, factors=task.factors, order=task.order)
        assert library_positions(task, raw, factors).attrs["warmup_rows"] == first_position_row(task)


def test_property_checks_reject_what_the_old_checks_accepted():
    task = next(t for t in frozen_suite().heldout if t.id == "bollinger_reversion_a")
    good = reference(task, 101)
    n = good.shape[1]
    early = good.copy()
    early[3, 0] = 1 / n
    cases = {"takes_positions": good * 0, "position_scale": good / n, "warmup_rows_zero": early}
    for name, values in cases.items():
        result = checks(task, values)
        assert not result[name] and sum(not ok for ok in result.values()) == 1, (name, result)
    assert not checks(task, good * n)["position_scale"]            # not scaled at all
    table = next(t for t in suite().tasks if t.kind == "table")
    assert checks(table, np.zeros((4, 2))) == {}


def test_the_default_judge_keeps_the_three_frozen_checks():
    task = frozen_suite().heldout[0]
    code = ("from fin_skills.algorithms import positions\n"
            "def solve(prices, split_factors):\n"
            f"    return positions('{task.algorithm}', prices, split_factors) / 4\n")
    old = Judge(task, seed=101, library=True).check(code)
    assert list(old.checks) == ["finite_output", "split_representation", "execution_lag"]
    assert old.accepted
    new = Judge(task, seed=101, library=True, properties=True).check(code)
    assert new.failed == ["position_scale"] and not new.accepted
    assert set(CAUSES) == set(CHECKS)
    text = new.check_text(tools=False)             # Explain feedback names the cause
    assert text == f"Check position_scale failed: likely {CAUSES['position_scale']}."
    assert old.check_text(tools=False) == ""       # accepted by the three frozen checks


def test_explain_text_joins_frozen_and_declared_causes():
    from fin_skills.tasks.judge import Visible
    seen = Visible(code="", executed=True, checks=dict(finite_output=True, execution_lag=False,
                                                       warmup_rows_zero=False))
    text = seen.check_text(tools=False)
    assert text.startswith("Check execution_lag failed") and text.endswith(
        f"Check warmup_rows_zero failed: likely {CAUSES['warmup_rows_zero']}.")


def test_hint_names_the_call_and_its_parameters():
    assert routing.call_text("rsi_reversion") == (
        'positions("rsi_reversion", prices, split_factors, lookback=..., lower=..., upper=...)')
    assert routing.call_text("inverse_volatility").endswith("window=..., rebalance_every=...)")
    assert set(routing.PARAMETERS) == set(routing.RULES)
    for name in routing.PARAMETERS:
        assert routing.call_text(name) in routing.hint(name)
    from fin_skills.algorithms.technical import SIGNAL_DEFAULTS
    for name, defaults in SIGNAL_DEFAULTS.items():      # the hint lists what the rule accepts
        if name in routing.PARAMETERS:
            assert set(routing.PARAMETERS[name]) <= set(defaults), name
    with pytest.raises(KeyError, match="positions\\(\\) accepts"):
        routing.call_text("rsi")
    assert routing.POINTER == 'The fin_skills library implements this rule as positions("{algorithm}", ...).'


@pytest.mark.parametrize("reply, want", [
    ("rsi_reversion", "rsi_reversion"), ("**rsi_reversion**", "rsi_reversion"),
    ("The rule is an RSI rule.\n\nrsi_reversion", "rsi_reversion"),
    ("<think>macd or rsi_reversion?</think>\nbollinger_reversion", "bollinger_reversion"),
    ("vol_target_momentum", "vol_target_momentum"), ("NONE", None), ("none of them", None),
    ("close to rsi_reversion, but it uses Wilder smoothing.\nNONE", None),
    ("momentum or macd", None), ("I think donchian_breakout fits.\nThat is my answer.",
                                 "donchian_breakout"), ("", None), ("<think>unfinished", None)])
def test_parse_route(reply, want):
    assert routing.parse_route(reply) == want


def test_route_asks_once_and_returns_what_the_model_named():
    asked = []

    def generate(messages):
        asked.append(messages)
        return "donchian_breakout"
    assert routing.route("highest and lowest close of the last 20 rows", generate) == "donchian_breakout"
    assert len(asked) == 1 and asked[0] == routing.route_messages(
        "highest and lowest close of the last 20 rows")
    listing = asked[0][1]["content"]
    assert all(f"{name}: {rule}" in listing for name, rule in routing.RULES.items())
    with pytest.raises(TypeError):
        routing.route_messages("  ")
