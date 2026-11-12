"""Scalar execution shares the production boundary; no model or external answer is used."""
import pytest

from benchmarks.agent_study.linux_sandbox import confinement_available
from fin_skills.tasks.execution import evaluate_scalar


@pytest.fixture(autouse=True)
def confined(monkeypatch):
    if not confinement_available():
        pytest.skip("requires the production Linux confinement capabilities")
    monkeypatch.setenv("FIN_TASKS_REQUIRE_CONFINEMENT", "1")
    monkeypatch.setenv("FIN_TASKS_CONFINE", "benchmarks.agent_study.linux_sandbox:confine")


@pytest.mark.parametrize("library", [False, True])
def test_scalar_executes_without_position_assumptions_and_no_answer_material(library):
    code = '''
import pathlib, json, sys
packet = json.loads(pathlib.Path(sys.argv[1], 'input.json').read_text())
assert not any(key in packet for key in ('answer', 'reference', 'question', 'prices', 'factors'))
assert not any(n.startswith(('fin_skills.tasks', 'fin_skills.rsi')) for n in sys.modules)
part, total = 20, 80
100 * part / total
'''
    seen = evaluate_scalar(code, library=library)
    assert seen["executed"] and seen["value"] == 25
    assert seen["confinement"]["landlock_abi"] >= 3


@pytest.mark.parametrize("code", ["True", "[25]", "'25'", "float('nan')", "x = 25"])
def test_wrong_scalar_contract_is_candidate_failure(code):
    seen = evaluate_scalar(code)
    assert not seen["executed"] and not seen.get("infrastructure_error")


def test_scalar_cannot_load_library_through_importlib_when_disabled():
    seen = evaluate_scalar("import importlib\nimportlib.import_module('fin_skills.algorithms')\n1")
    assert not seen["executed"] and not seen.get("infrastructure_error")


@pytest.mark.parametrize("value", [2**53 + 1, -(2**53 + 1), 10**400])
def test_exact_large_integer_survives_transport_and_scoring(value):
    from benchmarks.agent_study.external_packets import ScalarPolicy
    seen = evaluate_scalar(str(value))
    assert seen["executed"] and type(seen["value"]) is int and seen["value"] == value
    policy = ScalarPolicy("integer units", 0, 0, "synthetic exact integer test")
    assert policy.correct(seen["value"], str(value))


def test_numpy_integer_survives_without_float_conversion():
    seen = evaluate_scalar("import numpy as np\nnp.int64(9007199254740993)")
    assert seen["executed"] and seen["value"] == 9007199254740993
