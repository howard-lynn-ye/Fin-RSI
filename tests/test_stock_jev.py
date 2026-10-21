"""Acceptance of the selectively integrated JEV templates and strict guard."""
from __future__ import annotations

import copy
import subprocess
import sys

import pandas as pd
import pytest

from fin_skills.api.base import GuardResult
from fin_skills.api.guards.stock_jev_gate import check_stock_jev_gate
from fin_skills.jev import (
    audit_stock_jev_causality, build_disqualification_question,
    build_macro_regime_question, build_ordinal_rating_question,
)
from fin_skills.model_zoo.jev import _questions
from fin_skills.tools import call_tool


def example():
    questions = build_macro_regime_question()
    questions.update(build_disqualification_question("ABC"))
    questions.update(build_ordinal_rating_question("ABC"))
    options = list(questions["regime_gate"]["criteria"])
    # Synthetic protocol fixture, not a model prediction or measured token count.
    response = {"model": "test-fixture", "usage": {"input_tokens": 0, "output_tokens": 0},
                "answers": {
                    "regime_gate": {"type": "choice", "choice": options[0],
                                    "confidence": .4,
                                    "probabilities": dict(zip(options, [.4, .3, .2, .1]))},
                    "disqualify_ABC": {"type": "noul", "noul": .9},
                    "rating_ABC": {"type": "score", "score": 2., "confidence": .2,
                                   "probabilities": {str(i): .2 for i in range(5)},
                                   "legend": {str(i): label for i, label in enumerate(
                                       questions["rating_ABC"]["criteria"])}}}}
    return questions, response


def test_all_templates_and_standard_guard_result():
    questions, response = example()
    assert _questions(questions) == questions
    result = check_stock_jev_gate(response, questions)
    assert isinstance(result, GuardResult) and result.passed
    assert result.evidence["disqualification_ratio"] == 1.
    assert not check_stock_jev_gate(response, questions, .35).passed
    assert response == example()[1]


@pytest.mark.parametrize("defect", [
    "unknown_type", "negative_probability", "nan", "missing_answer", "extra_answer",
    "wrong_choice", "sum", "score", "legend", "model", "usage",
])
def test_malformed_model_answers_fail_closed(defect):
    questions, response = example()
    ans = response["answers"]
    choice = ans["regime_gate"]
    if defect == "unknown_type": choice["type"] = "invented"
    elif defect == "negative_probability":
        choice["probabilities"] = dict(zip(choice["probabilities"], [2., -1., 0., 0.]))
    elif defect == "nan": ans["disqualify_ABC"]["noul"] = float("nan")
    elif defect == "missing_answer": del ans["regime_gate"]
    elif defect == "extra_answer": ans["extra"] = copy.deepcopy(choice)
    elif defect == "wrong_choice": choice["choice"] = "LIQUIDITY_CONTRACTION"
    elif defect == "sum": choice["probabilities"]["BULL_MOMENTUM"] = .5
    elif defect == "score": ans["rating_ABC"]["score"] = 4.
    elif defect == "legend": del ans["rating_ABC"]["legend"]["0"]
    elif defect == "model": del response["model"]
    elif defect == "usage": del response["usage"]
    result = check_stock_jev_gate(response, questions)
    assert not result.passed and result.errors


def test_templates_do_not_share_mutable_criteria():
    build_ordinal_rating_question("ABC")["rating_ABC"]["criteria"][0] = "changed"
    assert build_ordinal_rating_question("ABC")["rating_ABC"]["criteria"][0] != "changed"


@pytest.mark.parametrize("ticker", ["", " A", None, 12])
def test_bad_ticker_rejected(ticker):
    with pytest.raises(ValueError): build_disqualification_question(ticker)


@pytest.mark.parametrize("dates", [[], [None], ["0000-bad"], ["NaT"], [123],
                                    ["2026-10-13"], ["2026-10-12T01:00Z"],
                                    ["2026-10-01", 123]])
def test_invalid_or_late_feature_dates_rejected(dates):
    with pytest.raises(ValueError):
        audit_stock_jev_causality(pd.DataFrame({"date": dates}), "2026-10-12")


def test_cutoff_uses_timezones_and_chronological_dates():
    assert audit_stock_jev_causality(
        pd.DataFrame({"date": ["2026-10-11T20:00:00-04:00", "2026-9-1"]}), "2026-10-12",
    )


def test_guard_is_available_over_json():
    questions, response = example()
    result = call_tool("check_stock_jev_gate", {"decision_result": response, "questions": questions})
    assert result["passed"]


def test_template_import_does_not_load_torch_or_make_network_calls():
    script = """
import sys, socket
def deny(*a, **k): raise AssertionError('network access')
socket.socket.connect = deny
socket.create_connection = deny
from fin_skills.jev import build_macro_regime_question
assert build_macro_regime_question()
assert 'torch' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True)
