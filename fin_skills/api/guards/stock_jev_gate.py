"""Validate financial JEV answers using the existing master wire validator."""
from __future__ import annotations

import math

from fin_skills.api.base import Guard, GuardResult, Outcome, get, register
from fin_skills.jev.stock_jev import validate_stock_jev_decision
from fin_skills.model_zoo.jev import JevError


@register
class StockJevGateGuard(Guard):
    """Validate actual questions and answers, with an optional caller-selected limit.

    Inputs
        decision_result : model response including answers, model and usage.
        questions : original JEV questions defining IDs, types and criteria.
        max_disqualification_rate : optional limit on the fraction with noul > 0.5.
    """
    name = "stock_jev_gate"
    skill = "stock-jev-meta-gate"
    summary = "Checks JEV answer structure and probabilities against the original questions."
    wraps = ("fin_skills.jev.stock_jev.validate_stock_jev_decision",)
    required = ("decision_result", "questions")
    optional = ("max_disqualification_rate",)

    def check(self, decision_result: dict, questions: dict,
              max_disqualification_rate: float | None = None) -> Outcome:
        if max_disqualification_rate is not None and (
            type(max_disqualification_rate) not in (int, float)
            or not math.isfinite(max_disqualification_rate)
            or not 0 <= max_disqualification_rate <= 1
        ):
            raise ValueError("max_disqualification_rate must be finite and in [0, 1]")
        out = Outcome()
        try:
            checked = validate_stock_jev_decision(decision_result, questions)
        except JevError as exc:
            out.error(str(exc), where=self.name)
            return out
        values = [a["noul"] for a in checked["answers"].values() if a["type"] == "noul"]
        count = sum(p > 0.5 for p in values)
        ratio = count / len(values) if values else None
        out.note(n_answers=len(checked["answers"]), n_disqualification_checks=len(values),
                 disqualified_count=count, disqualification_ratio=ratio,
                 max_disqualification_rate=max_disqualification_rate)
        if ratio is not None and max_disqualification_rate is not None and (
            ratio > max_disqualification_rate
        ):
            out.error("Disqualification rate exceeds the caller's limit", where=self.name)
        else:
            out.info("JEV answer contract passed; predictive quality is not evaluated",
                     where=self.name)
        return out


def check_stock_jev_gate(decision_result: dict, questions: dict,
                         max_disqualification_rate: float | None = None) -> GuardResult:
    """Return the standard GuardResult, matching every other direct guard wrapper."""
    return get("stock_jev_gate").run(
        decision_result=decision_result, questions=questions,
        max_disqualification_rate=max_disqualification_rate,
    )
