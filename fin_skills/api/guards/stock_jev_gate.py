"""Guard: Stock-JEV structured decision wire contract, causality, and disqualification plausibility (stock-jev-meta-gate)."""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Union
import pandas as pd

from fin_skills.api.base import Guard, GuardResult, Outcome, get, register


@register
class StockJevGateGuard(Guard):
    """Audit Stock-JEV structured decisions for wire contract conformity, causality, and disqualification plausibility.

    Inputs
        decision_result : dict returned by JEV predict(state, questions).
        max_disqualification_rate : maximum tolerable fraction of disqualified tickers (default 0.35).
    """

    name = "stock_jev_gate"
    skill = "stock-jev-meta-gate"
    summary = "Audits Financial JEV decision contract conformity, probability calibration, and universe disqualification plausibility."
    wraps = (
        "fin_skills.jev.stock_jev.audit_stock_jev_causality",
        "fin_skills.jev.stock_jev.AnalyticalStockJEV.predict",
    )
    required = ("decision_result",)
    optional = ("max_disqualification_rate",)

    def check(
        self,
        decision_result: Dict[str, Any],
        max_disqualification_rate: float = 0.35,
    ) -> Outcome:
        out = Outcome()
        if not isinstance(decision_result, dict):
            out.error("decision_result must be a dictionary", where="stock_jev_gate")
            return out

        answers = decision_result.get("answers")
        if not isinstance(answers, dict) or not answers:
            out.error("decision_result missing nonempty 'answers' map", where="stock_jev_gate")
            return out

        disqualified_count = 0
        total_noul_checks = 0

        for q_id, ans in answers.items():
            if not isinstance(ans, dict):
                out.error(f"Answer for {q_id} must be an object", where="stock_jev_gate")
                continue
            kind = ans.get("type")
            if kind == "choice":
                probs = ans.get("probabilities", {})
                if not probs or not math.isclose(sum(probs.values()), 1.0, abs_tol=1e-4):
                    out.error(f"Choice question {q_id} probabilities do not sum to 1.0", where="stock_jev_gate")
                choice = ans.get("choice")
                if not choice or choice not in probs:
                    out.error(f"Choice {choice} for {q_id} is not declared in probabilities", where="stock_jev_gate")
            elif kind == "noul":
                prob = ans.get("noul")
                if prob is None or not (0.0 <= prob <= 1.0):
                    out.error(f"Noul question {q_id} probability out of bounds: {prob}", where="stock_jev_gate")
                else:
                    total_noul_checks += 1
                    if prob > 0.5:
                        disqualified_count += 1
            elif kind == "score":
                score = ans.get("score")
                probs = ans.get("probabilities", {})
                if score is None or not probs:
                    out.error(f"Score question {q_id} missing score or probabilities", where="stock_jev_gate")
                elif not math.isclose(sum(probs.values()), 1.0, abs_tol=1e-4):
                    out.error(f"Score question {q_id} probabilities do not sum to 1.0", where="stock_jev_gate")

        if total_noul_checks > 0:
            disq_ratio = disqualified_count / total_noul_checks
            out.note(
                total_noul_checks=total_noul_checks,
                disqualified_count=disqualified_count,
                disqualification_ratio=disq_ratio,
                max_allowed_ratio=max_disqualification_rate,
            )
            if disq_ratio > max_disqualification_rate:
                out.error(
                    f"Disqualification ratio {disq_ratio:.2%} exceeds threshold {max_disqualification_rate:.2%}",
                    where="stock_jev_gate",
                )
            else:
                out.info(
                    f"Universe disqualification ratio {disq_ratio:.2%} within limits (<= {max_disqualification_rate:.2%})",
                    where="stock_jev_gate",
                )
        else:
            out.info("JEV decision contract passed; 0 disqualification checks evaluated.", where="stock_jev_gate")

        return out


def check_stock_jev_gate(
    decision_result: Dict[str, Any],
    max_disqualification_rate: float = 0.35,
) -> Outcome:
    """Convenience functional wrapper for StockJevGateGuard."""
    return StockJevGateGuard().check(
        decision_result=decision_result,
        max_disqualification_rate=max_disqualification_rate,
    )
