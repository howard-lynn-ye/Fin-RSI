"""Optional, offline-by-default agent playbook improvement.

No trading runtime imports this package automatically. Enable explicitly in a new
development run and supply a caller-owned proposer and independent evaluator, or use
loop_from_checks() to drive the loop from property-check results (see wiring.py).
"""
from .core import (
    CaseResult, Decision, Entry, Evaluation, JSONProposer, Playbook, Policy,
    ProposalContext, RSILoop, Trace,
)
from .wiring import Attempt, check_evaluator, loop_from_checks, observe

__all__ = ["Attempt", "CaseResult", "Decision", "Entry", "Evaluation", "JSONProposer",
           "Playbook", "Policy", "ProposalContext", "RSILoop", "Trace", "check_evaluator",
           "loop_from_checks", "observe"]
