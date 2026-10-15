"""Optional, offline-by-default agent playbook improvement.

No trading runtime imports this package automatically. Enable explicitly in a new
development run and supply a caller-owned proposer and independent evaluator.
"""
from .core import (
    CaseResult, Decision, Entry, Evaluation, JSONProposer, Playbook, Policy,
    ProposalContext, RSILoop, Trace,
)

__all__ = ["CaseResult", "Decision", "Entry", "Evaluation", "JSONProposer", "Playbook",
           "Policy", "ProposalContext", "RSILoop", "Trace"]
