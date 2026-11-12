"""Optional, offline-by-default agent playbook improvement, and a benchmark of RSI methods.

No trading runtime imports this package automatically. Enable explicitly in a new
development run and supply a caller-owned proposer and independent evaluator, or use
loop_from_checks() to drive the loop from property-check results (see wiring.py).

The RSI methods and the benchmark (methods.py, meter.py, bench.py) share one interface:

    from fin_skills import rsi
    from fin_skills.tasks import suite
    m = rsi.Reflexion(library=True).fit(generate, suite().dev)
    episode = m.solve(generate, suite().heldout[0], seed=11, budget=rsi.Budget())
    report = rsi.benchmark(rsi.METHODS, generate, suite())
    ablation = rsi.benchmark([rsi.CheckFeedback], generate, suite(), arms=tuple(rsi.ARMS))

They are loaded on first use, so `import fin_skills.rsi` stays free of numpy and pandas.
"""
from .core import (
    CaseResult, Decision, Entry, Evaluation, JSONProposer, Playbook, Policy,
    ProposalContext, RSILoop, Trace,
)
from .meter import Budget, BudgetExceeded, FitBudget, Meter
from .wiring import Attempt, check_evaluator, loop_from_checks, observe

_METHODS = ("ARMS", "BaseRSI", "CheckFeedback", "Episode", "Evolve", "METHODS", "NoImprove",
            "PlaybookLearning", "Reflexion", "RepeatedSampling", "SelfDebug", "Step", "Team",
            "arm_name", "extract")
_BENCH = ("Report", "benchmark")

__all__ = ["Attempt", "Budget", "BudgetExceeded", "CaseResult", "Decision", "Entry",
           "Evaluation", "FitBudget", "JSONProposer", "Meter", "Playbook", "Policy",
           "ProposalContext", "RSILoop", "Trace", "check_evaluator", "loop_from_checks",
           "observe", *_METHODS, *_BENCH]


def __getattr__(name):
    if name in _METHODS:
        from . import methods
        return getattr(methods, name)
    if name in _BENCH:
        from . import bench
        return getattr(bench, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
