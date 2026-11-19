"""What every method sends for a task: one system text, one task text.

The library interface is generated from the functions' own signatures and docstrings plus the
algorithms' defaults, so it cannot drift from the code (v2 described run()'s second argument as
data_dictionary while the function names it data). tests/test_tasks_suite.py checks that every
listed default is what the algorithm actually uses.

Two switches give the skills ablation: library (the fin_skills interface, and permission to
import it) and cards (the task's rule cards). Both off is a model with only the task text.
"""
from __future__ import annotations

import hashlib
import inspect

from fin_skills.api.rules import prompt_block, rules_for
from .data import panel, sample_text
from .spec import FACTOR_TEXT, Task

from fin_skills.algorithms.routing import PARAMETERS as DEFAULTS  # noqa: E402  (listing)
# Ready algorithms usable with rolling(), and the defaults they use when a parameter is omitted.
ROLLING_DEFAULTS = {
    "historical_volatility": {"periods_per_year": 1},
    "ewma_volatility": {"periods_per_year": 1, "decay": 0.94},
    "historical_var_es": {"confidence": 0.95},
    "normal_var_es": {"confidence": 0.95},
    "drift": {"horizon": 1},
    "mean": {"horizon": 1},
}

SYSTEM = (
    "Write one Python program that defines solve(prices, split_factors). Both arguments are "
    "date-indexed pandas DataFrames of the same shape, one column per asset. Return a finite "
    "float DataFrame of {what} with exactly the same index and columns, in the same order. "
    "prices are RAW quotes. {factors} {timing} Implement the rule exactly, including "
    "warm-up rows and timing. Use numpy, pandas and scipy. No files, network, subprocesses, "
    "printing or hard-coded data: solve() will be called on other data too. Reply with the "
    "complete program in one python fence.")
WHAT = {"positions": "positions", "table": "values"}
TIMING = {
    "positions": ("A position on row t earns the return from row t-1 to row t, so it may use "
                  "closes through row t-1 only."),
    "table": ("The value on row t is what is known at close t-1, so it may use closes through "
              "row t-1 only."),
}
NO_LIBRARY = " Do not import fin_skills; it is not available for this task."
WITH_LIBRARY = (" You may use the fin_skills library described below; a correct library call "
                "is accepted exactly like hand-written code.")


def _doc(fn) -> str:
    return inspect.cleandoc(fn.__doc__ or "")


def _signature(fn) -> str:
    """name(params) without type annotations, which are noise for a model."""
    sig = inspect.signature(fn)
    plain = sig.replace(parameters=[q.replace(annotation=inspect.Parameter.empty)
                                    for q in sig.parameters.values()],
                        return_annotation=inspect.Signature.empty)
    return f"{fn.__name__}{plain}"


FROZEN_INTERFACE_SHA256 = "f0940c646523ca388a970de725e9088510d33e4248506d2ce2eeba08e4058a4f"


from fin_skills.algorithms.routing import RULES  # noqa: E402  (shown by documented=True)


def library_text(*, rolling: bool = True, frozen: bool | None = None,
                 documented: bool = False) -> str:
    """The interface shown to the model.

    rolling=False returns, byte for byte, the interface the frozen RSI study and P-adopt showed
    (position tasks only), kept in frozen_interface.py and checked by hash, so later changes
    to the library's docstrings cannot change what reproducing those studies sends.
    rolling=False, frozen=False is the current library's interface for position tasks only
    (no rolling()), built from the current docstrings: what P-fix shows.
    documented=True adds the rule each position algorithm computes (P-fix arm F3); the text
    without it is unchanged.
    """
    if documented and (rolling or frozen):
        raise ValueError("documented=True is the current position-task interface only")
    if frozen is None:
        frozen = not rolling
    if frozen and rolling:
        raise ValueError("the frozen interface has no rolling(); use rolling=False")
    if frozen:
        from .frozen_interface import FROZEN_INTERFACE
        if hashlib.sha256(FROZEN_INTERFACE.encode()).hexdigest() != FROZEN_INTERFACE_SHA256:
            raise RuntimeError("frozen_interface.py differs from the interface the studies showed")
        return FROZEN_INTERFACE
    from fin_skills.algorithms import adjust_prices, catalog, positions, signal_positions
    from fin_skills.algorithms import rolling as rolling_fn
    from fin_skills.algorithms import run
    ready = {c["id"]: c for c in catalog() if c["status"] == "ready"}
    lines = ["from fin_skills.algorithms import positions, signal_positions, "
             + ("rolling, " if rolling else "") + "adjust_prices, run",
             "", _signature(positions), _doc(positions),
             "", _signature(signal_positions), _doc(signal_positions)]
    if rolling:
        lines += ["", _signature(rolling_fn), _doc(rolling_fn)]
    lines += ["", _signature(adjust_prices), _doc(adjust_prices),
             "", "run" + _signature(run)[len(run.__name__):],
             "Run one algorithm once on its inputs: a dict keyed by input name, or the input "
             "itself when the algorithm has one; returns one allocation (portfolio) or one "
             "already-lagged exposure series for one asset (signal). For a date x asset table "
             "use positions().",
             "", "Algorithms usable with positions() (id: task, inputs; defaults):"]
    if documented:
        lines[-1] = ("Algorithms usable with positions() (id: task, inputs; defaults), each with "
                     "the rule it computes at a close; positions() then applies lag, warm-up and "
                     "scale. If a task's rule is one of these, one positions() call with the "
                     "task's parameters and factors= convention computes it, so there is no need "
                     "to write the rule yourself:")
    for algorithm_id, defaults in DEFAULTS.items():
        card = ready[algorithm_id]
        shown = ", ".join(f"{k}={v!r}" for k, v in defaults.items()) or "no parameters"
        lines.append(f"  {algorithm_id}: {card['task']}, {' '.join(card['inputs'])}; {shown}")
        if documented:
            lines.append(f"      {RULES[algorithm_id]}")
    if not rolling:
        return "\n".join(lines)
    lines += ["", "Algorithms usable with rolling() (id: task, inputs; defaults):"]
    for algorithm_id, defaults in ROLLING_DEFAULTS.items():
        card = ready[algorithm_id]
        shown = ", ".join(f"{k}={v!r}" for k, v in defaults.items())
        lines.append(f"  {algorithm_id}: {card['task']}, {' '.join(card['inputs'])}; {shown}")
    return "\n".join(lines)


def system_prompt(task: Task, *, library: bool, cards: bool = True) -> str:
    text = SYSTEM.format(what=WHAT[task.kind], factors=FACTOR_TEXT[task.factors],
                         timing=TIMING[task.kind])
    text += WITH_LIBRARY if library else NO_LIBRARY
    cards = prompt_block(rules_for(task.traits), tools=library) if cards else ""
    parts = [text]
    if library:
        parts.append("fin_skills interface:\n" + library_text())
    if cards:
        parts.append(cards)
    return "\n\n".join(parts)


def task_prompt(task: Task, *, seed: int) -> str:
    raw, factors, _ = panel(seed, rows=task.rows, factors=task.factors, order=task.order)
    return (f"Rule:\n{task.definition}\n\nThe data has {len(raw)} rows and columns "
            f"{list(raw.columns)}.\n\n" + sample_text(raw, factors))


def messages(task: Task, *, seed: int, library: bool, cards: bool = True,
             extra_system: str = "") -> list[dict]:
    system = system_prompt(task, library=library, cards=cards)
    if extra_system:
        system += "\n\n" + extra_system
    return [dict(role="system", content=system),
            dict(role="user", content=task_prompt(task, seed=seed))]


def library_positions(task: Task, raw, factors):
    """The library call that computes the task's rule (qualification only)."""
    from fin_skills.algorithms import positions, rolling
    options = dict(task.library_options)
    params = {k: v for k, v in task.parameters.items() if k not in options}
    call = rolling if task.kind == "table" else positions
    return call(task.algorithm, raw, factors, factors=task.factors, parameters=params or None,
                **options)
