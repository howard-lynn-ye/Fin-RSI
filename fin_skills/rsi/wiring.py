"""Default wiring: drive the RSI loop from property-check results.

RSILoop (core.py) needs a proposer and a trusted evaluator. Writing the evaluator was left to
every host, so the loop was never connected to anything. This module supplies the common case:
a fixed set of tasks, and a host function that runs one attempt with the current playbook in
the prompt and reports whether the program ran, which checks passed, and whether a hidden
grader accepted it. Every guarantee of core.py is kept - the proposer sees development traces
only, receipts carry evidence hashes, validation tasks are disjoint from development tasks -
and the hidden grader's verdict is used as a score, never shown to the proposer.

    from fin_skills.rsi.wiring import Attempt, check_evaluator, observe, loop_from_checks
    loop = loop_from_checks("runs/rsi", generate, run_attempt,
                            {"development": dev_ids, "validation": val_ids},
                            model_id="qwen-coder-14b@<rev>", protocol_id="codegen-v2",
                            available_at="2026-11-01T00:00:00Z")
    observe(loop, attempts, available_at=..., source_version="<commit>")
    decision = loop.improve(as_of=...)
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from .core import CaseResult, Evaluation, JSONProposer, Playbook, Policy, RSILoop, Trace

SCORES = ("valid", "checks")


@dataclass(frozen=True)
class Attempt:
    """What the host reports for one attempt at one task."""

    task_id: str
    code: str
    executed: bool
    error: str = ""
    checks: Mapping[str, bool] = field(default_factory=dict)
    valid: bool = False

    def __post_init__(self):
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("task_id must be a nonempty string")
        if type(self.executed) is not bool or type(self.valid) is not bool:
            raise ValueError("executed and valid must be booleans")
        if not all(isinstance(k, str) and type(v) is bool for k, v in dict(self.checks).items()):
            raise ValueError("checks must map check names to booleans")
        if self.valid and not (self.executed and all(dict(self.checks).values())):
            raise ValueError("a valid attempt must have executed and passed every check")

    @property
    def evidence(self) -> dict:
        return dict(asdict(self), checks=dict(self.checks))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.evidence, sort_keys=True).encode()).hexdigest()


def traces(attempt: Attempt, *, available_at: str, source_version: str, tools: bool = True,
           rules: bool = True) -> list[Trace]:
    """Development traces for one attempt. Never includes the hidden grader's verdict.

    tools and rules are passed to rules.explain(): a run without the library must record
    tools=False, or the proposer reads library calls the run may not make."""
    from fin_skills.api.rules import CAUSES, explain

    base = dict(task_id=attempt.task_id, available_at=available_at,
                source_version=source_version, request=attempt.code[:8000])
    if not attempt.executed:
        return [Trace(trace_id=f"{attempt.task_id}:{attempt.sha256[:12]}", kind="tool_contract",
                      summary=f"The program did not run: {attempt.error[:600]}",
                      result=attempt.error[:4000], **base)]
    failed = [name for name, ok in attempt.checks.items() if not ok]
    if not failed:
        return [Trace(trace_id=f"{attempt.task_id}:{attempt.sha256[:12]}", kind="success",
                      summary="The program ran and every check passed.",
                      result=json.dumps(dict(attempt.checks), sort_keys=True), **base)]
    known = [name for name in failed if name in CAUSES]
    other = [name for name in failed if name not in CAUSES]
    summary = explain(known, tools=tools, rules=rules) if known else ""
    if other:
        summary = (summary + "\n" if summary else "") + "Other failed checks: " + ", ".join(other)
    return [Trace(trace_id=f"{attempt.task_id}:{attempt.sha256[:12]}", kind="analysis",
                  summary=summary[:4000], result=json.dumps(dict(attempt.checks), sort_keys=True),
                  **base)]


def observe(loop: RSILoop, attempts: Iterable[Attempt], *, available_at: str,
            source_version: str, tools: bool = True, rules: bool = True) -> int:
    """Record development attempts as traces. Returns the number of traces recorded."""
    n = 0
    for attempt in attempts:
        for trace in traces(attempt, available_at=available_at, source_version=source_version,
                            tools=tools, rules=rules):
            loop.observe(trace)
            n += 1
    return n


def check_evaluator(run_attempt: Callable[[str, str], Attempt],
                    splits: Mapping[str, Sequence[str]], *, protocol_id: str,
                    score: str = "valid", evidence_dir: str | Path | None = None):
    """An evaluate(playbook, split, as_of) callback for RSILoop.

    run_attempt(task_id, playbook_text) runs one attempt with the playbook in the prompt and
    returns an Attempt. score="valid" scores 1.0 for a hidden-grader pass and 0.0 otherwise;
    score="checks" scores the share of checks passed. With evidence_dir, every attempt record
    is written there under its SHA256, so each receipt hash names a file that exists.
    """
    if score not in SCORES:
        raise ValueError(f"score must be one of {SCORES}")
    if set(splits) != {"development", "validation"}:
        raise ValueError("splits must name exactly development and validation tasks")
    dev, val = (tuple(splits[k]) for k in ("development", "validation"))
    if not dev or not val or set(dev) & set(val):
        raise ValueError("development and validation tasks must be nonempty and disjoint")
    store = Path(evidence_dir) if evidence_dir is not None else None

    def evaluate(playbook, split, as_of):
        cases = []
        for task_id in (dev if split == "development" else val):
            a = run_attempt(task_id, playbook.render())
            if not isinstance(a, Attempt) or a.task_id != task_id:
                raise TypeError("run_attempt must return an Attempt for the requested task")
            if store is not None:
                store.mkdir(parents=True, exist_ok=True)
                path = store / f"{a.sha256}.json"
                if not path.exists():
                    path.write_text(json.dumps(a.evidence, sort_keys=True))
            checks = dict(a.checks)
            if score == "valid":
                s = 1.0 if a.valid else 0.0
            else:
                s = (sum(checks.values()) / len(checks)) if (a.executed and checks) else 0.0
            cases.append(CaseResult(case_id=task_id, score=float(s), available_at=as_of,
                                    evidence_sha256=a.sha256,
                                    execution_errors=0 if a.executed else 1,
                                    guards_passed=bool(a.executed and all(checks.values()))))
        return Evaluation(version=playbook.version, split=split, protocol_id=protocol_id,
                          cases=tuple(cases))

    return evaluate


def loop_from_checks(directory: str | Path, generate: Callable[[str], str],
                     run_attempt: Callable[[str, str], Attempt],
                     splits: Mapping[str, Sequence[str]], *, model_id: str, protocol_id: str,
                     available_at: str, policy: Policy | None = None, score: str = "valid",
                     evidence_dir: str | Path | None = None,
                     seed: Playbook = Playbook()) -> RSILoop:
    """An enabled RSILoop whose evaluator is check_evaluator().

    The default policy uses guard_rule="no_regression": a candidate may not make any task
    that passed every check fail one, but tasks that already failed do not veto it. The
    core default ("all_pass") vetoes any candidate while a single task still fails, which
    blocks learning whenever the model fails most tasks.
    """
    policy = policy or Policy(enabled=True, guard_rule="no_regression")
    evaluator = check_evaluator(run_attempt, splits, protocol_id=protocol_id, score=score,
                                evidence_dir=evidence_dir)
    return RSILoop(directory, model_id=model_id, protocol_id=protocol_id,
                   available_at=available_at, seed=seed, policy=policy,
                   proposer=JSONProposer(generate), evaluator=evaluator)
