"""One user model, several roles. Code checks decide; no role does.

The user connects one text generator, generate(messages) -> str, where messages is a list of
{"role": "system" | "user" | "assistant", "content": str}. A Team turns that one model into
roles that each see only what their job needs. Our audits found that more rules in one prompt
meant more mistakes; here the writer holds only the task's rule cards, and each reviewer holds
only its own rule family.

    from fin_skills.agents import Team, check_solve
    team = Team(generate)                                   # the user's own model
    result = team.solve(task_text, run_checks, traits={"positions", "split_factors"})
    result.accepted, result.code, result.transcript

Roles
  writer      writes the program from the task, the interface text and the task's cards.
  reviewers   one per rule family (timing, corporate_actions, data_hygiene) plus interface
              for programs that do not run. A reviewer is asked only when a check of its
              family failed. It sees the program, the failed checks, their causes and its
              own family's cards - never other families' rules and never expected output -
              and answers with a diagnosis and a fix direction, not code.
  judge       code, not a model: run_checks(code) -> fin_skills.rsi.wiring.Attempt. A program
              is accepted only when it ran and every check passed.
  recorder    optional RSILoop; every attempt is observed as a development trace.

mode="single" skips the reviewers and sends the judge's findings with rules.explain(), so the
two set-ups can be compared with the same model, judge and attempt budget.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from fin_skills.api.rules import CARDS, CAUSES, explain, prompt_block, rules_for
from fin_skills.rsi.fences import extract
from fin_skills.rsi.wiring import Attempt, observe

MODES = ("roles", "single")
FAMILIES = tuple(dict.fromkeys(c.family for c in CARDS))
CHECK_FAMILY = {check: c.family for c in CARDS for check in c.checks}
assert set(CHECK_FAMILY) == set(CAUSES), "every explained check needs a family"
MAX_NOTE = 1200

WRITER = ("You write one Python program for a financial research task. Follow the rules "
          "below; a controller runs checks on your program and returns its findings. Reply "
          "with the complete program in one python fence.")
REVIEWER = ("You review one failed program for one family of rules: {family}. Only the rules "
            "listed here are yours. Say in at most five sentences what in the program breaks "
            "them and how to change it. Do not write code. Do not guess numbers the program "
            "should output.")
INTERFACE_REVIEWER = ("You review one program that failed to run. Say in at most five sentences "
                      "what caused the error and how to change the program. Do not write code.")


@dataclass(frozen=True)
class Turn:
    role: str            # writer | reviewer:<family> | judge
    attempt: int
    content: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(f"{self.role}\n{self.attempt}\n{self.content}".encode()).hexdigest()


@dataclass(frozen=True)
class Result:
    task_id: str
    mode: str
    accepted: bool
    code: str
    attempts: tuple[Attempt, ...]
    transcript: tuple[Turn, ...] = field(default=())

    @property
    def model_calls(self) -> int:
        return sum(t.role != "judge" for t in self.transcript)


class Team:
    """Roles built from one generator. The judge is the caller's run_checks."""

    def __init__(self, generate: Callable[[list[dict]], str], *, mode: str = "roles",
                 max_attempts: int = 3, interface: str = "", writer_rules: str = "task",
                 recorder=None, available_at: str = "", source_version: str = "",
                 tools: bool = True):
        if not callable(generate):
            raise TypeError("generate must be callable: generate(messages) -> str")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if type(max_attempts) is not int or max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        if writer_rules not in ("task", "none"):
            raise ValueError('writer_rules must be "task" or "none"')
        if recorder is not None and not (available_at and source_version):
            raise ValueError("a recorder needs available_at and source_version")
        self.generate, self.mode, self.max_attempts = generate, mode, max_attempts
        self.interface, self.writer_rules = interface, writer_rules
        self.recorder, self.available_at, self.source_version = (recorder, available_at,
                                                                  source_version)
        # tools=False renders rule cards and explanations without library calls, for a run in
        # which the program must not use fin_skills.
        self.tools = bool(tools)

    # roles ---------------------------------------------------------------------------
    def writer_messages(self, task: str, traits: Iterable[str]) -> list[dict]:
        parts = [WRITER]
        if self.interface:
            parts.append("Interface:\n" + self.interface)
        if self.writer_rules == "task":
            block = prompt_block(rules_for(traits), tools=self.tools)
            if block:
                parts.append(block)
        return [dict(role="system", content="\n\n".join(parts)), dict(role="user", content=task)]

    def reviewer_messages(self, family: str, code: str, failed: Sequence[str],
                          traits: Iterable[str]) -> list[dict]:
        mine = [name for name in failed if CHECK_FAMILY.get(name) == family]
        cards = [c for c in CARDS if c.family == family
                 and (c.applies & set(traits) or set(c.checks) & set(mine))]
        system = REVIEWER.format(family=family) + "\n\n" + prompt_block(cards, tools=self.tools)
        user = ("Failed checks:\n" + explain(mine, tools=self.tools)
                + "\n\nProgram:\n```python\n" + code
                + "```")
        return [dict(role="system", content=system), dict(role="user", content=user)]

    def interface_messages(self, code: str, error: str) -> list[dict]:
        system = INTERFACE_REVIEWER + (("\n\nInterface:\n" + self.interface)
                                       if self.interface else "")
        user = "Error:\n" + error[-1500:] + "\n\nProgram:\n```python\n" + code + "```"
        return [dict(role="system", content=system), dict(role="user", content=user)]

    # judge findings -------------------------------------------------------------------
    @staticmethod
    def findings(attempt: Attempt, number: int, total: int) -> str:
        head = f"Controller findings for attempt {number} of {total}:"
        if not attempt.executed:
            return head + "\nThe program did not run: " + attempt.error[-900:]
        failed = [name for name, ok in attempt.checks.items() if not ok]
        return head + "\nFailed checks: " + ", ".join(failed)

    def _ask(self, messages: list[dict]) -> str:
        text = self.generate([dict(m) for m in messages])
        if not isinstance(text, str):
            raise TypeError("generate must return a string")
        return text

    def solve(self, task: str, run_checks: Callable[[str], Attempt], *,
              traits: Iterable[str] = (), task_id: str = "task") -> Result:
        traits = set(traits)
        rules_for(traits)                                   # validates the traits
        history = self.writer_messages(task, traits)
        transcript, attempts = [], []
        code = ""
        for number in range(1, self.max_attempts + 1):
            text = self._ask(history)
            transcript.append(Turn("writer", number, text))
            code = extract(text)
            attempt = run_checks(code)
            if not isinstance(attempt, Attempt):
                raise TypeError("run_checks must return a fin_skills.rsi.wiring.Attempt")
            if attempt.task_id != task_id:
                attempt = Attempt(task_id=task_id, code=attempt.code, executed=attempt.executed,
                                  error=attempt.error, checks=attempt.checks,
                                  valid=attempt.valid)
            attempts.append(attempt)
            if self.recorder is not None:
                observe(self.recorder, [attempt], available_at=self.available_at,
                        source_version=self.source_version)
            passed = attempt.executed and all(attempt.checks.values())
            report = self.findings(attempt, number, self.max_attempts)
            transcript.append(Turn("judge", number, "accepted" if passed else report))
            if passed:
                return Result(task_id, self.mode, True, code, tuple(attempts), tuple(transcript))
            if number == self.max_attempts:
                break
            feedback = [report]
            failed = [n for n, ok in attempt.checks.items() if not ok] if attempt.executed else []
            known = [n for n in failed if n in CAUSES]
            if self.mode == "single":
                if known:
                    feedback.append(explain(known, tools=self.tools))
            elif not attempt.executed:
                note = self._ask(self.interface_messages(code, attempt.error))[:MAX_NOTE]
                transcript.append(Turn("reviewer:interface", number, note))
                feedback.append("Reviewer (interface): " + note)
            else:
                for family in FAMILIES:
                    if any(CHECK_FAMILY.get(n) == family for n in known):
                        note = self._ask(self.reviewer_messages(family, code, known,
                                                                traits))[:MAX_NOTE]
                        transcript.append(Turn(f"reviewer:{family}", number, note))
                        feedback.append(f"Reviewer ({family}): " + note)
            feedback.append("Return the complete corrected program.")
            history = history + [dict(role="assistant", content=text),
                                 dict(role="user", content="\n".join(feedback))]
        return Result(task_id, self.mode, False, code, tuple(attempts), tuple(transcript))


def check_solve(solve: Callable, prices: pd.DataFrame, split_factors: pd.DataFrame, *,
                ks: Sequence[int] = (), tol: float = 1e-8) -> dict[str, bool]:
    """The position-program checks, in process: split_representation, execution_lag and
    finite_output for solve(prices, split_factors) -> positions DataFrame.

    It calls solve directly; it does not sandbox anything. Run untrusted code in a confined
    worker (as benchmarks/agent_study does) and call this inside it.
    """
    from fin_skills.api import get
    base = solve(prices, split_factors)
    if not isinstance(base, pd.DataFrame) or not base.index.equals(prices.index) or list(
            base.columns) != list(prices.columns):
        raise ValueError("solve must return a DataFrame with the input index and columns")
    values = base.to_numpy(dtype=float)
    checks = {"finite_output": bool(np.isfinite(values).all())}
    ones = pd.DataFrame(1.0, index=prices.index, columns=prices.columns)
    try:
        other = solve(prices * split_factors, ones).to_numpy(dtype=float)
        checks["split_representation"] = bool(np.allclose(
            np.nan_to_num(other), np.nan_to_num(values), atol=tol, rtol=0))
    except Exception:
        checks["split_representation"] = False
    ks = list(ks) or [len(prices) // 2]
    lag_ok = True
    for k in ks:
        verdict = get("execution_lag").run(fn=lambda frame: solve(frame, split_factors),
                                           df=prices, k=int(k))
        lag_ok &= bool(verdict.passed) and not verdict.warnings
    checks["execution_lag"] = lag_ok
    return checks
