"""Equal budgets for every RSI method, enforced by one wrapper around the user's model.

Every model call a method makes - writer, reflection, reviewer, proposer - goes through a
Meter. The meter counts calls and generated tokens and refuses every call once a limit is
reached, so "method X used more compute" cannot happen silently. The token count is known only
after a call returns, so the last call can overshoot generated_tokens by at most the backend's
own per-call max_tokens; the overshoot is recorded, not hidden.

    from fin_skills.rsi import Budget, Meter
    meter = Meter(generate, Budget(attempts=3, calls=6, generated_tokens=12_000))
    text = meter(messages)          # raises BudgetExceeded once a limit is reached
    meter.calls, meter.generated_tokens, meter.attempts

generate(messages) may return a string or an OpenAI-style dict
{"choices": [{"message": {"content": ...}}], "usage": {"completion_tokens": n}}. With a dict
the backend's own token count is used; with a plain string the count is an estimate
(characters / 4, rounded up) and Meter.estimated_tokens is True.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Callable


class BudgetExceeded(RuntimeError):
    """A method asked for more model calls, tokens or program executions than allowed."""

    def __init__(self, limit: str, used: int, allowed: int):
        super().__init__(f"{limit} budget exhausted: {used} used of {allowed}")
        self.limit, self.used, self.allowed = limit, used, allowed


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


@dataclass(frozen=True)
class Budget:
    """Per-task, test-time limits shared by every method."""

    attempts: int = 3                 # program executions by the judge
    calls: int = 6                    # model calls of any role
    generated_tokens: int = 12_000    # tokens the model generates across those calls

    def __post_init__(self):
        for name in ("attempts", "calls", "generated_tokens"):
            _positive(getattr(self, name), name)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FitBudget:
    """Development-time limits for methods that learn before the test tasks."""

    episodes: int = 48                # development episodes the method may run
    proposals: int = 6                # playbook / prompt revisions it may propose
    calls: int = 600                  # model calls of any role across the whole fit
    generated_tokens: int = 1_200_000

    def __post_init__(self):
        for name in ("episodes", "proposals", "calls", "generated_tokens"):
            _positive(getattr(self, name), name)

    def as_dict(self) -> dict:
        return asdict(self)


def _content(response: Any) -> tuple[str, int | None]:
    if isinstance(response, str):
        return response, None
    if isinstance(response, dict):
        try:
            text = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise TypeError("a dict response must have choices[0].message.content") from None
        usage = response.get("usage") or {}
        tokens = usage.get("completion_tokens")
        if tokens is not None and (isinstance(tokens, bool) or not isinstance(tokens, int)
                                   or tokens < 0):
            raise TypeError("usage.completion_tokens must be a nonnegative integer")
        if not isinstance(text, str):
            raise TypeError("message content must be a string")
        return text, tokens
    raise TypeError("generate must return a string or an OpenAI-style response dict")


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


class Meter:
    """Counts and limits model calls, generated tokens and judge executions."""

    def __init__(self, generate: Callable[[list[dict]], Any], budget: Budget | FitBudget, *,
                 attempts: int | None = None):
        if not callable(generate):
            raise TypeError("generate must be callable: generate(messages) -> str")
        self._generate, self.budget = generate, budget
        self.max_attempts = attempts if attempts is not None else getattr(budget, "attempts",
                                                                          None)
        self.calls = self.generated_tokens = self.attempts = 0
        self.estimated_tokens = False
        self.log: list[dict] = []

    def __call__(self, messages: list[dict], *, role: str = "writer") -> str:
        if self.calls >= self.budget.calls:
            raise BudgetExceeded("calls", self.calls + 1, self.budget.calls)
        if self.generated_tokens >= self.budget.generated_tokens:
            raise BudgetExceeded("generated_tokens", self.generated_tokens,
                                 self.budget.generated_tokens)
        response = self._generate([dict(m) for m in messages])
        text, tokens = _content(response)
        if tokens is None:
            tokens, self.estimated_tokens = estimate_tokens(text), True
        self.calls += 1
        self.generated_tokens += tokens
        self.log.append(dict(role=role, tokens=tokens))
        return text

    def respond(self, messages: list[dict], *, role: str = "writer") -> dict:
        """Like calling the meter, but returns an OpenAI-style dict with this call's token
        count, so a Meter can wrap another Meter (an episode inside a fit) and both count."""
        text = self(messages, role=role)
        return {"choices": [{"message": {"content": text}}],
                "usage": {"completion_tokens": self.log[-1]["tokens"]}}

    def can_attempt(self) -> bool:
        return self.max_attempts is None or self.attempts < self.max_attempts

    def charge_attempt(self) -> None:
        """Called by the judge wrapper before each program execution."""
        if self.max_attempts is not None and self.attempts >= self.max_attempts:
            raise BudgetExceeded("attempts", self.attempts + 1, self.max_attempts)
        self.attempts += 1

    def usage(self) -> dict:
        return dict(calls=self.calls, generated_tokens=self.generated_tokens,
                    attempts=self.attempts, estimated_tokens=self.estimated_tokens,
                    roles=[entry["role"] for entry in self.log])
