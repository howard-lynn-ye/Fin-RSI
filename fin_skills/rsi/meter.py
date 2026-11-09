"""Equal budgets for every RSI method, enforced by one wrapper around the user's model.

Every model call a method makes - writer, reflection, reviewer, proposer - goes through a
Meter. The meter counts calls and generated tokens and refuses every call once a limit is
reached. A generate_bounded backend receives the remaining output cap and an explicit seed,
and must return exact usage. Legacy callables are diagnostic only: usage may be estimated and
overshoot is recorded, but an over-budget response cannot be selected as a deliverable.

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
from contextvars import ContextVar
import hashlib
import json
import math
from typing import Any, Callable

EVENT_SINK = ContextVar("fin_rsi_event_sink", default=None)
BACKEND_BINDING = ContextVar("fin_rsi_backend_binding", default=None)


def event(kind, **details):
    sink = EVENT_SINK.get()
    if sink is not None:
        sink(dict(kind=kind, **details))


class BudgetExceeded(RuntimeError):
    """A method asked for more model calls, tokens or program executions than allowed."""

    def __init__(self, limit: str, used: int, allowed: int):
        super().__init__(f"{limit} budget exhausted: {used} used of {allowed}")
        self.limit, self.used, self.allowed = limit, used, allowed


class ModelRefusal(RuntimeError):
    """An explicit provider refusal is a recorded model outcome, not an adapter outage."""


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
        if text is None and response["choices"][0]["message"].get("refusal"):
            raise ModelRefusal(str(response["choices"][0]["message"]["refusal"])[:2000])
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
                 attempts: int | None = None, seed: int | None = None):
        if not callable(generate):
            raise TypeError("generate must be callable: generate(messages) -> str")
        self._generate, self.budget = generate, budget
        self.seed = int(getattr(generate, "seed", 0) if seed is None else seed)
        self.max_attempts = attempts if attempts is not None else getattr(budget, "attempts",
                                                                          None)
        self.calls = self.generated_tokens = self.attempts = 0
        self.estimated_tokens = False
        self.log: list[dict] = []

    def __call__(self, messages: list[dict], *, role: str = "writer",
                 max_new_tokens: int | None = None, seed: int | None = None) -> str:
        binding = BACKEND_BINDING.get()
        if binding is not None and self._generate is binding[0]:
            from .backend import describe_backend
            if describe_backend(self._generate, strict=True) != binding[1]:
                raise ValueError("backend descriptor changed after protocol freeze")
        if self.calls >= self.budget.calls:
            raise BudgetExceeded("calls", self.calls + 1, self.budget.calls)
        if self.generated_tokens >= self.budget.generated_tokens:
            raise BudgetExceeded("generated_tokens", self.generated_tokens,
                                 self.budget.generated_tokens)
        remaining = self.budget.generated_tokens - self.generated_tokens
        if max_new_tokens is not None:
            remaining = min(remaining, max_new_tokens)
        nested = getattr(self._generate, "__self__", None)
        bounded = getattr(self._generate, "generate_bounded", None)
        nested_log_start = len(nested.log) if isinstance(nested, Meter) else 0
        if seed is None:
            base_seed = self.seed
            ordinal = sum(e["role"] == role for e in self.log)
            seed = int(hashlib.sha256(f"{base_seed}:{role}:{ordinal}".encode()).hexdigest()[:8], 16)
        mode = "nested" if isinstance(nested, Meter) else ("bounded" if bounded else "posthoc")
        event("call_started", role=role, messages=messages, max_new_tokens=remaining,
              seed=seed, backend_mode=mode, call_index=self.calls)
        self.calls += 1
        known_cost = None
        try:
            packet = [dict(m) for m in messages]
            if isinstance(nested, Meter):
                response = self._generate(packet, max_new_tokens=remaining, seed=seed, role=role)
            elif bounded:
                response = bounded(packet, max_new_tokens=remaining, seed=seed)
            else:
                if hasattr(self._generate, "seed"):
                    self._generate.seed = seed
                response = self._generate(packet)
            if isinstance(response, dict):
                usage = response.get("usage") or {}
                count = usage.get("completion_tokens") if isinstance(usage, dict) else None
                if type(count) is int and count >= 0:
                    known_cost = count
                    self.estimated_tokens |= bool(usage.get("completion_tokens_estimated"))
                receipt = {k: response[k] for k in ("id", "model", "choices", "usage", "status",
                           "error", "refusal", "system_fingerprint") if k in response}
            else:
                receipt = response if isinstance(response, str) else {
                    "unexpected_type": type(response).__name__}
            # Keep response/finish/refusal/usage evidence even when content parsing fails.
            receipt = json.loads(json.dumps(receipt, default=lambda value: {
                "unserializable_type": type(value).__name__}))
            event("response_received", role=role, response=receipt, seed=seed)
            if bounded and (known_cost is None or self.estimated_tokens or known_cost > remaining):
                raise RuntimeError("bounded backend violated its exact usage/output limit contract")
            text, tokens = _content(response)
        except Exception as exc:
            consumed = (nested.log[-1].get("tokens") if isinstance(nested, Meter)
                        and len(nested.log) > nested_log_start else known_cost)
            if consumed is not None:
                self.generated_tokens += consumed
                if isinstance(nested, Meter):
                    self.estimated_tokens |= nested.estimated_tokens
            self.log.append(dict(role=role, tokens=consumed, error=type(exc).__name__))
            event("call_failed", role=role, error=type(exc).__name__, message=str(exc),
                  token_cost=consumed, cost_known=consumed is not None, backend_mode=mode)
            raise
        if tokens is None:
            tokens, self.estimated_tokens = estimate_tokens(text), True
        if isinstance(response, dict):
            self.estimated_tokens |= bool((response.get("usage") or {}).get(
                "completion_tokens_estimated", False))
        self.generated_tokens += tokens
        self.log.append(dict(role=role, tokens=tokens, seed=seed, backend_mode=mode))
        event("call_completed", role=role, content=text, tokens=tokens,
              estimated_tokens=self.estimated_tokens, seed=seed, backend_mode=mode)
        if bounded and (self.estimated_tokens or tokens > remaining):
            raise RuntimeError("bounded backend violated its exact usage/output limit contract")
        if self.generated_tokens > self.budget.generated_tokens:
            raise BudgetExceeded("generated_tokens", self.generated_tokens,
                                 self.budget.generated_tokens)
        return text

    def respond(self, messages: list[dict], *, role: str = "writer",
                max_new_tokens: int | None = None, seed: int | None = None) -> dict:
        """Like calling the meter, but returns an OpenAI-style dict with this call's token
        count, so a Meter can wrap another Meter (an episode inside a fit) and both count."""
        text = self(messages, role=role, max_new_tokens=max_new_tokens, seed=seed)
        return {"choices": [{"message": {"content": text}}],
                "usage": {"completion_tokens": self.log[-1]["tokens"],
                          "completion_tokens_estimated": self.estimated_tokens}}

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
