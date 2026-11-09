"""Self-improvement methods behind one interface, the way PyOD puts detectors behind one.

    from fin_skills import rsi
    from fin_skills.tasks import suite
    tasks = suite()
    m = rsi.Reflexion(library=True)
    m.fit(generate, tasks.dev)                       # no-op for methods that do not learn
    episode = m.solve(generate, tasks.heldout[0], seed=11)
    episode.accepted, episode.code, episode.usage

generate(messages) -> str (or an OpenAI-style dict) is the user's own model. Every call goes
through a Meter, so all methods get the same budget. A method sees only what the judge shows
(ran?, error, failed checks and their causes); the hidden grade is computed by the benchmark
runner from the final program and is never passed back. library=False removes the fin_skills
interface, renders rule cards without library calls, and refuses `import fin_skills`.

Methods
  NoImprove      one program, no feedback
  SelfDebug      retry on errors only; stops once the program runs (Self-Debugging)
  CheckFeedback  retry with the failed checks, their causes and the rule cards
  Reflexion      CheckFeedback plus a written self-reflection before each retry (Reflexion)
  PlaybookLearning  learn an experience playbook on dev tasks with RSILoop, then CheckFeedback
                 with the frozen playbook in the prompt (ExpeL / Dynamic Cheatsheet)
  Team           writer plus per-rule-family reviewers built from the same model
  Evolve         evolve a system-prompt addendum on dev tasks, then CheckFeedback with it
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
import hashlib
import json
import tempfile
from typing import Callable, Sequence

from fin_skills.tasks import Judge, messages as task_messages, library_text, system_prompt
from fin_skills.tasks import task_prompt
from fin_skills.tasks.spec import Task

from .meter import Budget, BudgetExceeded, FitBudget, Meter, ModelRefusal
from .core import InvalidProposal
from .fences import extract, reply_format

FIT_START = datetime(2026, 11, 3, tzinfo=timezone.utc)
RETRY = "Return the complete corrected program in one python fence."


def _sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


@dataclass
class Step:
    number: int
    code: str
    executed: bool
    accepted: bool
    checks: dict
    error: str = ""
    feedback: str = ""
    library_calls: int = 0
    attempt_index: int = -1
    reply_format: str = ""            # fenced | unclosed_fence | no_fence; "" when no reply


@dataclass
class Episode:
    method: str
    task_id: str
    seed: int
    library: bool
    steps: list = field(default_factory=list)
    status: str = "exhausted"         # accepted | ran | exhausted | budget:<limit>
    usage: dict = field(default_factory=dict)
    state_sha256: str = ""
    valid: bool | None = None         # filled in by the benchmark runner, never by a method

    @property
    def code(self) -> str:
        return self.steps[-1].code if self.steps else ""

    @property
    def accepted(self) -> bool:
        """The final program passed every visible check (what a guarded controller accepts)."""
        return bool(self.steps) and self.steps[-1].accepted

    @property
    def library_used(self) -> bool:
        return bool(self.steps) and self.steps[-1].library_calls > 0

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update(accepted=self.accepted, library_used=self.library_used)
        return d


class _View:
    """The judge as a method sees it: check() only."""

    def __init__(self, judge: Judge):
        self.check = judge.check


class BaseRSI:
    name = "base"
    learns = False

    def __init__(self, *, library: bool = False):
        self.library = bool(library)
        self.fit_usage: dict = dict(calls=0, generated_tokens=0, episodes=0)
        self.fitted = not self.learns

    # public ------------------------------------------------------------------------------
    def fit(self, generate, dev_tasks: Sequence[Task], *, budget: FitBudget = FitBudget(),
            seeds: Sequence[int] = (11,), episode_budget: Budget = Budget()) -> "BaseRSI":
        """Learn on development tasks. Methods that do not learn only mark themselves fitted."""
        self.fitted = True
        return self

    def solve(self, generate, task: Task, *, seed: int, budget: Budget = Budget()) -> Episode:
        if not self.fitted:
            raise RuntimeError(f"{self.name} learns on dev tasks: call fit() before solve()")
        meter = Meter(generate, budget, seed=seed)
        judge = self.make_judge(task, seed=seed, meter=meter)
        episode = Episode(self.name, task.id, int(seed), self.library)
        try:
            self._solve(meter, _View(judge), task, seed, episode)
        except BudgetExceeded as exc:
            episode.status = f"budget:{exc.limit}"
        except ModelRefusal:
            episode.status = "model_refusal"
        episode.usage = meter.usage()
        episode.state_sha256 = _sha(self.state())
        episode._judge = judge                     # for the runner's grade; not for methods
        return episode

    def state(self) -> dict:
        """What the method learned (empty when it learns nothing)."""
        return {}

    # shared pieces -----------------------------------------------------------------------
    def make_judge(self, task, *, seed, meter):
        """Output-type extension point; every judge exposes the same check/grade boundary."""
        return Judge(task, seed=seed, library=self.library, meter=meter)

    def _messages(self, task, seed, extra=""):
        return task_messages(task, seed=seed, library=self.library, extra_system=extra)

    def _attempt(self, meter, judge, history, episode, *, role="writer"):
        if not meter.can_attempt():
            raise BudgetExceeded("attempts", meter.attempts + 1, meter.max_attempts)
        text = meter(history, role=role)
        seen = self._check_program(meter, judge, extract(text), episode)
        episode.steps[-1].reply_format = reply_format(text)
        return text, seen

    def _check_program(self, meter, judge, code, episode):
        """One common execution/recording boundary for writers and team callbacks."""
        if not meter.can_attempt():
            raise BudgetExceeded("attempts", meter.attempts + 1, meter.max_attempts)
        seen = judge.check(code)
        episode.steps.append(Step(len(episode.steps) + 1, code, seen.executed, seen.accepted,
                                  dict(seen.checks), seen.error[:2000],
                                  library_calls=seen.library_calls, attempt_index=seen.attempt_index))
        return seen

    def _check_feedback(self, seen) -> str:
        if not seen.executed:
            return "Your program did not run:\n" + seen.error[-1500:] + "\n" + RETRY
        return ("Your program ran but failed these checks:\n" + seen.check_text(tools=self.library)
                + "\n" + RETRY)

    def _loop(self, meter, judge, task, seed, episode, *, extra="", stop_when_runs=False,
              feedback=None):
        history = self._messages(task, seed, extra)
        while True:
            text, seen = self._attempt(meter, judge, history, episode)
            if seen.accepted or (stop_when_runs and seen.executed):
                episode.status = "accepted" if seen.accepted else "ran"
                return
            if not meter.can_attempt():
                episode.status = "exhausted"
                return
            note = (feedback or self._check_feedback)(seen)
            episode.steps[-1].feedback = note
            history = history + [dict(role="assistant", content=text),
                                 dict(role="user", content=note)]

    def _solve(self, meter, judge, task, seed, episode):
        raise NotImplementedError


class NoImprove(BaseRSI):
    name = "no_improve"

    def _solve(self, meter, judge, task, seed, episode):
        _, seen = self._attempt(meter, judge, self._messages(task, seed), episode)
        episode.status = "accepted" if seen.accepted else ("ran" if seen.executed else
                                                            "exhausted")


class SelfDebug(BaseRSI):
    """Retries only when the program fails to run; the checks are not shown."""

    name = "self_debug"

    def _solve(self, meter, judge, task, seed, episode):
        def error_only(seen):
            return "Your program did not run:\n" + seen.error[-1500:] + "\n" + RETRY
        self._loop(meter, judge, task, seed, episode, stop_when_runs=True, feedback=error_only)


class CheckFeedback(BaseRSI):
    name = "check_feedback"

    def _solve(self, meter, judge, task, seed, episode):
        self._loop(meter, judge, task, seed, episode)


class RepeatedSampling(BaseRSI):
    """Optional compute control: fresh independent attempts, same first-accepted selector.

    No failures, checks or previous programs are shown to the next writer. This comparator
    is available for a new frozen protocol; it does not expand the seven-method default.
    """
    name = "repeated_sampling"

    def _solve(self, meter, judge, task, seed, episode):
        while meter.can_attempt():
            _, seen = self._attempt(meter, judge, self._messages(task, seed), episode)
            if seen.accepted:
                episode.status = "accepted"
                return
        episode.status = "exhausted"


REFLECT = ("You wrote the program below and it failed. In at most five sentences, explain what "
           "caused the failure and what you will change. Do not write code.")


class Reflexion(BaseRSI):
    name = "reflexion"

    def _solve(self, meter, judge, task, seed, episode):
        history, reflections = self._messages(task, seed), []
        while True:
            text, seen = self._attempt(meter, judge, history, episode)
            if seen.accepted:
                episode.status = "accepted"
                return
            if not meter.can_attempt():
                episode.status = "exhausted"
                return
            findings = self._check_feedback(seen).replace("\n" + RETRY, "")
            reflection = meter([dict(role="system", content=REFLECT),
                                dict(role="user", content=f"Rule:\n{task.definition}\n\n"
                                     f"Program:\n```python\n{episode.code}```\n\n{findings}")],
                               role="reflection")[:1500]
            reflections.append(reflection)
            note = (findings + "\n\nYour reflections so far:\n"
                    + "\n".join(f"{i}. {r}" for i, r in enumerate(reflections, 1))
                    + "\n" + RETRY)
            episode.steps[-1].feedback = note
            history = history + [dict(role="assistant", content=text),
                                 dict(role="user", content=note)]


class Team(BaseRSI):
    name = "team"

    def _solve(self, meter, judge, task, seed, episode):
        from fin_skills.agents import Team as Roles
        from .wiring import Attempt

        def run_checks(code):
            seen = self._check_program(meter, judge, code, episode)
            return Attempt(task_id=task.id, code=code, executed=seen.executed,
                           error=seen.error, checks=dict(seen.checks))

        def labelled(messages):
            reviewer = messages and messages[0]["content"].startswith("You review")
            return meter(messages, role="reviewer" if reviewer else "writer")

        team = Roles(labelled, mode="roles", max_attempts=meter.max_attempts,
                     writer_rules="none", tools=self.library)
        text = (system_prompt(task, library=self.library) + "\n\n"
                + task_prompt(task, seed=seed))
        result = team.solve(text, run_checks, traits=task.traits, task_id=task.id)
        episode.status = "accepted" if result.accepted else "exhausted"


# Methods that learn on development tasks ---------------------------------------------------

class _Learner(BaseRSI):
    learns = True

    def __init__(self, *, library: bool = False):
        super().__init__(library=library)
        self.addendum = ""

    def _dev_episode(self, fit_meter, task, seed, episode_budget, extra):
        """One CheckFeedback episode on a dev task, graded (allowed: dev tasks only)."""
        if hasattr(fit_meter._generate, "seed"):
            fit_meter._generate.seed = seed
        meter = Meter(fit_meter.respond, episode_budget, seed=seed)
        judge = Judge(task, seed=seed, library=self.library, meter=meter)
        episode = Episode(self.name, task.id, int(seed), self.library)
        self.fit_usage["episodes"] += 1  # Reserve the episode before any charged request.
        try:
            self._loop(meter, _View(judge), task, seed, episode, extra=extra)
        except BudgetExceeded as exc:
            episode.status = f"budget:{exc.limit}"
            spent = fit_meter.budget
            if (fit_meter.calls >= spent.calls
                    or fit_meter.generated_tokens >= spent.generated_tokens):
                raise                            # the fit budget itself is spent
        except ModelRefusal:
            episode.status = "model_refusal"
        finally:
            statuses = self.fit_usage.setdefault("episode_statuses", {})
            statuses[episode.status] = statuses.get(episode.status, 0) + 1
        episode.valid = bool(episode.steps) and judge.grade(episode.code)
        return episode

    def _spend(self, fit_meter, budget):
        self.fit_usage.update(calls=fit_meter.calls, generated_tokens=fit_meter.generated_tokens)
        if self.fit_usage["episodes"] >= budget.episodes:
            raise BudgetExceeded("episodes", self.fit_usage["episodes"], budget.episodes)

    def _solve(self, meter, judge, task, seed, episode):
        self._loop(meter, judge, task, seed, episode, extra=self.addendum)

    def state(self) -> dict:
        return dict(addendum=self.addendum)


class PlaybookLearning(_Learner):
    """RSILoop over dev tasks with the no_regression gate; the frozen playbook is the addendum."""

    name = "playbook"

    def fit(self, generate, dev_tasks, *, budget=FitBudget(), seeds=(11,),
            episode_budget=Budget()):
        if len(seeds) != 1:
            raise ValueError("PlaybookLearning requires exactly one declared fit seed")
        from .core import Policy
        from .wiring import Attempt, loop_from_checks, observe
        dev_tasks = list(dev_tasks)
        algorithms = sorted({t.algorithm for t in dev_tasks})
        validation_algorithms = set(algorithms[1::2])        # disjoint halves by algorithm
        split = {"development": [t.id for t in dev_tasks
                                 if t.algorithm not in validation_algorithms],
                 "validation": [t.id for t in dev_tasks if t.algorithm in validation_algorithms]}
        by_id = {t.id: t for t in dev_tasks}
        fit_meter = Meter(generate, budget, attempts=None, seed=int(seeds[0]))
        seed = int(seeds[0])

        def to_attempt(ep):
            last = ep.steps[-1] if ep.steps else Step(0, "", False, False, {})
            return Attempt(task_id=ep.task_id, code=last.code or "\n", executed=last.executed,
                           error=last.error, checks=dict(last.checks),
                           valid=bool(ep.valid and last.executed and all(last.checks.values())))

        def run_attempt(task_id, playbook_text):
            self._spend(fit_meter, budget)
            return to_attempt(self._dev_episode(fit_meter, by_id[task_id], seed,
                                                episode_budget, playbook_text))

        directory = tempfile.mkdtemp(prefix="fin-rsi-playbook-")
        loop = loop_from_checks(
            directory, lambda prompt: fit_meter([dict(role="user", content=prompt)],
                                                role="proposer"),
            run_attempt, split, model_id="caller-model", protocol_id="fin-skills-rsi-bench",
            available_at=FIT_START.isoformat(),
            policy=Policy(enabled=True, guard_rule="no_regression",
                          max_proposals=budget.proposals,
                          max_evaluations=4 * budget.proposals + 4))
        decisions = []
        try:
            for task_id in split["development"]:
                observe(loop, [run_attempt(task_id, "")], available_at=FIT_START.isoformat(),
                        source_version="fin-skills")
            for i in range(budget.proposals):
                as_of = (FIT_START + timedelta(minutes=i + 1)).isoformat()
                decisions.append(loop.improve(as_of=as_of).status)
                if decisions[-1] in ("budget_exhausted", "no_evidence", "frozen"):
                    break
        except BudgetExceeded:
            decisions.append("fit_budget_exhausted")
            self.fit_usage["stopped_reason"] = "budget_exhausted"
        except (ModelRefusal, InvalidProposal) as exc:
            reason = "model_refusal" if isinstance(exc, ModelRefusal) else "invalid_proposal"
            decisions.append("fit_" + reason)
            self.fit_usage["stopped_reason"] = reason
        final = (FIT_START + timedelta(minutes=budget.proposals + 2)).isoformat()
        loop.freeze(as_of=final)
        self.addendum = loop.instructions(as_of=final)
        self.decisions, self.journal = decisions, directory
        self.fit_usage.update(calls=fit_meter.calls, generated_tokens=fit_meter.generated_tokens)
        self.fitted = True
        return self

    def state(self) -> dict:
        return dict(playbook=self.addendum)


MUTATE = ("You are improving the standing instructions given to a programmer who writes "
          "financial position programs. Below are the current instructions and what went wrong "
          "on recent tasks. Write improved instructions: at most 1200 characters, general "
          "advice only, no task-specific numbers or code. Reply with the instructions only.")


class Evolve(_Learner):
    """Evolve a system-prompt addendum on dev tasks; keep a candidate only if it scores
    strictly higher on the same dev tasks (valid count, then passed-check share)."""

    name = "evolve"

    def __init__(self, *, library: bool = False, generations: int = 3, minibatch: int = 4):
        super().__init__(library=library)
        self.generations, self.minibatch = generations, minibatch
        self.history: list[dict] = []

    def _score(self, episodes):
        valid = sum(bool(e.valid) for e in episodes)
        # Every planned development episode keeps its weight. Refusing to deliver a
        # program cannot improve a candidate by removing a low-scoring denominator.
        shares = []
        for episode in episodes:
            last = episode.steps[-1] if episode.steps else None
            shares.append(sum(last.checks.values()) / len(last.checks)
                          if last and last.executed and last.checks else 0.0)
        return (valid, sum(shares) / len(shares) if shares else 0.0)

    def fit(self, generate, dev_tasks, *, budget=FitBudget(), seeds=(11,),
            episode_budget=Budget()):
        if len(seeds) != 1:
            raise ValueError("Evolve requires exactly one declared fit seed")
        dev_tasks = sorted(dev_tasks, key=lambda t: t.id)
        batch = dev_tasks[:self.minibatch]
        fit_meter = Meter(generate, budget, attempts=None, seed=int(seeds[0]))
        seed = int(seeds[0])

        def evaluate(addendum):
            out = []
            for t in batch:
                self._spend(fit_meter, budget)
                out.append(self._dev_episode(fit_meter, t, seed, episode_budget, addendum))
            return out

        best = ""
        try:
            best_eps = evaluate(best)
            best_score = self._score(best_eps)
            self.history.append(dict(generation=0, score=best_score, accepted=True))
            for g in range(1, min(self.generations, budget.proposals) + 1):
                failures = [f"- {e.task_id}: " + ((e.steps[-1].feedback or e.steps[-1].error
                                                  or "failed checks") if e.steps else e.status)[:400]
                            for e in best_eps if not e.valid]
                if not failures:
                    break
                candidate = fit_meter([dict(role="system", content=MUTATE),
                                       dict(role="user", content="Current instructions:\n"
                                            + (best or "(none)") + "\n\nRecent failures:\n"
                                            + "\n".join(failures))],
                                      role="proposer").strip()
                if not candidate or len(candidate) > 1200:
                    raise InvalidProposal("instructions must contain 1 to 1200 characters")
                eps = evaluate(candidate)
                score = self._score(eps)
                improved = score > best_score
                self.history.append(dict(generation=g, score=score, accepted=improved))
                if improved:
                    best, best_eps, best_score = candidate, eps, score
        except BudgetExceeded:
            self.fit_usage["stopped_reason"] = "budget_exhausted"
            self.history.append(dict(generation=None, score=None, accepted=False,
                                     note="fit budget exhausted"))
        except (ModelRefusal, InvalidProposal) as exc:
            reason = "model_refusal" if isinstance(exc, ModelRefusal) else "invalid_proposal"
            self.fit_usage["stopped_reason"] = reason
            self.history.append(dict(generation=None, score=None, accepted=False, note=reason))
        self.addendum = best
        self.fit_usage.update(calls=fit_meter.calls, generated_tokens=fit_meter.generated_tokens)
        self.fitted = True
        return self


METHODS = (NoImprove, SelfDebug, CheckFeedback, Reflexion, PlaybookLearning, Team, Evolve)
