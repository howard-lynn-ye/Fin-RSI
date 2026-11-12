"""Controlled feedback x library arms for the next study, not new scientific results.

Initial instructions contain necessary task semantics but no teaching rule cards. All arms
select the first visibly accepted program. Reference grades are added only after solving.
The existing seven RSI methods and their prompts remain unchanged.
"""
from .methods import BaseRSI, RETRY
from .bench import benchmark
from .meter import Budget
from .run_record import digest
from fin_skills.tasks.prompts import (SYSTEM, NO_LIBRARY, TIMING, WHAT, WITH_LIBRARY,
                                      library_text, task_prompt)
from fin_skills.tasks.spec import FACTOR_TEXT
from fin_skills.api.rules import explain


def initial_messages(task, *, seed, library):
    system = SYSTEM.format(what=WHAT[task.kind], factors=FACTOR_TEXT[task.factors],
                           timing=TIMING[task.kind])
    system += WITH_LIBRARY if library else NO_LIBRARY
    if library:
        system += "\n\nfin_skills interface:\n" + library_text(rolling=False)
    return [dict(role="system", content=system),
            dict(role="user", content=task_prompt(task, seed=seed))]


class StudyArm(BaseRSI):
    """No arm shows rule cards: the first prompt is initial_messages(), which has none."""
    feedback = None

    def __init__(self, *, library: bool = False, cards: bool = False):
        if cards:
            raise ValueError("controlled study arms show no rule cards")
        super().__init__(library=library, cards=False)

    def _messages(self, task, seed, extra=""):
        if extra:
            raise ValueError("controlled study does not permit learned prompt addenda")
        return initial_messages(task, seed=seed, library=self.library)

    def _check_feedback(self, seen):
        if not seen.executed:
            return "Program execution failed:\n" + seen.error[-1500:] + "\n" + RETRY
        text = "Failed checks: " + ", ".join(sorted(seen.failed)) + ".\n"
        if self.feedback == "explain":
            text += seen.check_text(tools=False) + "\n"
        return text + RETRY

    def _solve(self, meter, judge, task, seed, episode):
        if self.feedback != "repeat":
            return self._loop(meter, judge, task, seed, episode)
        while meter.can_attempt():
            _, seen = self._attempt(meter, judge, self._messages(task, seed), episode)
            if seen.accepted:
                episode.status = "accepted"
                return
        episode.status = "exhausted"


class Repeat(StudyArm):
    name = feedback = "repeat"


class Check(StudyArm):
    name = feedback = "check"


class Explain(StudyArm):
    name = feedback = "explain"


METHODS = (Repeat, Check, Explain)
PRIMARY_CONTRASTS = ((("check", False), ("repeat", False)),
                     (("explain", False), ("check", False)),
                     (("explain", True), ("explain", False)))


def manifest(tasks, *, seeds=(11, 23, 37), budget=Budget()):
    tasks, seeds = tuple(tasks), tuple(seeds)
    packets = [dict(arm=f"{method.name}_{'library' if library else 'scratch'}",
                    method=method.name, feedback=method.feedback, library=library,
                    task_id=task.id, task_sha256=task.sha256, seed=seed,
                    messages_sha256=digest(initial_messages(task, seed=seed, library=library)))
               for method in METHODS for library in (False, True)
               for task in tasks for seed in seeds]
    return dict(version="controlled-feedback-v1", publication_ready=False,
                exposure="public regression unless independently authored and retained",
                budget=budget.as_dict(), initial_teaching_cards=False,
                selector="first visible acceptance; no reference-grade selection",
                library_effect="numerical API plus necessary interface documentation",
                explanatory_feedback_tools=False, primary_contrasts=PRIMARY_CONTRASTS,
                explanation_sha256={name: digest(explain([name], tools=False)) for name in
                    ("finite_output", "execution_lag", "split_representation")},
                packets=packets)


def run(generate, tasks, *, seeds=(11, 23, 37), budget=Budget(), heldout=None, **kwargs):
    test_tasks = tuple(tasks.heldout if heldout is None else heldout)
    protocol = manifest(test_tasks, seeds=seeds, budget=budget)
    return benchmark(METHODS, generate, tasks, seeds=seeds, library=(False, True),
                     budget=budget, heldout=test_tasks, primary_contrasts=PRIMARY_CONTRASTS,
                     study_manifest=protocol, **kwargs)
