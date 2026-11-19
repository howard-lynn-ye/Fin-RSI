"""P-confirm: the frozen library on tasks written after the freeze (confirmatory).

Design: docs/P_CONFIRM_STUDY_DESIGN_20261119.md. One system prompt and task prompt (P-fix's),
Check loop, at most three programs per episode, the six visible checks. The arms differ only
in what is added:

  C0  no library
  C1  the library's position interface in the system prompt
  C2  C1 after route(): the model is first asked which catalogued rule the task describes; if
      it names one, the task gets routing.hint(id), the call with its parameter names
  C3  C1 with the hint for the task's own rule when the catalogue has it, and no hint when it
      does not. The harness reads the task's label, so this is the upper bound for C2.
  C4  C1 with the hint for the top BM25 catalogue entry (text retrieval; it cannot abstain)

Nothing in C0, C1, C2 or C4 comes from the task's label. Feedback is "check" (the names of the
failed checks) or "cause" (the names and each check's cause sentence, no rule cards); the
development run on the earlier tasks chooses one, and the confirmatory run uses only that.
"""
from __future__ import annotations

from fin_skills.algorithms import routing
from fin_skills.tasks.data import as_seed
from fin_skills.tasks.judge import Judge
from fin_skills.tasks.prompts import task_prompt
from . import adopt, pfix
from .meter import Budget
from .run_record import digest
from .study import RETRY, StudyArm

ARMS = ("C0", "C1", "C2", "C3", "C4")
FEEDBACK = ("check", "cause")
# The feedback of the confirmatory run, set once from the development run by the rule in the
# design (cause only if it is valid in at least three more of the 144 episodes). None: not
# chosen yet, and the confirmatory run refuses to start.
MAIN_FEEDBACK = None
PRIMARY_MODELS = ("deepseek-r1-distill-14b", "mistral-small-24b", "phi4-14b", "qwen-coder-14b",
                  "qwen-coder-32b", "qwen-general-14b", "qwen-general-32b", "qwen3-30b-a3b")
ROUTE_CAP = pfix.ROUTE_CAP
VISIBLE_CHECKS = pfix.VISIBLE_CHECKS


def catalogued(task) -> bool:
    """Whether positions() has the task's rule (an out-of-catalogue task has no right id)."""
    return task.algorithm in routing.RULES


def retrieval_documents() -> dict:
    """P-adopt's catalogue entries with each algorithm's rule line added: what route() shows the
    model, as BM25 documents. Chosen on the earlier tasks, where it names the right algorithm
    for 16 of 18 from the task prompt (metadata alone: 7; rule lines alone: 14)."""
    return {key: dict(entry, caveat=(entry.get("caveat", "") + " " + routing.RULES[key]).strip())
            for key, entry in adopt.retrieval_pool().items()}


def retrieved_id(task, *, seed) -> str:
    """C4: the top BM25 entry for the task prompt. Deterministic; never abstains."""
    return adopt.bm25_ranking(task_prompt(task, seed=seed), retrieval_documents())[0]


def route_messages(task, *, seed):
    return pfix.route_messages(task, seed=seed)


def confirm_messages(task, *, seed, arm, named=None, extra=""):
    """named: the catalogue id the arm's hint is for (C2: the model's answer to route_messages;
    C3: the task's own rule; C4: the retrieved entry), or None for no hint."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    if named is not None and (arm in ("C0", "C1") or named not in routing.RULES):
        raise ValueError("a hint needs arm C2, C3 or C4 and a catalogue id")
    messages = pfix.pfix_messages(task, seed=seed, arm="F0" if arm == "C0" else "F1", extra=extra)
    if named:
        messages[1]["content"] += "\n\n" + routing.hint(named)
    return messages


class _ConfirmArm(StudyArm):
    confirm_arm = None
    feedback = "check"
    causes = False        # True: Check feedback also gives each failed check's cause sentence

    def __init__(self, *, library: bool = False, cards: bool = False):
        if bool(library) != (self.confirm_arm != "C0"):
            raise ValueError(f"arm {self.confirm_arm} runs only with "
                             f"library={self.confirm_arm != 'C0'}")
        if cards:
            raise ValueError("P-confirm arms show no rule cards")
        super().__init__(library=library, cards=False)
        self._routes = {}

    def named(self, task, seed):
        if self.confirm_arm == "C2":
            return self._routes.get((task.id, as_seed(seed)))
        if self.confirm_arm == "C3":
            return task.algorithm if catalogued(task) else None
        if self.confirm_arm == "C4":
            return retrieved_id(task, seed=seed)
        return None

    def _messages(self, task, seed, extra=""):
        return confirm_messages(task, seed=seed, arm=self.confirm_arm,
                                named=self.named(task, seed), extra=extra)

    def make_judge(self, task, *, seed, meter):
        return Judge(task, seed=seed, library=self.library, meter=meter, properties=True)

    def _check_feedback(self, seen):
        if not seen.executed or not self.causes:
            return super()._check_feedback(seen)
        return ("Failed checks: " + ", ".join(sorted(seen.failed)) + ".\n"
                + seen.check_text(tools=False, rules=False) + "\n" + RETRY)

    def _solve(self, meter, judge, task, seed, episode):
        if self.confirm_arm == "C2":
            reply = meter(route_messages(task, seed=seed), role="router",
                          max_new_tokens=ROUTE_CAP)
            self._routes[(task.id, as_seed(seed))] = routing.parse_route(reply)
            episode.extra = dict(routed=self.named(task, seed), route_reply=reply[-400:])
        elif self.confirm_arm in ("C3", "C4"):
            episode.extra = dict(routed=self.named(task, seed))
        return self._loop(meter, judge, task, seed, episode)


def _arm(arm, causes=False):
    name = f"confirm_{arm}" + ("_cause" if causes else "")
    return type(name.title().replace("_", ""), (_ConfirmArm,),
                dict(confirm_arm=arm, name=name, causes=causes))


C0, C1, C2, C3, C4 = (_arm(arm) for arm in ARMS)
C0Cause, C1Cause, C2Cause, C3Cause, C4Cause = (_arm(arm, True) for arm in ARMS)
METHODS = {"check": (C0, C1, C2, C3, C4), "cause": (C0Cause, C1Cause, C2Cause, C3Cause, C4Cause)}


def first_messages(method, task, *, seed):
    """The first prompt an arm sends for a task (C2: the routing question)."""
    if method.confirm_arm == "C2":
        return route_messages(task, seed=seed)
    return confirm_messages(task, seed=seed, arm=method.confirm_arm,
                            named=method(library=method.confirm_arm != "C0").named(task, seed))


def manifest(tasks, methods, *, seeds, budget=Budget(), stage="main"):
    tasks, seeds, methods = tuple(tasks), tuple(seeds), tuple(methods)
    packets = [dict(arm=method.confirm_arm, method=method.name, causes=method.causes,
                    library=method.confirm_arm != "C0", task_id=task.id, task_sha256=task.sha256,
                    catalogued=catalogued(task), seed=seed,
                    messages_sha256=digest(first_messages(method, task, seed=seed)))
               for method in methods for task in tasks for seed in seeds]
    return dict(version="p-confirm-v1", stage=stage,
                exploratory=stage != "main", publication_ready=False,
                budget=budget.as_dict(), output_cap=adopt.OUTPUT_CAP, cards=False,
                interface_sha256=digest(pfix.interface_text()),
                visible_checks=VISIBLE_CHECKS,
                hint=routing.HINT,
                route=dict(system=routing.ROUTE_SYSTEM, question=routing.ROUTE_QUESTION,
                           rules_sha256=digest(routing.RULES), output_cap=ROUTE_CAP),
                retrieval=dict(ranking="BM25 k1=1.5 b=0.75, top entry, no abstention",
                               query="task prompt",
                               documents_sha256=digest(retrieval_documents())),
                task_ids=[task.id for task in tasks], seeds=list(seeds), packets=packets)
