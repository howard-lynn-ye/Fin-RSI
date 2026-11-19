"""P-fix: the frozen P track on the library after its agent-interface fix (exploratory).

Design: docs/P_FIX_STUDY_DESIGN_20261118.md. The frozen P study's library arms reached for the
library in 519 programs and 480 of those calls raised; the interface was then changed (PR #77:
keyword parameters, run() on a single input, actionable errors, signal_positions(), an explicit
"the result is final" contract). P-fix asks whether models now use the library and get the
held-out tasks right with it. It reuses P-adopt's machinery and its no-library arm unchanged:

  F0  no library: identical messages to P-adopt L0 (and to the frozen no-library first prompt)
  F1  the current library and its current position-task interface text
  F2  F1 plus P-adopt's one-sentence pointer naming the algorithm that implements the rule
  F3  (amendment 2, exploratory) F1 with the documented interface: the rule each algorithm
      computes, and that a matching task needs one positions() call; nothing task-specific
  F4  (amendment 3, exploratory) F1 after the library's route(): the model is first asked which
      catalogue rule, if any, computes the task's rule; when it names one, F2's sentence is
      added with that id. Nothing in the prompt comes from the task's label.

Amendment 3 also turns on the declared-property checks (tasks/properties.py) for every arm, so
all arms are rerun together (v4).

Every arm uses the frozen system prompt (including "Implement the rule exactly"), no rule
cards, names-only Check feedback (an execution error is shown in full, as in the frozen study),
at most 3 attempts and a 4,096-token output cap per reply, on the 6 held-out tasks x 3 seeds.
"""
from fin_skills.tasks.prompts import (SYSTEM, NO_LIBRARY, TIMING, WHAT, WITH_LIBRARY,
                                      library_text, task_prompt)
from fin_skills.tasks.spec import FACTOR_TEXT
from fin_skills.tasks.data import as_seed
from fin_skills.tasks.judge import Judge
from fin_skills.tasks.properties import CHECKS as PROPERTY_CHECKS
from fin_skills.algorithms import routing

from .adopt import BUDGET, OUTPUT_CAP, POINTER
from .run_record import digest
from .study import StudyArm

ARMS = ("F0", "F1", "F2", "F3", "F4")
VISIBLE_CHECKS = ("finite_output", "split_representation", "execution_lag") + PROPERTY_CHECKS
ROUTE_CAP = 1024   # generated tokens for F4's routing reply; they count against the episode budget


def interface_text(*, documented: bool = False) -> str:
    """What F1 and F2 append to the system prompt (the current positions-only interface);
    documented=True is F3's, which adds the rule each algorithm computes."""
    return library_text(rolling=False, frozen=False, documented=documented)


def route_messages(task, *, seed):
    """F4's first prompt: the library's routing question about the task as the model sees it."""
    return routing.route_messages(task_prompt(task, seed=seed))


def pfix_messages(task, *, seed, arm, extra="", routed=None):
    """routed: for F4, the catalogue id the model named in answer to route_messages()."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    library = arm != "F0"
    system = SYSTEM.format(what=WHAT[task.kind], factors=FACTOR_TEXT[task.factors],
                           timing=TIMING[task.kind])
    system += WITH_LIBRARY if library else NO_LIBRARY
    if library:
        system += "\n\nfin_skills interface:\n" + interface_text(documented=arm == "F3")
    messages = [dict(role="system", content=system),
                dict(role="user", content=task_prompt(task, seed=seed))]
    if routed is not None and (arm != "F4" or routed not in routing.RULES):
        raise ValueError("routed is F4's, and a catalogue id")
    if arm == "F2" or routed:
        messages[1]["content"] += "\n\n" + POINTER.format(
            algorithm=task.algorithm if arm == "F2" else routed)
    if extra:
        messages[0]["content"] += "\n\n" + extra
    return messages


class _FixArm(StudyArm):
    """Check arm with a fixed library condition and no rule cards."""
    fix_arm = None

    def __init__(self, *, library: bool = False, cards: bool = False):
        if bool(library) != (self.fix_arm != "F0"):
            raise ValueError(f"arm {self.fix_arm} runs only with library={self.fix_arm != 'F0'}")
        if cards:
            raise ValueError("P-fix arms show no rule cards")
        super().__init__(library=library, cards=False)

    def _messages(self, task, seed, extra=""):
        return pfix_messages(task, seed=seed, arm=self.fix_arm, extra=extra)

    def make_judge(self, task, *, seed, meter):
        return Judge(task, seed=seed, library=self.library, meter=meter, properties=True)


class _RoutedArm(_FixArm):
    """F4: one routing call (charged to the episode's budget like any other), then F1's loop
    with F2's sentence when the model named a catalogue rule."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._routes = {}

    def _messages(self, task, seed, extra=""):
        return pfix_messages(task, seed=seed, arm=self.fix_arm, extra=extra,
                             routed=self._routes.get((task.id, as_seed(seed))))

    def _solve(self, meter, judge, task, seed, episode):
        reply = meter(route_messages(task, seed=seed), role="router", max_new_tokens=ROUTE_CAP)
        routed = routing.parse_route(reply)
        self._routes[(task.id, as_seed(seed))] = routed
        episode.extra = dict(routed=routed, route_reply=reply[-400:])
        return self._loop(meter, judge, task, seed, episode)


def _check(arm):
    return type(f"Check{arm}", (_RoutedArm if arm == "F4" else _FixArm,),
                dict(fix_arm=arm, name=f"check_{arm}", feedback="check"))


CheckF0, CheckF1, CheckF2, CheckF3, CheckF4 = (_check(arm) for arm in ARMS)
SCRATCH = (CheckF0,)
LIBRARY = (CheckF1, CheckF2)
DOCUMENTED = (CheckF3,)
ROUTED = (CheckF4,)
EXPLORATORY_CONTRASTS = ((("check_F3", True), ("check_F1", True)),
                         (("check_F3", True), ("check_F2", True)),
                         (("check_F3", True), ("check_F0", False)),
                         (("check_F4", True), ("check_F1", True)),
                         (("check_F4", True), ("check_F2", True)),
                         (("check_F4", True), ("check_F0", False)))
PRIMARY_CONTRASTS = ((("check_F1", True), ("check_F0", False)),
                     (("check_F2", True), ("check_F0", False)),
                     (("check_F2", True), ("check_F1", True)))


def manifest(tasks, *, seeds=(101, 211, 307), budget=BUDGET):
    """Everything a run may not change: arms, prompts, budget, contrasts, interface text."""
    tasks, seeds = tuple(tasks), tuple(seeds)
    packets = [dict(arm=method.fix_arm, method=method.name, library=method.fix_arm != "F0",
                    task_id=task.id, task_sha256=task.sha256, seed=seed,
                    messages_sha256=digest(
                        route_messages(task, seed=seed) if method.fix_arm == "F4" else
                        pfix_messages(task, seed=seed, arm=method.fix_arm)))
               for method in (CheckF0, CheckF1, CheckF2, CheckF3, CheckF4)
               for task in tasks for seed in seeds]
    return dict(version="p-fix-v4", exploratory=True, publication_ready=False,
                design="docs/P_FIX_STUDY_DESIGN_20261118.md",
                arms=dict(F0="no library (= P-adopt L0)", F1="current library interface",
                          F2="F1 + " + POINTER,
                          F3="F1 with the documented interface (amendment 2)",
                          F4="F1 after route(): F2's sentence with the id the model named, "
                             "if any (amendment 3)"),
                visible_checks=VISIBLE_CHECKS,
                route=dict(system=routing.ROUTE_SYSTEM, question=routing.ROUTE_QUESTION,
                           rules_sha256=digest(routing.RULES), output_cap=ROUTE_CAP),
                interface_sha256=digest(interface_text()),
                documented_interface_sha256=digest(interface_text(documented=True)),
                pointer=POINTER,
                output_cap=OUTPUT_CAP, budget=budget.as_dict(), initial_teaching_cards=False,
                feedback="failed check names; a failed run shows its error (Check)",
                primary_contrasts=PRIMARY_CONTRASTS,
                exploratory_contrasts=EXPLORATORY_CONTRASTS, packets=packets)
