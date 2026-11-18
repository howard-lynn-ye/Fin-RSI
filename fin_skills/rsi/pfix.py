"""P-fix: the frozen P track on the library after its agent-interface fix (exploratory).

Design: docs/P_FIX_STUDY_DESIGN_20261118.md. The frozen P study's library arms reached for the
library in 519 programs and 480 of those calls raised; the interface was then changed (PR #77:
keyword parameters, run() on a single input, actionable errors, signal_positions(), an explicit
"the result is final" contract). P-fix asks whether models now use the library and get the
held-out tasks right with it. It reuses P-adopt's machinery and its no-library arm unchanged:

  F0  no library: identical messages to P-adopt L0 (and to the frozen no-library first prompt)
  F1  the current library and its current position-task interface text
  F2  F1 plus P-adopt's one-sentence pointer naming the algorithm that implements the rule

Every arm uses the frozen system prompt (including "Implement the rule exactly"), no rule
cards, names-only Check feedback (an execution error is shown in full, as in the frozen study),
at most 3 attempts and a 4,096-token output cap per reply, on the 6 held-out tasks x 3 seeds.
"""
from fin_skills.tasks.prompts import (SYSTEM, NO_LIBRARY, TIMING, WHAT, WITH_LIBRARY,
                                      library_text, task_prompt)
from fin_skills.tasks.spec import FACTOR_TEXT

from .adopt import BUDGET, OUTPUT_CAP, POINTER
from .run_record import digest
from .study import StudyArm

ARMS = ("F0", "F1", "F2")


def interface_text() -> str:
    """What F1 and F2 append to the system prompt: the current positions-only interface."""
    return library_text(rolling=False, frozen=False)


def pfix_messages(task, *, seed, arm, extra=""):
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    library = arm != "F0"
    system = SYSTEM.format(what=WHAT[task.kind], factors=FACTOR_TEXT[task.factors],
                           timing=TIMING[task.kind])
    system += WITH_LIBRARY if library else NO_LIBRARY
    if library:
        system += "\n\nfin_skills interface:\n" + interface_text()
    messages = [dict(role="system", content=system),
                dict(role="user", content=task_prompt(task, seed=seed))]
    if arm == "F2":
        messages[1]["content"] += "\n\n" + POINTER.format(algorithm=task.algorithm)
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


def _check(arm):
    return type(f"Check{arm}", (_FixArm,), dict(fix_arm=arm, name=f"check_{arm}",
                                                feedback="check"))


CheckF0, CheckF1, CheckF2 = (_check(arm) for arm in ARMS)
SCRATCH = (CheckF0,)
LIBRARY = (CheckF1, CheckF2)
PRIMARY_CONTRASTS = ((("check_F1", True), ("check_F0", False)),
                     (("check_F2", True), ("check_F0", False)),
                     (("check_F2", True), ("check_F1", True)))


def manifest(tasks, *, seeds=(101, 211, 307), budget=BUDGET):
    """Everything a run may not change: arms, prompts, budget, contrasts, interface text."""
    tasks, seeds = tuple(tasks), tuple(seeds)
    packets = [dict(arm=method.fix_arm, method=method.name, library=method.fix_arm != "F0",
                    task_id=task.id, task_sha256=task.sha256, seed=seed,
                    messages_sha256=digest(pfix_messages(task, seed=seed, arm=method.fix_arm)))
               for method in (CheckF0, CheckF1, CheckF2) for task in tasks for seed in seeds]
    return dict(version="p-fix-v1", exploratory=True, publication_ready=False,
                design="docs/P_FIX_STUDY_DESIGN_20261118.md",
                arms=dict(F0="no library (= P-adopt L0)", F1="current library interface",
                          F2="F1 + " + POINTER),
                interface_sha256=digest(interface_text()), pointer=POINTER,
                output_cap=OUTPUT_CAP, budget=budget.as_dict(), initial_teaching_cards=False,
                feedback="failed check names; a failed run shows its error (Check)",
                primary_contrasts=PRIMARY_CONTRASTS, packets=packets)
