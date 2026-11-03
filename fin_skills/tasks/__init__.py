"""Graded financial programming tasks for agents, with a judge and a hidden grader.

    from fin_skills.tasks import suite, Judge, messages
    tasks = suite()                          # TaskSet: .dev, .heldout, .get(id)
    task = tasks.heldout[0]
    judge = Judge(task, seed=11, library=True)
    seen = judge.check(code)                 # what a method may see
    judge.grade(code)                        # benchmark runner only

The suite is PROVISIONAL until the benchmark protocol freezes it; the split is by algorithm,
and the four E4 tasks' algorithms are always development tasks.
"""
from .data import ASSETS, panel
from .judge import Judge, Visible, reference
from .prompts import library_positions, library_text, messages, system_prompt, task_prompt
from .spec import SPLITS, Task, TaskSet, assign_splits, suite

__all__ = ["ASSETS", "Judge", "SPLITS", "Task", "TaskSet", "Visible", "assign_splits",
           "library_positions", "library_text", "messages", "panel", "reference", "suite",
           "system_prompt", "task_prompt"]
