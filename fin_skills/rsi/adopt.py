"""P-adopt: how the library is presented, on the frozen study's position tasks (exploratory).

Design: docs/P_ADOPTION_STUDY_DESIGN_20261112.md. Designed after the frozen study's results were
seen, so it is a new exploratory protocol, not a confirmation. Every arm uses the frozen study's
first prompt (``study.initial_messages``) and the Check feedback (failed check names only, no
rule cards); the arms differ only in what follows the task text:

  L0  no library (``import fin_skills`` refused)
  L1  the library interface, as in the frozen study
  L2  L1 plus one sentence naming the library algorithm that implements the rule
  L3  L1 plus the catalogue entry a fixed BM25 retriever ranks first for the rule text,
      over the 12 algorithms positions() accepts. The retriever is deterministic, so which
      tasks get a correct entry is known before any run (``retrieval_table``).

PlaybookLearning is re-run on the same prompts (L0 and L1): no rule cards in the first prompt,
names-only Check feedback, and development traces without library hints in either condition.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from .methods import PlaybookLearning
from .meter import Budget
from .run_record import digest
from .study import StudyArm, initial_messages

ARMS = ("L0", "L1", "L2", "L3")
POINTER = 'The fin_skills library implements this rule as positions("{algorithm}", ...).'
RETRIEVED = "Library entry retrieved for this rule by text search:\n{entry}"
STOP = set("a an the of to and or in on at by for is be it its as with from that this when each "
           "every over than then into not no are was were has have which where while".split())


def _tokens(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP and not w.isdigit()]


def _document(entry):
    return " ".join([entry["id"].replace("_", " "), entry["name"], entry.get("caveat", ""),
                     " ".join(entry.get("tags", [])), " ".join(entry.get("objectives", [])),
                     entry.get("task", ""), entry.get("skill", "").replace("-", " ")])


def retrieval_pool():
    """Catalogue entries of the algorithms positions() accepts, as listed in the P interface."""
    from fin_skills.algorithms import catalog
    from fin_skills.tasks.prompts import DEFAULTS
    entries = {c["id"]: c for c in catalog() if c["id"] in DEFAULTS}
    if set(entries) != set(DEFAULTS):
        raise ValueError("retrieval pool differs from the positions() algorithms in the interface")
    return entries


def bm25_ranking(query, entries, k1=1.5, b=0.75):
    """Deterministic BM25 over catalogue entries; ties broken by algorithm id."""
    docs = {key: _tokens(_document(entry)) for key, entry in entries.items()}
    average = sum(len(d) for d in docs.values()) / len(docs)
    frequency = Counter(w for d in docs.values() for w in set(d))
    scores = {}
    for key, doc in docs.items():
        counts, score = Counter(doc), 0.0
        for word in _tokens(query):
            if counts[word]:
                idf = math.log(1 + (len(docs) - frequency[word] + .5) / (frequency[word] + .5))
                score += idf * counts[word] * (k1 + 1) / (
                    counts[word] + k1 * (1 - b + b * len(doc) / average))
        scores[key] = score
    return sorted(scores, key=lambda key: (-scores[key], key))


def entry_text(entry):
    from fin_skills.tasks.prompts import DEFAULTS
    shown = ", ".join(f"{k}={v!r}" for k, v in DEFAULTS[entry["id"]].items()) or "no parameters"
    return (f'  {entry["id"]}: {entry["name"]} ({entry["task"]}; defaults {shown}). '
            f'{entry.get("caveat", "")}').rstrip()


def retrieved(task):
    entries = retrieval_pool()
    return entries[bm25_ranking(task.definition, entries)[0]]


def adopt_messages(task, *, seed, arm, extra=""):
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    messages = initial_messages(task, seed=seed, library=arm != "L0")
    if arm == "L2":
        messages[1]["content"] += "\n\n" + POINTER.format(algorithm=task.algorithm)
    elif arm == "L3":
        messages[1]["content"] += "\n\n" + RETRIEVED.format(entry=entry_text(retrieved(task)))
    if extra:
        messages[0]["content"] += "\n\n" + extra
    return messages


class _Arm:
    """Fixes each arm's library condition so a run cannot pair an arm with the other one.
    No arm shows rule cards (cards=False), so episodes record the skills arm "none" or
    "library"; adopt_arm is this study's own label (L0-L3)."""
    adopt_arm = None

    def __init__(self, *, library: bool = False, cards: bool = False):
        if bool(library) != (self.adopt_arm != "L0"):
            raise ValueError(f"arm {self.adopt_arm} runs only with "
                             f"library={self.adopt_arm != 'L0'}")
        if cards:
            raise ValueError("P-adopt arms show no rule cards")
        super().__init__(library=library, cards=False)

    def _messages(self, task, seed, extra=""):
        return adopt_messages(task, seed=seed, arm=self.adopt_arm, extra=extra)

    def _check_feedback(self, seen):
        return StudyArm._check_feedback(self, seen)


def _check(arm):
    return type(f"Check{arm}", (_Arm, StudyArm),
                dict(adopt_arm=arm, name=f"check_{arm}", feedback="check"))


def _playbook(arm):
    return type(f"Playbook{arm}", (_Arm, PlaybookLearning),
                dict(adopt_arm=arm, name=f"playbook_{arm}", feedback="check",
                     trace_tools=False, trace_rules=True))


CheckL0, CheckL1, CheckL2, CheckL3 = (_check(arm) for arm in ARMS)
PlaybookL0, PlaybookL1 = _playbook("L0"), _playbook("L1")
SCRATCH = (CheckL0, PlaybookL0)                          # run with library=(False,)
LIBRARY = (CheckL1, CheckL2, CheckL3, PlaybookL1)        # run with library=(True,)
PRIMARY_CONTRASTS = ((("check_L2", True), ("check_L1", True)),
                     (("check_L3", True), ("check_L1", True)),
                     (("check_L1", True), ("check_L0", False)))
BUDGET = Budget(attempts=3, calls=6, generated_tokens=3 * 4096)
OUTPUT_CAP = 4096


def retrieval_table(tasks):
    entries = retrieval_pool()
    return [dict(task_id=t.id, algorithm=t.algorithm,
                 top3=bm25_ranking(t.definition, entries)[:3],
                 rank=bm25_ranking(t.definition, entries).index(t.algorithm) + 1)
            for t in tasks]


def manifest(tasks, *, seeds=(101, 211, 307), budget=BUDGET):
    """Everything a run may not change: arms, prompts, retrieval results, budget, contrasts."""
    tasks, seeds = tuple(tasks), tuple(seeds)
    packets = [dict(arm=method.adopt_arm, method=method.name,
                    library=method.adopt_arm != "L0",
                    task_id=task.id, task_sha256=task.sha256, seed=seed,
                    messages_sha256=digest(adopt_messages(task, seed=seed,
                                                          arm=method.adopt_arm)))
               for method in (CheckL0, CheckL1, CheckL2, CheckL3)
               for task in tasks for seed in seeds]
    return dict(version="p-adopt-v1", exploratory=True, publication_ready=False,
                design="docs/P_ADOPTION_STUDY_DESIGN_20261112.md",
                arms=dict(L0="no library", L1="library interface", L2="L1 + " + POINTER,
                          L3="L1 + BM25 top-1 of the 12 positions() algorithms"),
                pointer=POINTER, retrieved_prefix=RETRIEVED.split("\n")[0],
                retrieval=retrieval_table(tasks), output_cap=OUTPUT_CAP,
                budget=budget.as_dict(), initial_teaching_cards=False,
                feedback="failed check names only (Check)",
                learners=dict(playbook_L0="PlaybookLearning on L0 prompts",
                              playbook_L1="PlaybookLearning on L1 prompts; traces without "
                                          "library hints"),
                primary_contrasts=PRIMARY_CONTRASTS, packets=packets)
