"""Run several RSI methods on the same tasks, seeds, budget and model, and grade them.

    from fin_skills import rsi
    from fin_skills.tasks import suite
    report = rsi.benchmark(rsi.METHODS, generate, suite(), seeds=(11, 23, 37),
                           arms=("cards", "cards+library"), out="runs/rsi-bench/episodes.jsonl")
    print(report.table())

An arm is one setting of the skills (rsi.ARMS): "none" (task text only), "cards" (rule cards,
no library), "library" (the fin_skills interface, no cards) or "cards+library". The default
compares the methods with and without the library; the skills ablation runs one method on all
four arms.

For every (method, arm) pair the method is fitted once on the development tasks (a no-op
for methods that do not learn) and then solves every held-out task for every seed. The grade
comes from the judge's hidden grader on the final program and is added to the episode only
after the method has returned. Results are written as they finish, one JSON line per episode,
so an interrupted run keeps everything it completed.

A seed is a synthetic panel (an integer) or "real", the real-price panel of
fin_skills.tasks.real. The report keeps the two apart: every summary row is one method, arm
and data source, so real-panel episodes are never averaged into the synthetic ones.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import random
from typing import Iterable, Sequence

from fin_skills.tasks.data import REAL, sampling_seed

from .meter import Budget, FitBudget
from .methods import ARMS, arm_name


def _arm(row: dict) -> str:
    return row.get("arm") or arm_name(row["library"], row.get("cards", True))


def _data(row: dict) -> str:
    return "real" if row.get("seed") == REAL else "synthetic"


@dataclass
class Report:
    rows: list = field(default_factory=list)
    fits: list = field(default_factory=list)

    def groups(self) -> dict:
        out = {}
        for r in self.rows:
            out.setdefault((r["method"], _arm(r), _data(r)), []).append(r)
        return out

    def summary(self, *, draws: int = 2000, seed: int = 0) -> list[dict]:
        """Per method, arm and data source: valid rate with a task-clustered bootstrap 95%
        interval, accepted rate, accepted-but-wrong rate, library use and mean cost."""
        rng = random.Random(seed)
        table = []
        for (method, arm, data), rows in sorted(self.groups().items()):
            by_task = {}
            for r in rows:
                by_task.setdefault(r["task_id"], []).append(bool(r["valid"]))
            tasks = sorted(by_task)
            rate = lambda ts: (sum(sum(by_task[t]) for t in ts)
                               / max(1, sum(len(by_task[t]) for t in ts)))
            boot = sorted(rate([rng.choice(tasks) for _ in tasks]) for _ in range(draws))
            n = len(rows)
            table.append(dict(
                method=method, arm=arm, data=data, library=ARMS[arm][0], cards=ARMS[arm][1],
                episodes=n, tasks=len(tasks),
                valid=rate(tasks), valid_ci=(boot[int(.025 * draws)], boot[int(.975 * draws) - 1]),
                accepted=sum(r["accepted"] for r in rows) / n,
                accepted_but_wrong=sum(r["accepted"] and not r["valid"] for r in rows) / n,
                library_used=sum(r["library_used"] for r in rows) / n,
                calls=sum(r["usage"]["calls"] for r in rows) / n,
                generated_tokens=sum(r["usage"]["generated_tokens"] for r in rows) / n))
        return table

    def table(self) -> str:
        head = (f"{'method':16s} {'arm':>13s} {'data':>9s} {'n':>4s} {'valid':>6s} {'95% CI':>13s} "
                f"{'acc':>5s} {'acc-wrong':>9s} {'lib-use':>7s} {'calls':>5s} {'tokens':>7s}")
        lines = [head]
        for s in self.summary():
            lo, hi = s["valid_ci"]
            lines.append(f"{s['method']:16s} {s['arm']:>13s} {s['data']:>9s} "
                         f"{s['episodes']:4d} {s['valid']:6.3f} [{lo:5.3f},{hi:5.3f}] "
                         f"{s['accepted']:5.3f} {s['accepted_but_wrong']:9.3f} "
                         f"{s['library_used']:7.3f} {s['calls']:5.2f} "
                         f"{s['generated_tokens']:7.0f}")
        return "\n".join(lines)


def benchmark(methods: Iterable[type], generate, tasks, *,
              seeds: Sequence[int | str] = (11, 23, 37),
              arms: Sequence[str] = ("cards", "cards+library"), budget: Budget = Budget(),
              fit_budget: FitBudget = FitBudget(), fit_seeds: Sequence[int] = (11,),
              heldout=None, out: str | Path | None = None, model_id: str = "") -> Report:
    """Fit each method on tasks.dev, solve tasks.heldout (or the given tasks), grade, report."""
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise ValueError(f"unknown arms {unknown}; known: {list(ARMS)}")
    test_tasks = tuple(heldout if heldout is not None else tasks.heldout)
    dev_ids = {t.id for t in tasks.dev}
    if dev_ids & {t.id for t in test_tasks}:
        raise ValueError("test tasks must not include development tasks")
    path = Path(out) if out is not None else None
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
    report = Report()
    for cls in methods:
        for arm in arms:
            lib, cards = ARMS[arm]
            method = cls(library=lib, cards=cards)
            method.fit(generate, tasks.dev, budget=fit_budget, seeds=fit_seeds,
                       episode_budget=budget)
            report.fits.append(dict(method=method.name, arm=arm, library=lib, cards=cards,
                                    model_id=model_id,
                                    usage=dict(method.fit_usage), state=method.state()))
            for task in test_tasks:
                for seed in seeds:
                    if hasattr(generate, "seed"):
                        generate.seed = sampling_seed(seed)   # for backends that take one
                    episode = method.solve(generate, task, seed=seed, budget=budget)
                    judge = episode.__dict__.pop("_judge")
                    episode.valid = bool(episode.steps) and judge.grade(episode.code)
                    row = dict(episode.as_dict(), model_id=model_id, task_sha256=task.sha256,
                               budget=budget.as_dict())
                    report.rows.append(row)
                    if path is not None:
                        with path.open("a", encoding="utf-8") as fh:
                            fh.write(json.dumps(row, sort_keys=True) + "\n")
    return report
