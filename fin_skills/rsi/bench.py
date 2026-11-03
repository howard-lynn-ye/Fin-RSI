"""Run several RSI methods on the same tasks, seeds, budget and model, and grade them.

    from fin_skills import rsi
    from fin_skills.tasks import suite
    report = rsi.benchmark(rsi.METHODS, generate, suite(), seeds=(11, 23, 37),
                           library=(False, True), out="runs/rsi-bench/episodes.jsonl")
    print(report.table())

For every (method, library) pair the method is fitted once on the development tasks (a no-op
for methods that do not learn) and then solves every held-out task for every seed. The grade
comes from the judge's hidden grader on the final program and is added to the episode only
after the method has returned. Results are written as they finish, one JSON line per episode,
so an interrupted run keeps everything it completed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import random
from typing import Iterable, Sequence

from .meter import Budget, FitBudget


@dataclass
class Report:
    rows: list = field(default_factory=list)
    fits: list = field(default_factory=list)

    def groups(self) -> dict:
        out = {}
        for r in self.rows:
            out.setdefault((r["method"], r["library"]), []).append(r)
        return out

    def summary(self, *, draws: int = 2000, seed: int = 0) -> list[dict]:
        """Per method and library: valid rate with a task-clustered bootstrap 95% interval,
        accepted rate, accepted-but-wrong rate, library use and mean cost."""
        rng = random.Random(seed)
        table = []
        for (method, library), rows in sorted(self.groups().items()):
            by_task = {}
            for r in rows:
                by_task.setdefault(r["task_id"], []).append(bool(r["valid"]))
            tasks = sorted(by_task)
            rate = lambda ts: (sum(sum(by_task[t]) for t in ts)
                               / max(1, sum(len(by_task[t]) for t in ts)))
            boot = sorted(rate([rng.choice(tasks) for _ in tasks]) for _ in range(draws))
            n = len(rows)
            table.append(dict(
                method=method, library=library, episodes=n, tasks=len(tasks),
                valid=rate(tasks), valid_ci=(boot[int(.025 * draws)], boot[int(.975 * draws) - 1]),
                accepted=sum(r["accepted"] for r in rows) / n,
                accepted_but_wrong=sum(r["accepted"] and not r["valid"] for r in rows) / n,
                library_used=sum(r["library_used"] for r in rows) / n,
                calls=sum(r["usage"]["calls"] for r in rows) / n,
                generated_tokens=sum(r["usage"]["generated_tokens"] for r in rows) / n))
        return table

    def table(self) -> str:
        head = (f"{'method':16s} {'lib':>3s} {'n':>4s} {'valid':>6s} {'95% CI':>13s} "
                f"{'acc':>5s} {'acc-wrong':>9s} {'lib-use':>7s} {'calls':>5s} {'tokens':>7s}")
        lines = [head]
        for s in self.summary():
            lo, hi = s["valid_ci"]
            lines.append(f"{s['method']:16s} {('yes' if s['library'] else 'no'):>3s} "
                         f"{s['episodes']:4d} {s['valid']:6.3f} [{lo:5.3f},{hi:5.3f}] "
                         f"{s['accepted']:5.3f} {s['accepted_but_wrong']:9.3f} "
                         f"{s['library_used']:7.3f} {s['calls']:5.2f} "
                         f"{s['generated_tokens']:7.0f}")
        return "\n".join(lines)


def benchmark(methods: Iterable[type], generate, tasks, *, seeds: Sequence[int] = (11, 23, 37),
              library: Sequence[bool] = (False, True), budget: Budget = Budget(),
              fit_budget: FitBudget = FitBudget(), fit_seeds: Sequence[int] = (11,),
              heldout=None, out: str | Path | None = None, model_id: str = "") -> Report:
    """Fit each method on tasks.dev, solve tasks.heldout (or the given tasks), grade, report."""
    test_tasks = tuple(heldout if heldout is not None else tasks.heldout)
    dev_ids = {t.id for t in tasks.dev}
    if dev_ids & {t.id for t in test_tasks}:
        raise ValueError("test tasks must not include development tasks")
    path = Path(out) if out is not None else None
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
    report = Report()
    for cls in methods:
        for lib in library:
            method = cls(library=lib)
            method.fit(generate, tasks.dev, budget=fit_budget, seeds=fit_seeds,
                       episode_budget=budget)
            report.fits.append(dict(method=method.name, library=lib, model_id=model_id,
                                    usage=dict(method.fit_usage), state=method.state()))
            for task in test_tasks:
                for seed in seeds:
                    if hasattr(generate, "seed"):
                        generate.seed = seed          # sampling seed for backends that take one
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
