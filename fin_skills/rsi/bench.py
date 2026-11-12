"""Run several RSI methods on the same tasks, seeds, budget and model, and grade them.

    from fin_skills import rsi
    from fin_skills.tasks import suite
    report = rsi.benchmark(rsi.METHODS, generate, suite(), seeds=(11, 23, 37),
                           arms=("cards", "cards+library"), out="runs/rsi-bench/episodes.jsonl")
    print(report.table())

A condition is either an arm (rsi.ARMS): "none" (task text only), "cards" (rule cards, no
library), "library" (the fin_skills interface, no cards) or "cards+library"; or, with
library=(False, True), only the library switch, each method keeping its own prompt (the frozen
study and its follow-ups run this way). Give arms or library, not both; the default compares
"cards" with "cards+library".

For every (method, condition) pair the method is fitted once on the development tasks (a no-op
for methods that do not learn) and then solves every held-out task for every seed. The grade
comes from the judge's hidden grader on the final program and is added to the episode only
after the method has returned. Results are written as they finish, one JSON line per episode,
so an interrupted run keeps everything it completed.

A seed is a synthetic panel (an integer) or "real", the real-price panel of
fin_skills.tasks.real. The report keeps the two apart: every summary row is one method,
condition and data source, so real-panel episodes are never averaged into the synthetic ones.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import itertools
import os
from pathlib import Path
import platform
from importlib.metadata import version
import random
from typing import Iterable, Sequence

from fin_skills.tasks.data import REAL, sampling_seed

from .meter import Budget, FitBudget, EVENT_SINK, BACKEND_BINDING, event
from .run_record import RunRecord, digest, source_hashes, EXECUTION_BUNDLE
from .backend import describe_backend
from .methods import ARMS, arm_name


def _arm(row: dict) -> str:
    return row.get("arm") or arm_name(row["library"], row.get("cards", True))


def _data(row: dict) -> str:
    return "real" if row.get("seed") == REAL else "synthetic"


@dataclass
class Report:
    rows: list = field(default_factory=list)
    fits: list = field(default_factory=list)
    complete: bool = False
    protocol: dict = field(default_factory=dict)

    def condition(self, row: dict):
        """The row's condition: its arm in an arms run, its library switch otherwise."""
        return _arm(row) if self.protocol.get("conditions") == "arms" else row["library"]

    def _validate(self):
        if not self.complete:
            raise ValueError("incomplete benchmark: no model-quality summary is available")
        keys = [(r["method"], self.condition(r), r["task_id"], r["seed"]) for r in self.rows]
        planned = [tuple(k) for k in self.protocol.get("plan", [])]
        if len(keys) != len(set(keys)) or set(keys) != set(planned):
            raise ValueError("benchmark rows do not match the frozen complete plan")
        if any(type(r.get("valid")) is not bool for r in self.rows):
            raise ValueError("ungraded or infrastructure-failed episodes cannot be scored")
        if "test_tasks" in self.protocol:
            expected = {t["id"]: digest(t) for t in self.protocol["test_tasks"]}
            if any(r["task_sha256"] != expected[r["task_id"]] for r in self.rows):
                raise ValueError("result task identity differs from frozen protocol")

    def groups(self) -> dict:
        """(method, condition, data source) -> rows."""
        out = {}
        for r in self.rows:
            out.setdefault((r["method"], self.condition(r), _data(r)), []).append(r)
        return out

    def summary(self, *, draws: int = 2000, seed: int = 0) -> list[dict]:
        """Per method, condition and data source: valid rate with an algorithm-clustered
        bootstrap interval, accepted rate, accepted-but-wrong rate, library use and mean cost."""
        self._validate()
        if isinstance(draws, bool) or not isinstance(draws, int) or draws < 40:
            raise ValueError("draws must be an integer >= 40")
        rng = random.Random(seed)
        table = []
        for (method, condition, data), rows in sorted(self.groups().items(),
                                                       key=lambda kv: tuple(map(str, kv[0]))):
            by_task = {}
            for r in rows:
                by_task.setdefault(r["task_algorithm"], []).append(r["valid"])
            tasks = sorted(by_task)
            rate = lambda ts: (sum(sum(by_task[t]) for t in ts)
                               / max(1, sum(len(by_task[t]) for t in ts)))
            boot = sorted(rate([rng.choice(tasks) for _ in tasks]) for _ in range(draws))
            n = len(rows)
            table.append(dict(
                method=method, condition=condition, arm=_arm(rows[0]), data=data,
                library=rows[0]["library"], cards=rows[0].get("cards", True), episodes=n,
                tasks=len({r["task_id"] for r in rows}), algorithm_clusters=len(tasks),
                valid=rate(tasks), valid_ci=((boot[int(.025 * draws)], boot[int(.975 * draws) - 1])
                                            if len(tasks) > 1 else None),
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
            interval = s["valid_ci"]
            ci = f"[{interval[0]:5.3f},{interval[1]:5.3f}]" if interval else "not estimable"
            lines.append(f"{s['method']:16s} {s['arm']:>13s} {s['data']:>9s} "
                         f"{s['episodes']:4d} {s['valid']:6.3f} {ci} "
                         f"{s['accepted']:5.3f} {s['accepted_but_wrong']:9.3f} "
                         f"{s['library_used']:7.3f} {s['calls']:5.2f} "
                         f"{s['generated_tokens']:7.0f}")
        return "\n".join(lines)

    def contrasts(self, *, draws=2000, seed=0) -> list[dict]:
        """Frozen paired contrasts, resampling whole algorithms, with Holm adjustment.

        The two-sided sign-flip p-value assumes exchangeable signs of independent algorithm
        effects. It is descriptive for this public suite, not evidence of independent custody.
        """
        self._validate()
        if not isinstance(draws, int) or draws < 40:
            raise ValueError("draws must be >= 40")
        rng, results = random.Random(seed), []
        groups = self.groups()
        for left, right in self.protocol.get("contrasts", []):
            a, b = (groups[(*tuple(side), "synthetic")] for side in (left, right))
            indexed = lambda rows: {(r["task_sha256"], r["seed"]): r for r in rows}
            aa, bb = indexed(a), indexed(b)
            if set(aa) != set(bb):
                raise ValueError("paired contrast has missing or mismatched tasks/seeds")
            clusters = {}
            for key in aa:
                row = aa[key]
                clusters.setdefault(row["task_algorithm"], []).append(
                    int(row["valid"]) - int(bb[key]["valid"]))
            effects = [sum(v) / len(v) for _, v in sorted(clusters.items())]
            mean = sum(effects) / len(effects)
            interval = pvalue = None
            if len(effects) > 1:
                boot = sorted(sum(rng.choice(effects) for _ in effects) / len(effects)
                              for _ in range(draws))
                interval = (boot[int(.025 * draws)], boot[int(.975 * draws) - 1])
                if len(effects) <= 12:
                    signs = itertools.product((-1, 1), repeat=len(effects))
                    values = [abs(sum(x*s for x, s in zip(effects, pattern)) / len(effects))
                              for pattern in signs]
                    pvalue = sum(v >= abs(mean) - 1e-12 for v in values) / len(values)
                else:
                    values = [abs(sum(x*rng.choice((-1, 1)) for x in effects) / len(effects))
                              for _ in range(draws)]
                    pvalue = (1 + sum(v >= abs(mean) - 1e-12 for v in values)) / (draws + 1)
            results.append(dict(left=left, right=right, effect=mean, paired_ci=interval,
                                algorithm_clusters=len(effects), pairs=len(aa),
                                p_sign_flip=pvalue, p_holm=None,
                                interpretation="descriptive public regression; algorithm-weighted"))
        ordered = sorted((r for r in results if r["p_sign_flip"] is not None),
                         key=lambda r: r["p_sign_flip"])
        previous = 0.0
        for i, row in enumerate(ordered):
            previous = min(1.0, max(previous, (len(ordered)-i)*row["p_sign_flip"]))
            row["p_holm"] = previous
        return results


def benchmark(methods: Iterable[type], generate, tasks, *,
              seeds: Sequence[int | str] = (11, 23, 37), arms: Sequence[str] | None = None,
              library: Sequence[bool] | None = None, budget: Budget = Budget(),
              fit_budget: FitBudget = FitBudget(), fit_seeds: Sequence[int] = (11,),
              heldout=None, out: str | Path | None = None, model_id: str = "",
              require_bounded_backend: bool = False, primary_contrasts=(),
              execution_mode: str = "trusted-fixtures", study_manifest=None) -> Report:
    """Fit each method on tasks.dev, solve tasks.heldout (or the given tasks), grade, report."""
    if arms is not None and library is not None:
        raise ValueError("give arms or library, not both")
    if library is not None:                  # each method keeps its own prompt; label = switch
        conditions = [(bool(lib), dict(library=bool(lib))) for lib in library]
    else:
        arms = ("cards", "cards+library") if arms is None else tuple(arms)
        unknown = [a for a in arms if a not in ARMS]
        if unknown:
            raise ValueError(f"unknown arms {unknown}; known: {list(ARMS)}")
        conditions = [(a, dict(library=ARMS[a][0], cards=ARMS[a][1])) for a in arms]
    methods, seeds = tuple(methods), tuple(seeds)
    labels = [label for label, _ in conditions]
    fit_seeds = tuple(fit_seeds)
    if execution_mode not in ("trusted-fixtures", "model"):
        raise ValueError("execution_mode must be trusted-fixtures or model")
    if execution_mode == "model":
        if not out or not model_id.strip():
            raise ValueError("model execution requires durable output and a model identifier")
        if (os.environ.get("FIN_TASKS_REQUIRE_CONFINEMENT") != "1" or
                os.environ.get("FIN_TASKS_CONFINE") !=
                "benchmarks.agent_study.linux_sandbox:confine"):
            raise ValueError("model execution requires the qualified Linux confinement hook")
        from benchmarks.agent_study.linux_sandbox import confinement_available
        if not confinement_available():
            raise ValueError("required Linux confinement is unavailable")
        require_bounded_backend = True
    if len(fit_seeds) != 1:
        raise ValueError("current learners require exactly one fit seed; multiplicity is not implicit")
    if require_bounded_backend and not callable(getattr(generate, "generate_bounded", None)):
        raise ValueError("backend must implement generate_bounded(messages, max_new_tokens, seed)")
    test_tasks = tuple(heldout if heldout is not None else tasks.heldout)
    if any(t.sha256 != digest(t.manifest()) for t in (*tasks.dev, *test_tasks)):
        raise ValueError("task identity must hash its complete canonical manifest")
    if not methods or not seeds or not conditions or not test_tasks or not fit_seeds:
        raise ValueError("benchmark dimensions must be nonempty")
    if (len({c.name for c in methods}) != len(methods) or len(set(seeds)) != len(seeds)
            or len(set(labels)) != len(labels)
            or len({t.id for t in test_tasks}) != len(test_tasks)):
        raise ValueError("duplicate benchmark dimensions")
    dev_ids = {t.id for t in tasks.dev}
    if dev_ids & {t.id for t in test_tasks}:
        raise ValueError("test tasks must not include development tasks")
    if {t.algorithm for t in tasks.dev} & {t.algorithm for t in test_tasks}:
        raise ValueError("development and test algorithms must be disjoint")
    group_keys = {(c.name, label) for c in methods for label in labels}
    backend = describe_backend(generate, strict=execution_mode == "model")
    if execution_mode == "model" and backend["configuration"]["model_id"] != model_id:
        raise ValueError("backend model identity differs from requested model_id")
    for left, right in primary_contrasts:
        if tuple(left) not in group_keys or tuple(right) not in group_keys or left == right:
            raise ValueError("contrast must name two different planned groups")
    path = Path(out) if out is not None else None
    protocol = dict(schema_version=1, model_id=model_id, budget=budget.as_dict(),
                    study_manifest=study_manifest,
                    fit_budget=vars(fit_budget), fit_seeds=list(fit_seeds),
                    dev_tasks=[t.manifest() for t in tasks.dev],
                    test_tasks=[t.manifest() for t in test_tasks], source_hashes=source_hashes(),
                    exposure="public-regression; heldout from fit only; no independent custody",
                    publication_ready=False, require_bounded_backend=require_bounded_backend,
                    backend=backend,
                    execution_mode=execution_mode,
                    runtime=dict(python=platform.python_version(), system=platform.system(),
                                 confinement_hook=os.environ.get("FIN_TASKS_CONFINE", ""),
                                 required=os.environ.get("FIN_TASKS_REQUIRE_CONFINEMENT") == "1",
                                 packages={p: version(p) for p in ("numpy", "pandas", "scipy")}),
                    contrasts=list(primary_contrasts),
                    plan=[[c.name, label, t.id, seed] for c in methods for label in labels
                          for t in test_tasks for seed in seeds])
    if library is None:
        protocol["conditions"] = "arms"
    record = RunRecord(path, protocol)
    report = Report(protocol=protocol)
    active = None
    sink_token = EVENT_SINK.set(record.event)
    binding_token = BACKEND_BINDING.set((generate, backend) if execution_mode == "model" else None)
    bundle_token = EXECUTION_BUNDLE.set(record.bundle)
    try:
        for cls in methods:
            for label, options in conditions:
                lib = options["library"]
                active = dict(method=cls.name, library=lib, phase="fit")
                if library is None:
                    active["arm"] = label
                method = cls(**options)
                if hasattr(generate, "seed"):
                    generate.seed = fit_seeds[0]
                event("fit_started", **active, data_seed=fit_seeds[0])
                method.fit(generate, tasks.dev, budget=fit_budget, seeds=fit_seeds,
                           episode_budget=budget)
                if [t.manifest() for t in tasks.dev] != protocol["dev_tasks"]:
                    raise ValueError("development task changed during fit")
                fitted = dict(method=method.name, library=lib, model_id=model_id,
                              usage=dict(method.fit_usage), state=method.state())
                if library is None:
                    fitted.update(arm=label, cards=options["cards"])
                fitted["state_sha256"] = digest(fitted["state"])
                report.fits.append(fitted)
                record.state("running", fits=report.fits, active=active)
                for task in test_tasks:
                    for seed in seeds:
                        expected = next(t for t in protocol["test_tasks"] if t["id"] == task.id)
                        if task.manifest() != expected:
                            raise ValueError("task changed after protocol freeze")
                        active = dict(method=cls.name, library=lib, task_id=task.id,
                                      seed=seed, phase="solve")
                        if library is None:
                            active["arm"] = label
                        if hasattr(generate, "seed"):
                            generate.seed = sampling_seed(seed)   # for backends that take one
                        event("episode_started", **active, task_sha256=task.sha256)
                        episode = method.solve(generate, task, seed=seed, budget=budget)
                        if task.manifest() != expected:
                            raise ValueError("task changed during solve")
                        if digest(method.state()) != fitted["state_sha256"]:
                            raise ValueError("fitted method state changed during evaluation")
                        judge = episode.__dict__.pop("_judge")
                        episode.valid = bool(episode.steps) and judge.grade_attempt(
                            episode.steps[-1].attempt_index)
                        first_valid = bool(episode.steps) and judge.grade_attempt(
                            episode.steps[0].attempt_index)
                        row = dict(episode.as_dict(), model_id=model_id,
                                   first_valid=first_valid,
                                   task_algorithm=task.algorithm, task_sha256=task.sha256,
                                   protocol_sha256=record.protocol_sha256, budget=budget.as_dict())
                        report.rows.append(row)
                        record.append(row)
        report.complete = True
        if ([t.manifest() for t in tasks.dev] != protocol["dev_tasks"] or
                [t.manifest() for t in test_tasks] != protocol["test_tasks"]):
            raise ValueError("task contracts changed before completion")
        if source_hashes() != protocol["source_hashes"]:
            raise ValueError("source changed during benchmark execution")
        report._validate()
        record.state("complete", fits=report.fits)
    except BaseException as exc:
        record.state("invalid", fits=report.fits, active=active,
                     error=dict(type=type(exc).__name__, message=str(exc)[:2000]))
        raise
    finally:
        EVENT_SINK.reset(sink_token)
        BACKEND_BINDING.reset(binding_token)
        EXECUTION_BUNDLE.reset(bundle_token)
    return report
