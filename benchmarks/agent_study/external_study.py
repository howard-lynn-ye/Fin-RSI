"""Matched one-shot scalar code delivery on publicly exposed external questions.

This secondary experiment has two library conditions. Finite/executable acceptance cannot
detect a numerically wrong answer, so it is not presented as the six-arm feedback study.
The same budget meter, execution boundary, run records and post-selection scorer are used.
"""
from dataclasses import dataclass
import inspect
import json
from types import SimpleNamespace

from fin_skills.rsi.bench import benchmark
from fin_skills.rsi.methods import BaseRSI
from fin_skills.rsi.meter import Budget, event
from fin_skills.rsi.run_record import digest
from fin_skills.tasks.execution import evaluate_scalar
from fin_skills.tasks.judge import EvaluationInfrastructureError, Visible
from .external_packets import ScalarPolicy, digest as public_digest, numeric_reference, prompt


@dataclass(frozen=True)
class ScalarTask:
    public: dict
    scoring: dict

    def __post_init__(self):
        prompt(self.public)  # Validate the public schema before any model sees it.
        if self.scoring["public_sha256"] != public_digest(self.public):
            raise ValueError("scoring record is bound to a different public packet")
        numeric_reference(self.scoring["original_answer"])
        ScalarPolicy(**self.scoring["scoring_policy"])

    @property
    def id(self):
        return self.public["id"]

    @property
    def algorithm(self):
        return self.public["source_cluster"]

    def manifest(self):
        self.__post_init__()
        return json.loads(json.dumps(dict(id=self.id, algorithm=self.algorithm,
            family="external_scalar", public=self.public, reference_sha256=digest(self.scoring),
            scoring_policy=self.scoring["scoring_policy"]), sort_keys=True, allow_nan=False))

    @property
    def sha256(self):
        return digest(self.manifest())


class ScalarJudge:
    def __init__(self, task, *, seed, library, meter):
        self.task, self.seed, self.library, self.meter = task, seed, library, meter
        self.attempts = []

    def check(self, code):
        self.meter.charge_attempt()
        event("program_started", code=code, task_sha256=self.task.sha256, data_seed=self.seed)
        result = evaluate_scalar(code, library=self.library)
        event("program_completed", task_sha256=self.task.sha256, result=result)
        if result.get("infrastructure_error"):
            raise EvaluationInfrastructureError(str(result.get("error"))[:2000])
        seen = Visible(code, bool(result["executed"]), str(result.get("error", ""))[:2000],
                       checks={"finite_scalar": bool(result["executed"])},
                       library_calls=result.get("library_calls", 0),
                       attempt_index=len(self.attempts))
        self.attempts.append((seen, result.get("value")))
        return seen

    def grade_attempt(self, index):
        if type(index) is not int or not 0 <= index < len(self.attempts):
            raise ValueError("unknown scalar attempt")
        seen, value = self.attempts[index]
        policy = ScalarPolicy(**self.task.scoring["scoring_policy"])
        return seen.accepted and policy.correct(value, self.task.scoring["original_answer"])


def interface_text():
    from fin_skills import algorithms
    return (f"from fin_skills import algorithms\nalgorithms.run{inspect.signature(algorithms.run)}\n"
            f"algorithms.catalog{inspect.signature(algorithms.catalog)}\n"
            "Use NumPy/pandas/SciPy directly where the task has no matching library method.\n"
            "Available algorithm metadata (runtime status is not accuracy evidence):\n" +
            json.dumps(algorithms.catalog(), sort_keys=True, ensure_ascii=False))


class ExternalOneShot(BaseRSI):
    name = "external_code"
    interface_snapshot = None

    def state(self):
        return {"interface_sha256": digest(self.interface_snapshot)}

    def make_judge(self, task, *, seed, meter):
        return ScalarJudge(task, seed=seed, library=self.library, meter=meter)

    def _messages(self, task, seed, extra=""):
        if extra:
            raise ValueError("no learned prompt addenda in the external one-shot study")
        if self.interface_snapshot is None:
            raise ValueError("external interface must be frozen before evaluation")
        return prompt(task.public, interface_text=self.interface_snapshot if self.library else "")

    def _solve(self, meter, judge, task, seed, episode):
        _, seen = self._attempt(meter, judge, self._messages(task, seed), episode)
        episode.status = "accepted" if seen.accepted else "exhausted"


def run(generate, tasks, *, out, model_id, seeds, execution_mode="model"):
    tasks = tuple(tasks)
    if not tasks:
        raise ValueError("external evaluation task scope must be nonempty")
    if any(t.scoring["scoring_policy"][k] != 0 for t in tasks
           for k in ("absolute_tolerance", "relative_tolerance")):
        raise ValueError("external primary endpoint requires frozen exact numeric matching")
    if len({t.public["source"]["split"] for t in tasks}) != 1:
        raise ValueError("development and evaluation source splits must stay separate")
    budget = Budget(attempts=1, calls=1, generated_tokens=2048)
    interface = interface_text()
    class FrozenExternal(ExternalOneShot):
        interface_snapshot = interface
    manifest = dict(schema="external-one-shot-v1", primary=False, publication_ready=False,
        exposure="public external tasks; training contamination unknown; not official leaderboard",
        role="development" if tasks[0].public["source"]["split"] == "train" else "evaluation",
        selection="one generation per task/library/seed; no numeric feedback or retries",
        endpoint="finite executable scalar AND exact numeric match to public source answer",
        library_intervention="full numerical library plus catalog/interface documentation",
        library_prompt_sha256=digest(interface), budget=budget.as_dict())
    return benchmark([FrozenExternal], generate, SimpleNamespace(dev=(), heldout=tasks),
        seeds=seeds, library=(False, True), budget=budget, out=out, model_id=model_id,
        primary_contrasts=((("external_code", True), ("external_code", False)),),
        execution_mode=execution_mode, study_manifest=manifest)
