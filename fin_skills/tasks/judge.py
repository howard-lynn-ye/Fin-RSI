"""The judge a method sees, and the grader it never sees.

    judge = Judge(task, seed=11, library=True)
    seen = judge.check(code)        # Visible: ran?, error, checks, feedback text. No grade.
    judge.grade(code)               # the benchmark runner only: matches the reference?

check() charges one attempt to the method's Meter when one is attached, so every method gets
the same number of program executions. grade() re-uses the positions of an already checked
program; it is never passed to a method.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from fin_skills.api.rules import CAUSES, explain
from .data import panel
from .references import REFERENCES
from .spec import Task

LAG_KS = (32, 64)
TIMEOUT = 120


@dataclass(frozen=True)
class Visible:
    """What a method may see about one execution."""

    code: str
    executed: bool
    error: str = ""
    checks: dict = field(default_factory=dict)
    library_calls: int = 0

    @property
    def accepted(self) -> bool:
        return self.executed and bool(self.checks) and all(self.checks.values())

    @property
    def failed(self) -> list[str]:
        return [name for name, ok in self.checks.items() if not ok] if self.executed else []

    def error_text(self) -> str:
        return self.error

    def check_text(self, *, tools: bool) -> str:
        """Failed checks with their causes and rule cards (cards carry library hints only
        when tools is True). Empty when the program ran and passed."""
        known = [n for n in self.failed if n in CAUSES]
        if not known:
            return ""
        return explain(known, tools=tools)


def reference(task: Task, seed: int) -> np.ndarray:
    _, _, adjusted = panel(seed, rows=task.rows, factors=task.factors, order=task.order)
    return REFERENCES[task.algorithm](adjusted.to_numpy(), **task.parameters)


def _run_worker(code: str, request: dict, timeout: int) -> dict:
    with tempfile.TemporaryDirectory(prefix="fin-task-") as td:
        box = Path(td)
        (box / "submission.py").write_text(code)
        (box / "request.json").write_text(json.dumps(request))
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", PYTHONHASHSEED="0")
        try:
            proc = subprocess.run([sys.executable, "-B", "-m", "fin_skills.tasks.worker",
                                   str(box)], capture_output=True, text=True, timeout=timeout,
                                  env=env)
        except subprocess.TimeoutExpired:
            return dict(executed=False, error=dict(type="TimeoutExpired",
                                                   message=f"{timeout} s limit"))
        if not (box / "result.json").exists():
            return dict(executed=False, infrastructure_error=True,
                        error=dict(type="WorkerCrash", message=f"returncode {proc.returncode}",
                                   traceback=proc.stderr[-3000:]))
        return json.loads((box / "result.json").read_text())


class Judge:
    def __init__(self, task: Task, *, seed: int, library: bool, meter=None,
                 lag_ks=LAG_KS, timeout: int = TIMEOUT):
        self.task, self.seed, self.library = task, int(seed), bool(library)
        self.meter, self.lag_ks, self.timeout = meter, tuple(lag_ks), timeout
        self.records: list[dict] = []
        self._positions: dict[str, list] = {}

    def check(self, code: str) -> Visible:
        """One attempt: charged to the meter, recorded, and returned without any grade."""
        if self.meter is not None:
            self.meter.charge_attempt()
        return self._execute(code)

    def _execute(self, code: str) -> Visible:
        request = dict(task_id=self.task.id, seed=self.seed, library=self.library,
                       lag_ks=list(self.lag_ks))
        result = _run_worker(code, request, self.timeout)
        key = hashlib.sha256(code.encode()).hexdigest()
        if result.get("executed"):
            self._positions[key] = result.pop("positions")
        error = result.get("error") or {}
        text = ""
        if error:
            text = f"{error.get('type', 'Error')}: {error.get('message', '')}"
            if error.get("traceback"):
                text += "\n" + error["traceback"][-1500:]
        seen = Visible(code=code, executed=bool(result.get("executed")), error=text,
                       checks=dict(result.get("checks") or {}),
                       library_calls=int(result.get("library_calls") or 0))
        self.records.append(dict(sha256=key, executed=seen.executed, checks=seen.checks,
                                 error=text[:400], library_calls=seen.library_calls,
                                 trace=result.get("trace", []),
                                 infrastructure_error=bool(result.get("infrastructure_error"))))
        return seen

    def grade(self, code: str) -> bool:
        """Valid: ran, passed every visible check, and matches the independent reference."""
        key = hashlib.sha256(code.encode()).hexdigest()
        if not any(r["sha256"] == key for r in self.records):
            self._execute(code)                 # grading is not one of the method's attempts
        if key not in self._positions:
            return False
        record = next(r for r in reversed(self.records) if r["sha256"] == key)
        if not (record["executed"] and record["checks"] and all(record["checks"].values())):
            return False
        got = np.asarray(self._positions[key], dtype=float)
        want = reference(self.task, self.seed)
        return got.shape == want.shape and bool(np.allclose(got, want, atol=self.task.tolerance,
                                                            rtol=0))
