"""Run one submitted program on one task in its own process.

    python -m fin_skills.tasks.worker BOX

BOX holds submission.py and request.json ({task_id, seed, library, lag_ks}). The worker
rebuilds the task's data, calls solve() with fresh copies, runs the visible checks and writes
result.json: executed, error, checks, library trace and the positions themselves. Grading
against the reference happens in the parent process and never reaches the model.

This is NOT a security sandbox by itself. It caps address space and output, and the parent
enforces a timeout. Set FIN_TASKS_CONFINE="module:function" to apply a confinement hook
(for example benchmarks.agent_study.linux_sandbox:confine on Linux) before the program runs;
the hook receives (box, readonly_paths) and returns a JSON-serializable description.
FIN_TASKS_READONLY (os.pathsep-separated) adds read-only roots to that list.
"""
from __future__ import annotations

import builtins
import importlib
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback

OUTPUT_CAP = 65536
ADDRESS_SPACE = 16 << 30


class _Capped(io.TextIOBase):
    def __init__(self):
        self.size = 0

    def writable(self):
        return True

    def write(self, text):
        self.size += len(text)
        return len(text)


def _error(exc: BaseException) -> dict:
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return dict(type=type(exc).__name__, message=str(exc)[:2000], traceback=tb[-3000:])


def _guarded_import(blocked: str):
    real = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and (name == blocked or name.startswith(blocked + ".")):
            raise ImportError(f"{blocked} is not available for this task")
        return real(name, globals, locals, fromlist, level)
    return guarded


def main(box: Path) -> None:
    import numpy as np
    import pandas as pd
    import scipy.optimize  # noqa: F401  (import before any confinement)
    import fin_skills.algorithms as algorithms
    import fin_skills.algorithms.panel as panel
    from fin_skills.api import get
    from fin_skills.tasks.data import panel as make_panel
    from fin_skills.tasks.spec import suite

    request = json.loads((box / "request.json").read_text())
    task = suite().get(request["task_id"])
    raw, factors, adjusted = make_panel(int(request["seed"]), rows=task.rows,
                                        factors=task.factors, order=task.order)
    lag_guard = get("execution_lag")          # bound before the program can touch fin_skills
    source = (box / "submission.py").read_text()
    trace = []
    real = dict(run=algorithms.run, positions=panel.positions, adjust=panel.adjust_prices)

    def run(algorithm_id, data, **parameters):
        trace.append(dict(kind="algorithm", name=algorithm_id, via="run"))
        return real["run"](algorithm_id, data, **parameters)

    def positions(algorithm_id, prices, split_factors=None, **options):
        trace.append(dict(kind="algorithm", name=algorithm_id, via="positions"))
        return real["positions"](algorithm_id, prices, split_factors, **options)

    def adjust_prices(prices, split_factors=None, **options):
        trace.append(dict(kind="adjust_prices"))
        return real["adjust"](prices, split_factors, **options)

    algorithms.run, algorithms.positions, algorithms.adjust_prices = run, positions, adjust_prices
    panel.positions, panel.adjust_prices = positions, adjust_prices

    confinement = {}
    hook = os.environ.get("FIN_TASKS_CONFINE")
    if hook:
        module, _, name = hook.partition(":")
        readonly = [Path(sys.prefix), Path(sys.base_prefix), Path("/usr"), Path("/lib"),
                    Path("/lib64"), Path(algorithms.__file__).parents[1]]
        # Extra read-only roots, e.g. a shared runtime whose site-packages sit outside
        # sys.prefix, so a confined program can still import scipy submodules lazily.
        readonly += [Path(p) for p in os.environ.get("FIN_TASKS_READONLY", "").split(os.pathsep)
                     if p]
        confinement = getattr(importlib.import_module(module), name)(box, readonly)
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (ADDRESS_SPACE, ADDRESS_SPACE))
    except (ImportError, ValueError, OSError):
        pass

    namespace = {"__name__": "submission"}
    if not request["library"]:
        namespace["__builtins__"] = dict(vars(builtins),
                                         __import__=_guarded_import("fin_skills"))
    result = dict(executed=False, trace=trace, confinement=confinement)
    sys.stdout, sys.stderr = _Capped(), _Capped()
    started = time.perf_counter()

    def call(p, f):
        out = solve(p.copy(), f.copy())
        if not isinstance(out, pd.DataFrame):
            raise TypeError(f"solve must return a pandas DataFrame, got {type(out).__name__}")
        if not out.index.equals(p.index) or list(out.columns) != list(p.columns):
            raise ValueError("solve must return exactly the input index and columns, in order")
        return out.astype(float)

    try:
        exec(compile(source, "submission.py", "exec"), namespace)
        solve = namespace["solve"]
        values = call(raw, factors).to_numpy()
    except BaseException as exc:            # the program's failure is a result, not a crash
        result["error"] = _error(exc)
        base_calls = len(trace)
    else:
        base_calls = len(trace)              # calls made by the program itself, not the checks
        result.update(executed=True, positions=np.where(np.isfinite(values), values,
                                                        0.0).tolist())
        checks = {"finite_output": bool(np.isfinite(values).all())}
        try:
            ones = pd.DataFrame(1.0, index=raw.index, columns=raw.columns)
            other = call(adjusted, ones).to_numpy()
            with np.errstate(invalid="ignore"):
                checks["split_representation"] = bool(np.allclose(
                    np.nan_to_num(other), np.nan_to_num(values), atol=1e-8, rtol=0))
        except BaseException:
            checks["split_representation"] = False
        try:
            lag_ok = True
            for k in request["lag_ks"]:
                verdict = lag_guard.run(fn=lambda frame: call(frame, factors), df=raw.copy(),
                                        k=int(k), tol=1e-8)
                lag_ok = lag_ok and bool(verdict.passed) and not verdict.warnings
        except BaseException:
            lag_ok = False
        checks["execution_lag"] = lag_ok
        result["checks"] = checks
    result["solve_seconds"] = time.perf_counter() - started
    result["trace"] = trace[:base_calls]
    result["library_calls"] = sum(t["kind"] == "algorithm" for t in trace[:base_calls])
    printed = sys.stdout.size + sys.stderr.size
    sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
    result["printed_chars"] = printed
    (box / "result.json").write_text(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
