"""Trusted checks over values from fresh disposable program processes.

Children receive no task IDs, references, checks or scoring keys. Their public library
excludes evaluation packages. All acceptance decisions are computed in this process.
"""
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile

import numpy as np

from .data import panel
from .spec import Task

MAX_RESULT_BYTES = 1 << 20
MAX_TRACE = 64
READY = b"FIN_TASKS_READY "


class ProgramSandbox:
    """One isolated public runtime, with a fresh process for each typed computation."""
    def __init__(self, *, library, timeout):
        from fin_skills.rsi.run_record import EXECUTION_BUNDLE
        self.original = Path(__file__).resolve().parents[1]
        bundle = EXECUTION_BUNDLE.get()
        self.private = bundle / "fin_skills" if bundle else self.original
        self.library, self.timeout = library, timeout
        self.require = os.environ.get("FIN_TASKS_REQUIRE_CONFINEMENT") == "1"
        self.temporary = tempfile.TemporaryDirectory(prefix="fin-task-")
        self.root, self.calls = Path(self.temporary.name), 0
        self.public = self.root / "public"

    def __enter__(self):
        if self.library:
            shutil.copytree(self.private, self.public / "fin_skills", ignore=shutil.ignore_patterns(
                "tasks", "rsi", "agents", "__pycache__", "*.pyc"))
        return self

    def __exit__(self, *_):
        self.temporary.cleanup()

    def run(self, code, *, kind, inputs):
        if kind not in ("positions", "scalar_expression"):
            raise ValueError("unknown output contract")
        if self.require and not os.environ.get("FIN_TASKS_CONFINE"):
            return dict(executed=False, infrastructure_error=True,
                        error=dict(type="MissingConfinement", message="OS confinement is required"))
        self.calls += 1
        box = self.root / str(self.calls)
        box.mkdir()
        if kind == "positions":
            expected_inputs = {"prices", "factors", "dates", "columns"}
        else:
            expected_inputs = set()
        if set(inputs) != expected_inputs:
            raise ValueError("candidate inputs do not match the public contract")
        payload = dict(inputs, result_kind=kind, library=self.library,
                       public_runtime=str(self.public), private_package=str(self.private),
                       runtime_readonly=_runtime_paths(self.private, self.original),
                       require_confinement=self.require)
        (box / "submission.py").write_text(code, encoding="utf-8")
        (box / "input.json").write_text(json.dumps(payload), encoding="utf-8")
        return _exchange(box, self.private, self.timeout, kind)


def _exchange(box, private, timeout, kind):
    """Shared bounded transport; output decoding and financial grading stay in the parent."""
    started, confinement = False, {}
    def failure(name, message):
        return dict(executed=False, infrastructure_error=not started,
                    error=dict(type=name, message=message[:2000]))
    try:
        with tempfile.TemporaryFile() as errors, subprocess.Popen(
                [sys.executable, "-B", str(private / "tasks/worker.py"), str(box)],
                stdout=subprocess.PIPE, stderr=errors,
                env=_child_environment(private)) as proc:
            timed_out = False
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                proc.kill()
                proc.wait()
            ready = proc.stdout.readline(4096)
            started = ready.startswith(READY)
            if started:
                confinement = json.loads(ready[len(READY):])
            if timed_out:
                return failure("TimeoutExpired", f"{timeout} s limit")
            if proc.returncode or not started:
                errors.seek(0, 2)
                errors.seek(max(0, errors.tell() - 1500))
                return failure("WorkerExit", f"exit {proc.returncode}: " +
                               errors.read(1500).decode("utf-8", errors="replace"))
        result = box / "result.json"
        if not result.exists() or not stat.S_ISREG(result.lstat().st_mode):
            raise ValueError("candidate result must be a regular file, not a link or pipe")
        with result.open("rb") as stream:
            encoded = stream.read(MAX_RESULT_BYTES + 1)
        if len(encoded) > MAX_RESULT_BYTES:
            return failure("OutputLimitExceeded", "candidate result exceeds byte limit")
        reply = json.loads(encoded)
        if not isinstance(reply, dict) or type(reply.get("executed")) is not bool:
            raise ValueError("invalid worker result")
        required = {"executed", "trace", "library_calls", "confinement"}
        if reply["executed"]:
            required |= {"values", "dates", "columns"} if kind == "positions" else {"value"}
        else:
            required |= {"error"}
        if set(reply) != required:
            raise ValueError("candidate result fields do not match the protocol")
        if (not isinstance(reply["trace"], list) or len(reply["trace"]) > MAX_TRACE
                or type(reply["library_calls"]) is not int
                or not 0 <= reply["library_calls"] <= MAX_TRACE):
            raise ValueError("invalid or oversized call telemetry")
        reply["confinement"] = confinement
        if not reply["executed"]:
            error = reply["error"]
            if (not isinstance(error, dict) or not isinstance(error.get("type"), str)
                    or not isinstance(error.get("message"), str)):
                raise ValueError("invalid candidate error record")
            reply["error"] = {k: str(error[k])[:2000] for k in
                              ("type", "message", "traceback") if k in error}
        elif kind == "scalar_expression":
            import math
            if (type(reply["value"]) not in (int, float) or
                    (type(reply["value"]) is float and not math.isfinite(reply["value"]))):
                raise ValueError("program must return one finite numerical scalar")
        return reply
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, OverflowError) as exc:
        return failure("WorkerProtocolError", str(exc))


def evaluate_scalar(code, *, library=False, timeout=120):
    """Execute a final-expression scalar program. No question, answer or grader is sent."""
    with ProgramSandbox(library=library, timeout=timeout) as sandbox:
        return sandbox.run(code, kind="scalar_expression", inputs={})


def _child_environment(private):
    # In particular, do not pass provider credentials, proxies or application settings.
    env = {k: os.environ[k] for k in ("SystemRoot", "WINDIR", "TEMP", "TMP")
           if k in os.environ}
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONHASHSEED="0", PYTHONPATH=str(private.parent), PYTHONNOUSERSITE="1",
               PYTHONIOENCODING="utf-8", LANG="C.UTF-8")
    if os.environ.get("FIN_TASKS_CONFINE"):
        env["FIN_TASKS_CONFINE"] = os.environ["FIN_TASKS_CONFINE"]
    return env


def _runtime_paths(private, original):
    roots = {Path(p).resolve() for p in sys.path if p and Path(p).is_dir()
             and any(Path(p).resolve().is_relative_to(Path(q).resolve())
                     for q in (sys.prefix, sys.base_prefix))}
    roots.update(p.resolve() for p in (Path("/lib"), Path("/lib64")) if p.exists())
    # Extra read-only roots, e.g. a shared runtime whose site-packages sit outside sys.prefix,
    # so a confined program can still import scipy submodules lazily.
    roots.update(Path(p).resolve() for p in os.environ.get("FIN_TASKS_READONLY", "").split(
        os.pathsep) if p and Path(p).is_dir())
    def excluding(root):
        if root in (private, original):
            return []
        if any(p.is_relative_to(root) for p in (private, original)):
            return [p for child in root.iterdir() for p in excluding(child)]
        return [str(root)]
    return sorted({p for root in roots for p in excluding(root)})


def evaluate(code, request, timeout):
    task = Task.from_manifest(request["task"])
    if task.sha256 != request["task_sha256"]:
        raise ValueError("task manifest hash mismatch")
    raw, factors, adjusted = panel(request["seed"], rows=task.rows,
                                   factors=task.factors, order=task.order)
    with ProgramSandbox(library=request["library"], timeout=timeout) as sandbox:
        def run(prices, splits):
            reply = sandbox.run(code, kind="positions", inputs=dict(
                prices=prices.to_numpy().tolist(), factors=splits.to_numpy().tolist(),
                dates=[str(v) for v in prices.index], columns=list(prices.columns)))
            if reply["executed"]:
                try:
                    rows = reply["values"]
                    if (not isinstance(rows, list) or len(rows) != len(prices)
                            or any(not isinstance(row, list) or len(row) != len(prices.columns)
                                   for row in rows)):
                        raise ValueError("solve must return exactly the input index and columns, in order")
                    values = np.asarray(rows, dtype=float)
                    if (values.shape != prices.shape or reply["columns"] != list(prices.columns)
                            or reply["dates"] != [str(v) for v in prices.index]):
                        raise ValueError("solve must return exactly the input index and columns, in order")
                    reply["array"] = values
                except (ValueError, KeyError, TypeError, OverflowError) as exc:
                    return dict(executed=False, infrastructure_error=False,
                                error=dict(type="WorkerProtocolError", message=str(exc)[:2000]))
            return reply
        base = run(raw, factors)
        if not base["executed"]:
            return base
        values = base.pop("array")
        checks = {"finite_output": bool(np.isfinite(values).all())}
        equivalent = run(adjusted, factors * 0 + 1)
        if equivalent.get("infrastructure_error"):
            return equivalent
        checks["split_representation"] = bool(equivalent["executed"] and np.isfinite(
            equivalent["array"]).all() and np.allclose(equivalent["array"], values,
                                                      atol=1e-8, rtol=0))
        lag_ok = True
        for k in request["lag_ks"]:
            if isinstance(k, bool) or not isinstance(k, int) or not 0 < k < len(raw):
                raise ValueError("lag probe must be inside task rows")
            for multiplier in (2.0, np.random.default_rng(0).uniform(.5, 1.5, len(raw.columns))):
                perturbed = raw.copy()
                perturbed.iloc[k:] *= multiplier
                probe = run(perturbed, factors)
                if probe.get("infrastructure_error"):
                    return probe
                lag_ok &= bool(probe["executed"] and np.isfinite(probe["array"]).all()
                               and np.allclose(probe["array"][:k+1], values[:k+1],
                                               atol=1e-8, rtol=0))
        checks["execution_lag"] = bool(lag_ok)
        return dict(executed=True, checks=checks, positions=np.nan_to_num(values).tolist(),
                    library_calls=base["library_calls"], trace=base["trace"],
                    confinement=base.get("confinement", {}), execution_processes=sandbox.calls)
