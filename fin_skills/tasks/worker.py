"""Execute one input in a disposable process without any evaluator or reference answers.

Run this file directly: importing tasks.__init__ would load the grader. OS confinement
comes from FIN_TASKS_CONFINE; without it this is a trusted local fixture runner only.
"""
import builtins
import ast
import importlib
import io
import json
import os
from pathlib import Path
import sys
import traceback


class Sink(io.TextIOBase):
    def write(self, text):
        return len(text)
    def flush(self):
        pass


def main(box):
    import numpy as np
    import pandas as pd
    request = json.loads((box / "input.json").read_text(encoding="utf-8"))
    source = (box / "submission.py").read_text(encoding="utf-8")
    hook = None
    if os.environ.get("FIN_TASKS_CONFINE"):
        module, _, name = os.environ["FIN_TASKS_CONFINE"].partition(":")
        hook = getattr(importlib.import_module(module), name)
    if request["require_confinement"] and hook is None:
        raise RuntimeError("OS confinement is required")
    public, private = Path(request["public_runtime"]), Path(request["private_package"])
    sys.path[:] = [str(public)] + [p for p in sys.path if p and not (
        Path(p).resolve() == private.parent or private in Path(p).resolve().parents
        or Path(p).resolve() == private)]
    trace = []
    if request["library"]:
        import fin_skills.algorithms as algorithms
        import fin_skills.algorithms.panel as panel
        for attr in ("run", "positions", "adjust_prices"):
            real = getattr(algorithms, attr)
            def observed(*args, _real=real, _name=attr, **kwargs):
                if len(trace) >= 64:
                    raise RuntimeError("library call trace limit exceeded")
                trace.append({"kind": "algorithm" if _name != "adjust_prices" else "adjust_prices",
                              "via": _name})
                return _real(*args, **kwargs)
            setattr(algorithms, attr, observed)
            if hasattr(panel, attr):
                setattr(panel, attr, observed)
    real_import = builtins.__import__
    def restricted(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and (name == "fin_skills" or name.startswith("fin_skills.")):
            if not request["library"]:
                raise ImportError("fin_skills is not available for this task")
            if name.startswith(("fin_skills.tasks", "fin_skills.rsi")):
                raise ImportError("evaluation modules are not part of the public runtime")
        return real_import(name, globals, locals, fromlist, level)
    readonly = [Path(p) for p in request["runtime_readonly"]]
    if request["library"]:
        readonly.append(public)
    confinement = hook(box, readonly) if hook else {}
    if request["require_confinement"] and not confinement:
        raise RuntimeError("confinement hook returned no evidence")
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (16 << 30, 16 << 30))
        resource.setrlimit(resource.RLIMIT_FSIZE, (2 << 20, 2 << 20))
    except ImportError:
        pass
    kind = request.get("result_kind", "positions")
    if kind == "positions":
        idx = pd.to_datetime(request["dates"])
        prices = pd.DataFrame(request["prices"], index=idx, columns=request["columns"])
        factors = pd.DataFrame(request["factors"], index=idx, columns=request["columns"])
    result = {"executed": False, "confinement": confinement}
    # These bytes precede candidate execution and cannot be erased from the parent's pipe.
    os.write(1, b"FIN_TASKS_READY " + json.dumps(confinement).encode() + b"\n")
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = Sink()
    try:
        namespace = {"__name__": "submission", "__builtins__": dict(
            vars(builtins), __import__=restricted)}
        if kind == "scalar_expression":
            tree = ast.parse(source, filename="submission.py", mode="exec")
            if not tree.body or not isinstance(tree.body[-1], ast.Expr):
                raise TypeError("the last statement must be a numerical expression")
            expression = ast.Expression(tree.body.pop().value)
            exec(compile(tree, "submission.py", "exec"), namespace)
            output = eval(compile(expression, "submission.py", "eval"), namespace)
            if isinstance(output, (bool, np.bool_)) or not isinstance(
                    output, (int, float, np.integer, np.floating)):
                raise TypeError("program must return one numerical scalar")
            value = int(output) if isinstance(output, (int, np.integer)) else float(output)
            result.update(executed=True, value=value)
        elif kind == "positions":
            exec(compile(source, "submission.py", "exec"), namespace)
            output = namespace["solve"](prices, factors)
            if not isinstance(output, pd.DataFrame):
                raise TypeError("solve must return a pandas DataFrame")
            result.update(executed=True, values=output.to_numpy(dtype=float).tolist(),
                          dates=[str(v) for v in output.index], columns=list(output.columns))
        else:
            raise ValueError("unknown result contract")
    except BaseException as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)[:2000],
                           "traceback": traceback.format_exc()[-2000:]}
    finally:
        sys.stdout, sys.stderr = original_stdout, original_stderr
    result["trace"] = trace
    result["library_calls"] = sum(t["kind"] == "algorithm" for t in trace)
    (box / "result.json").write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
