"""Confined v6 worker. No read grant covers the study source/data directory."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import traceback

from benchmarks.agent_study.trading_tools_v6 import OUTPUT_LIMIT, Tools, validate_weights


class Submitted(BaseException):
    pass


def run_python(code, tools):
    out, err, target = io.StringIO(), io.StringIO(), None
    def submit(weights):
        nonlocal target
        target, problem = validate_weights(weights)
        if problem:
            raise ValueError(problem)
        raise Submitted()
    namespace = dict(__name__='__main__', submit=submit, **tools.python_bindings())
    ok, error = True, None
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            exec(compile(code, 'agent_code.py', 'exec'), namespace)
        except Submitted:
            pass
        except BaseException as exc:
            error = f'{type(exc).__name__}: {str(exc)[:500]}'
            if isinstance(exc, AttributeError) and "'dict' object has no attribute 'weights'" in str(exc):
                error += "; algorithm replies are dicts: check reply['ok'], then use reply['result']['weights']"
            if isinstance(exc, TypeError) and "read_market" in str(exc) and "lookback" in str(exc):
                error += "; read_market is for RAW pages; use load_history(lookback=...) in library Python"
            ok = False
            target = None
            traceback.print_exc(limit=4)
    text = out.getvalue() + ('\n[stderr]\n' + err.getvalue() if err.getvalue() else '')
    return dict(ok=ok, error=error, output=text, truncated=False,
                submission=target, tool_calls=tools.calls)


def readonly_paths(workspace, arm, package):
    paths = [Path(sys.prefix), Path(sys.base_prefix), Path('/usr'), Path('/lib'),
             Path('/lib64'), Path(workspace)]
    if arm == 'library':
        paths.append(Path(package))
    shared = os.environ.get('FIN_STUDY_SHARED_RUNTIME')
    if shared:
        paths.append(Path(shared))
    study = Path(__file__).resolve().parent
    # Fail closed if a runtime grant would indirectly reopen the whole study tree.
    for path in paths:
        if study.is_relative_to(path.resolve()):
            raise RuntimeError(f'read grant contains study data: {path}')
    return paths


def main():
    box, workspace = (Path(p).resolve() for p in sys.argv[1:3])
    request = json.loads((box / 'request.json').read_text())
    arm = request['arm']
    if arm not in ('raw', 'library'):
        raise ValueError('unknown arm')
    # Preload helper source and scientific modules; no market data is loaded here.
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import scipy.optimize  # noqa: F401
    import scipy.cluster.hierarchy  # noqa: F401
    import fin_skills
    import fin_skills.algorithms  # noqa: F401
    import fin_skills.api  # noqa: F401
    import benchmarks.agent_study.trading_study  # noqa: F401
    import benchmarks.agent_study.trading_worker  # noqa: F401
    from benchmarks.agent_study.linux_sandbox import confine
    paths = readonly_paths(workspace, arm, Path(fin_skills.__file__).parent)
    if arm == 'raw':
        for name in list(sys.modules):
            if name == 'fin_skills' or name.startswith('fin_skills.'):
                del sys.modules[name]
        class Blocker:
            def find_spec(self, name, path=None, target=None):
                if name == 'fin_skills' or name.startswith('fin_skills.'):
                    raise ImportError('fin_skills unavailable in raw condition')
        sys.meta_path.insert(0, Blocker())
    os.chdir(workspace)
    confine(box, paths)
    tools = Tools(workspace, arm, request.get('menu_seed', 0))
    if request['tool'] == 'run_python':
        result = run_python(request['arguments'].get('code', ''), tools)
    else:
        result = tools.call(request['tool'], request['arguments'])
    (box / 'result.json').write_text(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
