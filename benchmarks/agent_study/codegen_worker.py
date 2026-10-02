"""Execute one submitted solve function in the existing Linux confinement boundary."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback


def main():
    import numpy as np
    import pandas as pd
    from benchmarks.agent_study.linux_sandbox import confine
    import fin_skills
    import fin_skills.algorithms as algorithms
    import fin_skills.api as api
    box = Path(sys.argv[1]).resolve()
    request = json.loads((box / 'input.json').read_text())
    source = (box / 'submission.py').read_text()
    index = pd.to_datetime(request['dates'])
    prices = pd.DataFrame(request['prices'], index=index, columns=request['assets'])
    factors = pd.DataFrame(request['factors'], index=index, columns=request['assets'])
    # Import numerical dependencies before seccomp denies thread/subprocess creation.
    import scipy.optimize
    import scipy.cluster.hierarchy
    for name in ('assert_causal', 'adjustment_check'):
        api.get(name)
    trace = []
    real_run, real_catalog, real_get = algorithms.run, algorithms.catalog, api.get

    def run(name, data, **kwargs):
        trace.append({'kind': 'algorithm', 'name': name})
        return real_run(name, data, **kwargs)

    def catalog(*args, **kwargs):
        trace.append({'kind': 'discovery'})
        return real_catalog(*args, **kwargs)

    def get(name):
        guard = real_get(name)
        original = guard.run
        class ObservedGuard:
            def run(self, **kwargs):
                result = original(**kwargs)
                trace.append({'kind': 'agent_guard', 'name': name,
                              'passed': result.passed, 'warnings': len(result.warnings)})
                return result
        return ObservedGuard()

    algorithms.run, algorithms.catalog, api.get = run, catalog, get
    readonly = [Path(sys.prefix), Path(sys.base_prefix), Path('/usr'), Path('/lib'),
                Path('/lib64'), Path(fin_skills.__file__).parent]
    # The shared environment is outside sys.prefix for the existing Beacon venv.
    shared = os.environ.get('FIN_STUDY_SHARED_RUNTIME')
    if shared:
        readonly.append(Path(shared))
    confinement = confine(box, readonly)
    started = time.perf_counter()
    result = {'executed': False, 'trace': trace, 'confinement': confinement}
    try:
        namespace = {'__name__': 'submission'}
        exec(compile(source, 'submission.py', 'exec'), namespace)
        value = namespace['solve'](prices, factors)
        if not isinstance(value, pd.DataFrame):
            raise TypeError('solve must return a DataFrame')
        if not value.index.equals(index) or list(value.columns) != request['assets']:
            raise ValueError('output index/columns do not match input')
        if not np.isfinite(value.to_numpy(dtype=float)).all():
            raise ValueError('output contains nonfinite values')
        result.update(executed=True, weights=value.to_numpy(dtype=float).tolist())
        result['solve_seconds'] = time.perf_counter() - started
        if request.get('audit_guard'):
            checked = real_get('assert_causal').run(
                fn=lambda frame: namespace['solve'](frame, factors), df=prices, k=64)
            result['posthoc_causality_guard'] = dict(passed=checked.passed,
                warnings=len(checked.warnings), summary=checked.summary())
    except Exception as exc:
        result['error'] = {'type': type(exc).__name__, 'message': str(exc),
                           'traceback': traceback.format_exc()}
    result.setdefault('solve_seconds', time.perf_counter() - started)
    (box / 'result.json').write_text(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
