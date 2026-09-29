"""Run one submitted solve() in the Linux confinement and apply the controller's checks.

Used by the enforced-library study. It never computes reference weights or grades; the hidden
grader in codegen_utility is separate. Checks: whether a library algorithm call was traced,
split-representation invariance, and the library's assert_causal guard applied to positions
shifted up one row, which tests the one-bar execution lag rather than feature causality.
"""
import json
import os
from pathlib import Path
import sys
import time
import traceback


def _frame(value, index, assets, np, pd):
    if not isinstance(value, pd.DataFrame):
        raise TypeError('solve must return a DataFrame')
    if not value.index.equals(index) or list(value.columns) != assets:
        raise ValueError('output index/columns do not match input')
    array = value.to_numpy(dtype=float)
    if not np.isfinite(array).all():
        raise ValueError('output contains nonfinite values')
    return array


def _error(exc):
    return {'type': type(exc).__name__, 'message': str(exc)[:600],
            'traceback': traceback.format_exc()[-3000:]}


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
    assets = request['assets']
    prices = pd.DataFrame(request['prices'], index=index, columns=assets)
    factors = pd.DataFrame(request['factors'], index=index, columns=assets)
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
    shared = os.environ.get('FIN_STUDY_SHARED_RUNTIME')
    if shared:
        readonly.append(Path(shared))
    confinement = confine(box, readonly)
    started = time.perf_counter()
    result = {'executed': False, 'trace': trace, 'confinement': confinement}
    try:
        namespace = {'__name__': 'submission'}
        exec(compile(source, 'submission.py', 'exec'), namespace)
        solve = namespace['solve']
        base = _frame(solve(prices, factors), index, assets, np, pd)
        result.update(executed=True, weights=base.tolist())
    except BaseException as exc:  # SystemExit from a submission is its failure, not ours
        result['error'] = _error(exc)
    result['solve_seconds'] = time.perf_counter() - started
    # Snapshot the agent's own calls before the controller re-runs solve() for its checks.
    result['trace'] = list(trace)
    result['library_calls'] = sum(t['kind'] == 'algorithm' for t in result['trace'])
    if result['executed'] and request.get('checks'):
        checks = {}
        try:
            ones = pd.DataFrame(1.0, index=index, columns=assets)
            other = _frame(solve(prices * factors, ones), index, assets, np, pd)
            diff = float(np.max(np.abs(other - base)))
            checks['split'] = {'passed': diff <= 1e-8, 'max_abs_diff': diff}
        except BaseException as exc:
            checks['split'] = {'passed': False, 'error': _error(exc)}
        lag = []
        for k in request['lag_ks']:
            try:
                verdict = real_get('assert_causal').run(
                    fn=lambda frame: solve(frame, factors).shift(-1), df=prices, k=k)
                lag.append({'k': k, 'passed': bool(verdict.passed),
                            'summary': verdict.summary()[:800]})
            except BaseException as exc:
                lag.append({'k': k, 'passed': False, 'error': _error(exc)})
        checks['lag'] = lag
        result['checks'] = checks
    (box / 'result.json').write_text(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
