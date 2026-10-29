"""Controller worker for codegen-enforced-v2: v1 behaviour with the repaired library surface.

Differences from codegen_controller_worker (v1, frozen with its study):
  * The call tracer has the library's own signature, run(algorithm_id, data, **parameters),
    so keyword calls are traced and executed instead of failing inside the tracer (v1 named
    the arguments name, data and raised on run(algorithm_id=..., data=...)).
  * fin_skills.algorithms.positions and adjust_prices are traced as library calls too.
The controller checks are unchanged from v1 (split representation; assert_causal on positions
shifted up one row), so acceptance in every arm is judged exactly as in v1. A request may add
strict_lag_ks for the post-hoc audit: the execution_lag guard, which also perturbs each asset by
its own factor, so a cross-sectional rank taken on the trading close is caught. The audit is
never run during an episode and never reaches the model.
"""
import json
import os
from pathlib import Path
import sys
import time

from benchmarks.agent_study.codegen_controller_worker import _error, _frame


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
    import scipy.optimize
    import scipy.cluster.hierarchy
    for name in ('assert_causal', 'adjustment_check', 'execution_lag'):
        api.get(name)
    trace = []
    real = dict(run=algorithms.run, catalog=algorithms.catalog, positions=algorithms.positions,
                adjust_prices=algorithms.adjust_prices, get=api.get)

    def run(algorithm_id, data, **parameters):
        trace.append({'kind': 'algorithm', 'name': algorithm_id, 'via': 'run'})
        return real['run'](algorithm_id, data, **parameters)

    def positions(algorithm_id, prices, split_factors=None, **options):
        trace.append({'kind': 'algorithm', 'name': algorithm_id, 'via': 'positions'})
        return real['positions'](algorithm_id, prices, split_factors, **options)

    def adjust_prices(prices, split_factors=None, **options):
        trace.append({'kind': 'adjust_prices'})
        return real['adjust_prices'](prices, split_factors, **options)

    def catalog(*args, **kwargs):
        trace.append({'kind': 'discovery'})
        return real['catalog'](*args, **kwargs)

    def get(name):
        guard = real['get'](name)
        original = guard.run

        class ObservedGuard:
            def run(self, **kwargs):
                result = original(**kwargs)
                trace.append({'kind': 'agent_guard', 'name': name,
                              'passed': result.passed, 'warnings': len(result.warnings)})
                return result
        return ObservedGuard()

    algorithms.run, algorithms.catalog, api.get = run, catalog, get
    algorithms.positions, algorithms.adjust_prices = positions, adjust_prices
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
    except BaseException as exc:
        result['error'] = _error(exc)
    result['solve_seconds'] = time.perf_counter() - started
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
                verdict = real['get']('assert_causal').run(
                    fn=lambda frame: solve(frame, factors).shift(-1), df=prices, k=k)
                lag.append({'k': k, 'passed': bool(verdict.passed),
                            'summary': verdict.summary()[:800]})
            except BaseException as exc:
                lag.append({'k': k, 'passed': False, 'error': _error(exc)})
        checks['lag'] = lag
        strict = []
        for k in request.get('strict_lag_ks', ()):
            try:
                verdict = real['get']('execution_lag').run(
                    fn=lambda frame: solve(frame, factors), df=prices, k=k)
                strict.append({'k': k, 'passed': bool(verdict.passed),
                               'warnings': len(verdict.warnings),
                               'summary': verdict.summary()[:800]})
            except BaseException as exc:
                strict.append({'k': k, 'passed': False, 'error': _error(exc)})
        if strict:
            checks['strict_lag'] = strict
        result['checks'] = checks
    (box / 'result.json').write_text(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
