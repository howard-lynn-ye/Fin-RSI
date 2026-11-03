"""Controller worker for codegen-enforced-v2: v1 behaviour with the repaired library surface.

Differences from codegen_controller_worker (v1, frozen with its study):
  * The call tracer has the library's own signature, run(algorithm_id, data, **parameters),
    so keyword calls are traced and executed instead of failing inside the tracer (v1 named
    the arguments name, data and raised on run(algorithm_id=..., data=...)).
  * fin_skills.algorithms.positions and adjust_prices are traced as library calls, both as
    package attributes and as fin_skills.algorithms.panel attributes.
  * Every call to solve() gets fresh copies of the inputs, so a program that modifies its
    arguments in place cannot change the data its own checks use (v1 reused one object).
  * A split difference too large to represent is recorded as the largest float instead of
    crashing the JSON write; printed output (stdout and stderr) is capped at 64 KiB per
    stream and address space at 16 GiB.
The controller checks are unchanged from v1 (split representation; assert_causal on positions
shifted up one row). A request may add strict_lag_ks for the post-hoc audit: the
execution_lag guard, which also perturbs each asset by its own factor, so a cross-sectional
rank taken on the trading close is caught. The audit is never run during an episode and
never reaches the model.
"""
import io
import json
import os
from pathlib import Path
import sys
import time

from benchmarks.agent_study.codegen_controller_worker import _error, _frame

OUTPUT_CAP = 65536
ADDRESS_SPACE = 16 << 30


class CappedText(io.TextIOBase):
    """Keeps the first OUTPUT_CAP characters a program prints and drops the rest."""

    def __init__(self):
        self.kept = []
        self.size = 0

    def writable(self):
        return True

    def write(self, text):
        if self.size < OUTPUT_CAP:
            self.kept.append(text[:OUTPUT_CAP - self.size])
        self.size += len(text)
        return len(text)


def cap_resources():
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (ADDRESS_SPACE, ADDRESS_SPACE))
        return {'address_space_bytes': ADDRESS_SPACE}
    except (ImportError, ValueError, OSError) as exc:
        return {'address_space_bytes': None, 'error': type(exc).__name__}


def main():
    import numpy as np
    import pandas as pd
    from benchmarks.agent_study.linux_sandbox import confine
    import fin_skills
    import fin_skills.algorithms as algorithms
    import fin_skills.algorithms.panel as panel
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
    real = dict(run=algorithms.run, catalog=algorithms.catalog, positions=panel.positions,
                adjust_prices=panel.adjust_prices, get=api.get)

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
    panel.positions, panel.adjust_prices = positions, adjust_prices
    readonly = [Path(sys.prefix), Path(sys.base_prefix), Path('/usr'), Path('/lib'),
                Path('/lib64'), Path(fin_skills.__file__).parent]
    shared = os.environ.get('FIN_STUDY_SHARED_RUNTIME')
    if shared:
        readonly.append(Path(shared))
    limits = cap_resources()
    confinement = dict(confine(box, readonly), **limits)
    printed, errors = CappedText(), CappedText()
    sys.stdout, sys.stderr = printed, errors
    started = time.perf_counter()
    result = {'executed': False, 'trace': trace, 'confinement': confinement}

    def call(p, f):
        return solve(p.copy(), f.copy())

    try:
        namespace = {'__name__': 'submission'}
        exec(compile(source, 'submission.py', 'exec'), namespace)
        solve = namespace['solve']
        base = _frame(call(prices, factors), index, assets, np, pd)
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
            other = _frame(call(prices * factors, ones), index, assets, np, pd)
            with np.errstate(over='ignore', invalid='ignore'):
                diff = float(np.max(np.abs(other - base)))
            if not np.isfinite(diff):
                diff = float(np.finfo(float).max)
            checks['split'] = {'passed': diff <= 1e-8, 'max_abs_diff': diff}
        except BaseException as exc:
            checks['split'] = {'passed': False, 'error': _error(exc)}
        lag = []
        for k in request['lag_ks']:
            try:
                verdict = real['get']('assert_causal').run(
                    fn=lambda frame: call(frame, factors).shift(-1), df=prices.copy(), k=k)
                lag.append({'k': k, 'passed': bool(verdict.passed),
                            'summary': verdict.summary()[:800]})
            except BaseException as exc:
                lag.append({'k': k, 'passed': False, 'error': _error(exc)})
        checks['lag'] = lag
        strict = []
        for k in request.get('strict_lag_ks', ()):
            try:
                verdict = real['get']('execution_lag').run(
                    fn=lambda frame: call(frame, factors), df=prices.copy(), k=k, tol=1e-8)
                strict.append({'k': k, 'passed': bool(verdict.passed),
                               'warnings': len(verdict.warnings),
                               'summary': verdict.summary()[:800]})
            except BaseException as exc:
                strict.append({'k': k, 'passed': False, 'error': _error(exc)})
        if strict:
            checks['strict_lag'] = strict
        result['checks'] = checks
    sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
    result['printed_chars'] = printed.size + errors.size
    (box / 'result.json').write_text(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
