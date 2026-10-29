"""Study v2: the enforced-library study rerun with the repaired library surface.

Same four rules, three data seeds, two Qwen2.5-Coder models, three attempts, controller checks
and grading equations as codegen-enforced-v1 (frozen in codegen_enforced.py). The raw arms'
prompts are byte-identical to v1 and their controller decisions are v1's; feedback text is v1's
up to file paths, timings and the worker's file name in traceback tails. What changes is the
library surface:

  * library interface: the call signatures are the real ones (v1 wrote data_dictionary for
    run's data argument) with the data dictionary keys named, and the panel layer
    fin_skills.algorithms.positions()/adjust_prices() is documented, which builds the whole
    date x asset table with the one-bar lag, split adjustment and warm-up handled.
  * rule cards (fin_skills.api.rules) for the task are delivered in the library arm's prompt,
    and after a failed check the controller adds the cause and the rule (rules.explain); it
    never reveals reference weights or grades.
  * call tracers have the library's own signature, so run(algorithm_id=..., data=...) is
    traced and executed (v1's tracers raised TypeError on it, in the controller and in the
    grader), and positions() counts as a library call.
  * the library requirement accepts either fin_skills.algorithms.positions or run.
  * in every arm the controller gives solve() fresh copies of its inputs (v1 reused one
    object, so a program that modified its arguments could fail its own checks), and printed
    output and address space are capped in the workers.

Optional arm rules_guards (not in the default set): raw implementation plus the same rule
cards and explained feedback without library calls, to separate the rules from the library.

Grading uses the E4 equations and probes unchanged, run by a grading worker whose only change
is the tracer signature. A secondary, post-hoc audit runs the execution_lag guard, which also
perturbs each asset by its own factor; it never reaches the model.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time

import numpy as np

from benchmarks.agent_study import codegen_enforced as v1
from benchmarks.agent_study.codegen_enforced import (
    ATTEMPTS, CHECKS, LAG_KS, MULTI, ONE_SHOT, load, sha, strip, write)
from benchmarks.agent_study.codegen_fence_reanalysis import EXTRACTOR, first_fence
from benchmarks.agent_study.codegen_utility import (
    DEFINITIONS, MODELS, SEEDS, TASKS, dataset, library_context, messages as e4_messages,
    reference)
from fin_skills.api.rules import prompt_block, rules_for, explain

VERSION = 'codegen-enforced-v2'
ARMS = ('raw_exec', 'raw_guards', 'rules_guards', 'library_guards')
DEFAULT_ARMS = ('raw_exec', 'raw_guards', 'library_guards')
GUARDED = ('raw_guards', 'rules_guards', 'library_guards')
STRICT_LAG_KS = (32, 64)
CONTROLLER = 'benchmarks.agent_study.codegen_controller_worker_v2'
GRADER = 'benchmarks.agent_study.codegen_worker_v2'
TRAITS = {
    'diagonal_risk_parity': {'positions', 'split_factors', 'returns_from_prices',
                             'rolling_window'},
    'cross_sectional_momentum': {'positions', 'split_factors', 'returns_from_prices',
                                 'rolling_window'},
    'ma_crossover': {'positions', 'split_factors', 'rolling_window'},
    'vol_target_momentum': {'positions', 'split_factors', 'returns_from_prices',
                            'rolling_window'}}
assert set(TRAITS) == set(TASKS) == set(DEFINITIONS)

INTERFACE = (
    'from fin_skills.algorithms import positions, adjust_prices, run, catalog. '
    'positions(algorithm_id, prices, split_factors, *, window=None, parameters=None, lag=1) '
    'returns the whole position table: a DataFrame with the same index and columns as prices. '
    'It adjusts the prices once (prices*split_factors; the factors are cumulative), computes '
    'the position on row t from adjusted closes through row t-1 only, and fills warm-up rows '
    'with 0. Portfolio algorithms (inverse_volatility, cross_sectional_momentum) receive simple '
    'returns of the adjusted closes; window=N limits each decision to the last N returns. '
    'Signal algorithms (ma_crossover, vol_target_momentum) run on each asset and are scaled by '
    '1/number of assets. parameters is a dict passed to the algorithm; pass every value the rule '
    'needs that differs from a default. Defaults: cross_sectional_momentum lookback=60, top_k=3 '
    '(each chosen asset gets 1/top_k); ma_crossover fast=5, slow=20; vol_target_momentum '
    'lookback=20, target_vol=0.10, periods_per_year=252, max_exposure=1.0; inverse_volatility '
    'takes none. '
    'adjust_prices(prices, split_factors) returns the adjusted closes. '
    'run(algorithm_id, data, **parameters) is the lower-level call for ONE decision or ONE '
    'asset: data is a dict keyed by the input name, {"asset_returns": DataFrame of simple '
    'returns with no missing rows} for portfolio algorithms, returning one allocation, or '
    '{"prices": Series of one asset\'s adjusted closes} for signal algorithms, returning '
    'already one-bar-lagged exposures. '
    'Guards: from fin_skills.api import get; get("execution_lag").run(fn=callable, '
    'df=DataFrame, k=integer) checks that positions through row k do not move when rows >= k '
    'change; get("assert_causal").run(fn=callable, df=DataFrame, k=integer) checks features; '
    'get("adjustment_check").run(close=Series, actions=[(date, ratio)]) checks a split '
    'convention. result.passed is a bool. Guards detect defects; they do not repair data.')

ARM_TEXT = dict(v1.ARM_TEXT,
    rules_guards=v1.ARM_TEXT['raw_guards'],
    library_guards=(' The controller rejects any program that does not compute the rule with '
                    'a fin_skills algorithm (fin_skills.algorithms.positions or '
                    'fin_skills.algorithms.run). Before accepting, it also runs two checks: '
                    + CHECKS))
V1_LIBRARY = ('Library requirement failed: no fin_skills.algorithms.run call was traced; '
              'compute the rule with the discovered library algorithm.')
V2_LIBRARY = ('Library requirement failed: no fin_skills.algorithms.positions or run call was '
              'traced; compute the rule with the discovered library algorithm.')


def library_context_v2():
    return dict(library_context(), interface=INTERFACE,
                panel_calls=['fin_skills.algorithms.positions', 'fin_skills.algorithms.adjust_prices'])


def cards(task):
    return [c.id for c in rules_for(TRAITS[task])]


def messages(task, data, arm, context):
    """Raw arms: v1 messages. Library and rules arms: v2 surface plus the task's rule cards."""
    if arm in ('raw_exec', 'raw_guards'):
        return v1.messages(task, data, arm, context)
    base = e4_messages(task, data, 'library' if arm == 'library_guards' else 'raw', context)
    system = base[0]['content']
    assert ONE_SHOT in system
    block = prompt_block(rules_for(TRAITS[task]), tools=arm == 'library_guards')
    system = system.replace(ONE_SHOT, MULTI) + ARM_TEXT[arm] + '\n\n' + block
    return [dict(role='system', content=system), base[1]]


def _run_worker(module, box, timeout):
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    package_root = str(Path(__file__).resolve().parents[2])
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [package_root, env.get('PYTHONPATH')]))
    return subprocess.run([sys.executable, '-B', '-m', module, str(box)], capture_output=True,
                          text=True, timeout=timeout, env=env)


def controller_execute(code, data, root, checks, strict_lag_ks=()):
    """v1 controller_execute with the v2 worker; strict_lag_ks only for the post-hoc audit."""
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='enforced-v2-', dir=root/'tmp') as td:
        box = Path(td)
        (box/'submission.py').write_text(code)
        (box/'input.json').write_text(json.dumps(dict(data, checks=checks, lag_ks=LAG_KS,
                                                      strict_lag_ks=list(strict_lag_ks))))
        timeout = 180 if strict_lag_ks else 120
        try:
            process = _run_worker(CONTROLLER, box, timeout)
            if (box/'result.json').exists():
                result = json.loads((box/'result.json').read_text())
            else:
                result = dict(executed=False, library_calls=0, error=dict(
                    type='WorkerCrash', message=f'returncode {process.returncode}',
                    traceback=process.stderr[-3000:]))
        except subprocess.TimeoutExpired:
            result = dict(executed=False, library_calls=0,
                          error=dict(type='TimeoutExpired', message=f'{timeout} s limit',
                                     traceback=''))
    result['controller_seconds'] = time.perf_counter() - started
    return result


def failed_checks(result):
    """Names of the controller checks a program failed outright (not those that raised)."""
    if not result.get('executed') or 'checks' not in result:
        return []
    failed = []
    split = result['checks']['split']
    if 'error' not in split and not split['passed']:
        failed.append('split_representation')
    if any('error' not in lag and not lag['passed'] for lag in result['checks']['lag']):
        failed.append('execution_lag')
    return failed


def decide(arm, result, attempt):
    """v1 acceptance; the library and rules arms add the cause and rule of each failure."""
    accepted, message = v1.decide('raw_guards' if arm == 'rules_guards' else arm, result, attempt)
    if accepted or arm in ('raw_exec', 'raw_guards'):
        return accepted, message
    message = message.replace(V1_LIBRARY, V2_LIBRARY)
    failed = failed_checks(result)
    if failed:
        body, _, tail = message.rpartition('\n')
        note = explain(failed, tools=arm == 'library_guards')
        message = '\n'.join([body, 'Why it failed and the rule to follow:', note, tail])
    return accepted, message


def grading_execute(code, data, root):
    """codegen_utility.execute with the v2 grading worker (tracer signature fixed)."""
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='codegen-v2-', dir=root/'tmp') as td:
        box = Path(td)
        (box/'submission.py').write_text(code)
        (box/'input.json').write_text(json.dumps(data))
        try:
            process = _run_worker(GRADER, box, 90)
            if process.returncode or not (box/'result.json').exists():
                result = dict(executed=False, infrastructure_error=True,
                              stderr=process.stderr[-6000:], returncode=process.returncode)
            else:
                result = json.loads((box/'result.json').read_text())
        except subprocess.TimeoutExpired:
            result = dict(executed=False, error={'type': 'TimeoutExpired'})
    result['worker_wall_seconds'] = time.perf_counter()-started
    return result


def audit(code, data, task, original, root):
    """codegen_utility.audit verbatim, executing through grading_execute."""
    result = dict(numerically_correct=False, future_invariant=None, split_invariant=None)
    if not original['executed']:
        return result
    base = np.array(original['weights'])
    result['max_abs_error'] = float(np.max(np.abs(base-reference(task, data))))
    result['numerically_correct'] = result['max_abs_error'] <= 1e-8
    canonical = dict(data, prices=(np.array(data['prices'])*np.array(data['factors'])).tolist(),
                     factors=np.ones_like(base).tolist())
    canonical_result = grading_execute(code, canonical, root)
    result['split_probe'] = canonical_result
    result['split_invariant'] = bool(canonical_result['executed'] and
        np.allclose(base, canonical_result['weights'], atol=1e-8, rtol=0))
    probes = []
    for k in (32, 64):
        changed = np.array(data['prices'])
        changed[k:] *= np.linspace(1.15, 1.9, len(changed)-k)[:, None]
        probe = grading_execute(code, dict(data, prices=changed.tolist()), root)
        probes.append(dict(k=k, result=probe, passed=bool(probe['executed'] and
            np.allclose(base[:k+1], np.array(probe['weights'])[:k+1], atol=1e-8, rtol=0))))
    result['future_probes'] = probes
    result['future_invariant'] = all(p['passed'] for p in probes)
    result['financially_valid'] = bool(result['numerically_correct'] and result['split_invariant']
                                       and result['future_invariant'])
    return result


def grade(code, data, task, root):
    initial = grading_execute(code, dict(data, audit_guard=True), root)
    if initial.get('infrastructure_error'):
        initial = dict(executed=False, worker_wall_seconds=initial.get('worker_wall_seconds'),
                       error=dict(type='WorkerCrash', message=initial.get('stderr', '')[-500:]))
    return initial, audit(code, data, task, initial, root)


def strict_lag(code, data, root):
    """Secondary audit: execution_lag guard with per-asset perturbations, k=32 and 64.

    status: pass (every probe ran clean and passed), fail (a probe found a position that moved
    while the program ran clean), unverified (the program ran but a probe raised, warned or
    timed out) or not_run (the program itself did not run).
    """
    out = controller_execute(code, data, root, checks=True, strict_lag_ks=STRICT_LAG_KS)
    probes = (out.get('checks') or {}).get('strict_lag', [])
    clean = [p for p in probes if 'error' not in p and not p.get('warnings')]
    if not out.get('executed'):
        status = 'not_run'
    elif any(not p['passed'] for p in clean):
        status = 'fail'
    elif len(clean) == len(STRICT_LAG_KS) and len(probes) == len(STRICT_LAG_KS):
        status = 'pass'
    else:
        status = 'unverified'
    return dict(executed=bool(out.get('executed')), status=status, passed=status == 'pass',
                probes=probes, error=out.get('error'))


def versions():
    """Interpreter and package versions; results depend on them (pandas 3 rejects some
    in-place dtype changes that pandas 2 accepted)."""
    out = dict(python=sys.version.split()[0])
    for name in ('numpy', 'pandas', 'scipy', 'torch', 'transformers'):
        try:
            out[name] = __import__(name).__version__
        except ImportError:
            out[name] = None
    out['source'] = os.environ.get('FIN_STUDY_SOURCE_SHA256')
    return out


def freeze(root, arms=DEFAULT_ARMS, only=None):
    """Freeze the episodes; `only` (a set of ids) exists for software tests."""
    arms = tuple(arms)
    if not arms or not set(arms) <= set(ARMS) or len(set(arms)) != len(arms):
        raise ValueError(f'arms must be distinct names from {ARMS}')
    root.mkdir(parents=True, exist_ok=False)
    (root/'tmp').mkdir()
    context = library_context_v2()
    rows = []
    for family, model, revision in MODELS:
        for task in TASKS:
            for seed in SEEDS:
                for arm in arms:
                    rows.append(dict(id=f'{family}-{task}-{seed}-{arm}', family=family,
                        model=model, revision=revision, task=task, seed=seed, arm=arm,
                        rule_cards=cards(task) if arm in ('rules_guards', 'library_guards')
                        else [], messages=messages(task, dataset(seed), arm, context)))
    random.Random(20261029).shuffle(rows)
    if only is not None:
        rows = [r for r in rows if r['id'] in only]
    write(root/'inputs.json', rows)
    write(root/'protocol.json', dict(version=VERSION, base_version='codegen-enforced-v1',
        created_utc=datetime.now(timezone.utc).isoformat(), tasks=TASKS, seeds=SEEDS,
        models=MODELS, arms=arms, planned=len(rows), attempts=ATTEMPTS, max_tokens=2048,
        temperature=.1, top_p=1., parser=EXTRACTOR, lag_ks=LAG_KS, tolerance=1e-8,
        strict_lag_ks=STRICT_LAG_KS, controller_worker=CONTROLLER, grading_worker=GRADER,
        rule_cards={task: cards(task) for task in TASKS}, library_interface=INTERFACE,
        input_sha256=sha(root/'inputs.json'),
        primary_endpoint='valid deliverable: accepted by the arm controller and valid under '
                         'the E4 grading equations and probes (execution, 1e-8 agreement, '
                         'timing probes k=32,64 including the same session, split invariance)',
        secondary=['grader validity of the final attempt regardless of acceptance',
                   'grader validity of the first attempt', 'acceptance', 'attempts used',
                   'traced library calls', 'strict execution-lag audit of the final program '
                   '(execution_lag guard, per-asset perturbation, k=32,64; post hoc only)',
                   'tokens', 'generation and controller seconds'],
        changes_from_v1=[
            'library arm: interface text with the real signatures and data keys, and the '
            'positions()/adjust_prices() panel layer documented',
            'library arm: rule cards for the task in the system prompt; failed checks get the '
            'cause and rule from fin_skills.api.rules.explain',
            'controller and grading tracers use run(algorithm_id, data, **parameters); '
            'positions() is traced as a library call',
            'raw arms: prompts, checks and feedback byte-identical to v1',
            'optional rules_guards arm (rule cards and explained feedback, no library)'],
        limits=['Same E4 tasks, seeds, models and grading equations; E4 and v1 failures '
                'informed the repairs, so this is not an independent holdout.',
                'Controller checks overlap grader properties (timing, split); only the grader '
                'checks numerical agreement with hidden references, with other perturbations.',
                'The library arm changes documentation, rules, enforced library use and checks '
                'together; raw_guards isolates the checks and rules_guards, if run, the rules.',
                'Synthetic author-designed rules; three seeds on four fixed rules are not '
                'twelve independent problems; two sizes of one model family.',
                'No financial-return or profitability endpoint; one shared GPU allocation.']))


QUALIFY = {
    # name: (arm, finding that must appear or None for acceptance, code, text that must not)
    'library_positions': ('library_guards', None, '''from fin_skills.algorithms import positions
def solve(prices, split_factors):
    return positions('ma_crossover', prices, split_factors, parameters={'fast': 5, 'slow': 20})
''', None),
    'library_keyword_run': ('library_guards', None, '''import pandas as pd
from fin_skills.algorithms import run
def solve(prices, split_factors):
    adj = prices * split_factors
    out = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for c in adj:
        out[c] = run(algorithm_id='ma_crossover', data={'prices': adj[c]},
                     fast=5, slow=20).fillna(0) / 4
    return out
''', None),
    'raw_without_library': ('library_guards', 'Library requirement failed',
                            v1.QUALIFY['raw_without_library'][2], None),
    'same_session_library_arm': ('library_guards', 'Rule: [execution_lag]',
                                 v1.QUALIFY['same_session'][2], None),
    'same_session_raw_arm': ('raw_guards', 'Lag check failed', v1.QUALIFY['same_session'][2],
                             'Rule:'),
    'recumulated_split_rules_arm': ('rules_guards', 'Rule: [split_once]',
                                    v1.QUALIFY['recumulated_split'][2], 'fin_skills'),
    'panel_module_import': ('library_guards', None, '''from fin_skills.algorithms.panel import positions
def solve(prices, split_factors):
    return positions('ma_crossover', prices, split_factors, parameters={'fast': 5, 'slow': 20})
''', None),
    'modifies_its_inputs': ('raw_guards', None, '''import numpy as np, pandas as pd
def solve(prices, split_factors):
    prices *= split_factors
    fast = prices.rolling(5).mean().shift(1)
    slow = prices.rolling(20).mean().shift(1)
    return (np.sign(fast - slow) / 4).fillna(0.0)
''', None)}

GRADER_CONTROLS = {
    # task: a library program the grader must score valid and the strict audit must pass
    'diagonal_risk_parity': '''from fin_skills.algorithms import positions
def solve(prices, split_factors):
    return positions('inverse_volatility', prices, split_factors, window=60)
''',
    'cross_sectional_momentum': '''from fin_skills.algorithms import positions
def solve(prices, split_factors):
    return positions('cross_sectional_momentum', prices, split_factors,
                     parameters={'lookback': 60, 'top_k': 2})
''',
    'ma_crossover': QUALIFY['library_positions'][2],
    'vol_target_momentum': '''from fin_skills.algorithms import positions
def solve(prices, split_factors):
    return positions('vol_target_momentum', prices, split_factors)
'''}


def qualify(root):
    """Sandbox, controller and grader qualification on a non-task seed; no model involved."""
    data = dataset(991)
    boundary = {
        'network': 'import socket\ndef solve(prices, split_factors):\n socket.socket()\n'
                   ' return prices*0.0',
        'private': 'def solve(prices, split_factors):\n open(' + repr(str(root/'protocol.json'))
                   + ').read()\n return prices*0.0'}
    results, passed = {}, True
    for name, code in boundary.items():
        out = controller_execute(code, data, root, checks=False)
        results[name] = strip(out)
        passed &= out.get('error', {}).get('type') == 'PermissionError'
    for name, (arm, finding, code, absent) in QUALIFY.items():
        out = controller_execute(code, data, root, checks=True)
        accepted, message = decide(arm, out, 1)
        ok = accepted if finding is None else (not accepted and finding in message)
        ok &= absent is None or absent not in (message or '')
        if 'reference' in (message or '').lower() or 'weight' in (message or '').lower():
            ok = False                  # feedback never mentions reference weights
        results[name] = dict(arm=arm, expected_finding=finding, accepted=accepted,
                             feedback=message, controller=strip(out), ok=ok)
        passed &= ok
    positives = [(task, code) for task, code in GRADER_CONTROLS.items()]
    positives.append(('ma_crossover', QUALIFY['library_keyword_run'][2]))
    for task, code in positives:
        graded, check = grade(code, data, task, root)
        strict = strict_lag(code, data, root)
        name = 'grader_positive_' + task + ('_keyword_run' if 'algorithm_id=' in code else '')
        results[name] = strip(dict(execution=graded, audit=check, strict_lag=strict))
        passed &= bool(check.get('financially_valid')) and strict['status'] == 'pass'
    # The default top_k=3 is not the rule; the grader must reject it although checks pass.
    default_k = GRADER_CONTROLS['cross_sectional_momentum'].replace(", 'top_k': 2", '')
    graded, check = grade(default_k, data, 'cross_sectional_momentum', root)
    results['grader_negative_default_top_k'] = strip(dict(execution=graded, audit=check))
    passed &= bool(graded.get('executed')) and check.get('numerically_correct') is False
    same = v1.QUALIFY['same_session'][2]
    graded, check = grade(same, data, 'ma_crossover', root)
    strict = strict_lag(same, data, root)
    results['grader_negative_same_session'] = strip(dict(execution=graded, audit=check,
                                                        strict_lag=strict))
    passed &= check.get('future_invariant') is False and strict['status'] == 'fail'
    write(root/'qualification.json', dict(passed=passed, results=results,
        scope='Software/controller/grader qualification on seed 991 only; not model '
              'performance.'))
    if not passed:
        raise RuntimeError('Controller qualification failed')


def run(root, backend_factory=None):
    protocol = load(root/'protocol.json')
    assert protocol['version'] == VERSION
    assert sha(root/'inputs.json') == protocol['input_sha256']
    assert load(root/'qualification.json')['passed']
    if backend_factory is None:
        from benchmarks.agent_study.transformers_chat import TransformersChat
        import torch

        def backend_factory(model, revision):
            return TransformersChat(model, revision, max_tokens=2048)
        device = dict(torch=torch.__version__, device=torch.cuda.get_device_name(0))
    else:
        device = dict(torch=None, device='test backend')
    write(root/'inference-started.json', dict(protocol_sha256=sha(root/'protocol.json'),
        job_id=os.environ.get('SLURM_JOB_ID'), versions=versions(), **device))
    rows = load(root/'inputs.json')
    inventory = []
    for family, model, revision in MODELS:
        group = [r for r in rows if r['family'] == family]
        if not group:
            continue
        backend = backend_factory(model, revision)
        for item in group:
            backend.seed, backend.calls = item['seed'], 0
            history, data, attempts = list(item['messages']), dataset(item['seed']), []
            for attempt in range(1, ATTEMPTS + 1):
                response = backend(history)
                text = response['choices'][0]['message']['content']
                code, status = first_fence(text)
                result = controller_execute(code, data, root, checks=item['arm'] in GUARDED)
                accepted, feedback = decide(item['arm'], result, attempt)
                path = root/'attempts'/f"{item['id']}-a{attempt}.json"
                write(path, dict(id=item['id'], attempt=attempt, response=response,
                                 extraction=status, code=code, controller=strip(result),
                                 accepted=accepted, feedback=feedback))
                attempts.append(dict(path=path.relative_to(root).as_posix(), sha256=sha(path),
                                     accepted=accepted))
                if accepted:
                    break
                history = history + [dict(role='assistant', content=text),
                                     dict(role='user', content=feedback)]
            inventory.append(dict(id=item['id'], attempts=attempts,
                                  accepted=attempts[-1]['accepted']))
            print(json.dumps(dict(phase='inference', completed=len(inventory),
                                  planned=len(rows))), flush=True)
        del backend
        gc.collect()
        if device['torch']:
            import torch
            torch.cuda.empty_cache()
    write(root/'inference-receipt.json', dict(rows=inventory, planned=len(rows),
        protocol_sha256=sha(root/'protocol.json')))


def score(root):
    protocol = load(root/'protocol.json')
    receipt = load(root/'inference-receipt.json')
    assert receipt['protocol_sha256'] == sha(root/'protocol.json')
    rows = {r['id']: r for r in load(root/'inputs.json')}
    assert len(receipt['rows']) == receipt['planned'] == protocol['planned'] == len(rows)
    assert {r['id'] for r in receipt['rows']} == set(rows)
    results = []
    for rec in receipt['rows']:
        item = rows[rec['id']]
        done = root/'audits'/(item['id']+'.json')
        if done.exists():                    # resume after an interrupted score; never regrade
            results.append(load(done))
            continue
        records = []
        for attempt in rec['attempts']:
            assert sha(root/attempt['path']) == attempt['sha256']
            records.append(load(root/attempt['path']))
        data = dataset(item['seed'])
        final_exec, final_audit = grade(records[-1]['code'], data, item['task'], root)
        if len(records) > 1:
            first_exec, first_audit = grade(records[0]['code'], data, item['task'], root)
        else:
            first_exec, first_audit = final_exec, final_audit
        strict = strict_lag(records[-1]['code'], data, root)
        usage = [r['response']['usage'] for r in records]
        final_valid = bool(final_audit.get('financially_valid'))
        result = dict(id=item['id'], family=item['family'], arm=item['arm'], task=item['task'],
            seed=item['seed'], attempts=len(records), accepted=rec['accepted'],
            valid_deliverable=bool(rec['accepted'] and final_valid), final_valid=final_valid,
            first_valid=bool(first_audit.get('financially_valid')),
            final_execution=strip(final_exec), final_audit=strip(final_audit),
            first_execution=strip(first_exec), first_audit=strip(first_audit),
            final_strict_lag=strip(strict),
            library_calls=records[-1]['controller'].get('library_calls', 0),
            prompt_tokens=sum(u['prompt_tokens'] for u in usage),
            completion_tokens=sum(u['completion_tokens'] for u in usage),
            generation_seconds=sum(r['response']['generation_seconds'] for r in records),
            controller_seconds=sum(r['controller']['controller_seconds'] for r in records),
            truncated=sum(r['response']['choices'][0]['finish_reason'] != 'stop'
                          for r in records))
        write(root/'audits'/(item['id']+'.json'), result)
        results.append(result)
        print(json.dumps(dict(phase='audit', completed=len(results), planned=len(rows))),
              flush=True)
    summary = {}
    for family in [m[0] for m in MODELS] + ['all']:
        for arm in protocol['arms']:
            group = [r for r in results if r['arm'] == arm and family in ('all', r['family'])]
            if not group:
                continue
            summary[family+'-'+arm] = dict(planned=len(group),
                valid_deliverable=sum(r['valid_deliverable'] for r in group),
                accepted=sum(r['accepted'] for r in group),
                final_valid=sum(r['final_valid'] for r in group),
                first_valid=sum(r['first_valid'] for r in group),
                final_executed=sum(r['final_execution']['executed'] for r in group),
                final_numerical=sum(r['final_audit']['numerically_correct'] for r in group),
                final_split_failures=sum(r['final_audit']['split_invariant'] is False
                                         for r in group),
                final_timing_failures=sum(r['final_audit']['future_invariant'] is False
                                          for r in group),
                final_strict_lag_failures=sum(r['final_strict_lag']['status'] == 'fail'
                                              for r in group),
                final_strict_lag_unverified=sum(r['final_strict_lag']['status'] == 'unverified'
                                                for r in group),
                library_users=sum(r['library_calls'] > 0 for r in group),
                mean_attempts=float(np.mean([r['attempts'] for r in group])),
                mean_prompt_tokens=float(np.mean([r['prompt_tokens'] for r in group])),
                mean_completion_tokens=float(np.mean([r['completion_tokens'] for r in group])),
                mean_generation_seconds=float(np.mean([r['generation_seconds']
                                                       for r in group])),
                mean_controller_seconds=float(np.mean([r['controller_seconds']
                                                       for r in group])),
                truncated_responses=sum(r['truncated'] for r in group),
                final_errors=dict(Counter(r['final_execution'].get('error', {}).get(
                    'type', 'none') for r in group)))
    write(root/'scores.json', dict(aggregate=summary,
        inference_receipt_sha256=sha(root/'inference-receipt.json'),
        audits={r['id']: sha(root/'audits'/(r['id']+'.json')) for r in results}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['freeze', 'qualify', 'run', 'score'])
    parser.add_argument('root', type=Path)
    parser.add_argument('--arms', default=','.join(DEFAULT_ARMS),
                        help='freeze only: comma-separated arms, from ' + ','.join(ARMS))
    args = parser.parse_args()
    if args.mode == 'freeze':
        freeze(args.root, arms=args.arms.split(','))
    else:
        globals()[args.mode](args.root)
