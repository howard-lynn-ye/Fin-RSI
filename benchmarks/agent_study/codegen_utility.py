"""Frozen, first-submission paired code generation benchmark; no model repair loop."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd

TASKS = ('diagonal_risk_parity', 'cross_sectional_momentum', 'ma_crossover',
         'vol_target_momentum')
SEEDS = (11, 23, 37)
MODELS = (
    ('7b', 'Qwen/Qwen2.5-Coder-7B-Instruct', 'c03e6d358207e414f1eca0bb1891e29f1db0e242'),
    ('14b', 'Qwen/Qwen2.5-Coder-14B-Instruct', 'aedcc2d42b622764e023cf882b6652e646b95671'))
DEFINITIONS = {
    'diagonal_risk_parity': 'For every session t >= 61, use the 60 simple returns ending at t-1. '
        'Normalize inverse sample standard deviations (ddof=1) to sum to 1. This is diagonal '
        'risk parity (inverse-volatility), not full-covariance equal risk contribution. Earlier rows are zero.',
    'cross_sectional_momentum': 'For every t >= 61, compound the 60 simple returns ending at t-1. '
        'Rank assets descending; stable ties follow column order. Among the top two assets, '
        'give each with strictly positive momentum weight 0.5; leave the remainder in cash. Earlier rows are zero.',
    'ma_crossover': 'For each asset, compute sign(mean(last 5 adjusted closes)-mean(last 20 adjusted closes)) '
        'at close t-1, and divide by 4 for the weight at t. Earliest nonzero-capable row is t=20. '
        'Earlier rows are zero. Use simple rolling means, not exponential means.',
    'vol_target_momentum': 'For each asset at close t-1, compute sign(P[t-1]/P[t-21]-1), multiply by '
        'min(1, 0.10/(sample_std(last 20 simple returns,ddof=1)*sqrt(252))), then divide by 4. '
        'Put that weight on row t. Earliest nonzero-capable row is t=21; earlier rows are zero.'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as out:
        json.dump(value, out, indent=2, allow_nan=False)


def dataset(seed):
    rng = np.random.default_rng(seed)
    returns = rng.normal([.001, -.0004, .0005, .0002], [.013, .009, .017, .011], (96, 4))
    adjusted = 100 * np.cumprod(1 + returns, axis=0)
    factors = np.ones_like(adjusted)
    factors[45:, 0] = 2
    factors[70:, 2] = 3
    return dict(dates=pd.bdate_range('2024-01-02', periods=96).strftime('%Y-%m-%d').tolist(),
                assets=['A', 'B', 'C', 'D'], prices=(adjusted/factors).tolist(),
                factors=factors.tolist())


def reference(task, data):
    """Independent explicit-window equations; never calls fin-skills."""
    p = np.array(data['prices']) * np.array(data['factors'])
    ret = p[1:] / p[:-1] - 1
    w = np.zeros_like(p)
    for t in range(1, len(p)):
        if task in TASKS[:2] and t >= 61:
            history = ret[t-61:t-1]
            if task == TASKS[0]:
                inv = 1 / history.std(axis=0, ddof=1)
                w[t] = inv / inv.sum()
            else:
                m = np.prod(1 + history, axis=0) - 1
                chosen = np.argsort(-m, kind='stable')[:2]
                w[t, chosen[m[chosen] > 0]] = .5
        elif task == 'ma_crossover' and t >= 20:
            w[t] = np.sign(p[t-5:t].mean(axis=0) - p[t-20:t].mean(axis=0)) / 4
        elif task == 'vol_target_momentum' and t >= 21:
            vol = ret[t-21:t-1].std(axis=0, ddof=1) * np.sqrt(252)
            w[t] = np.sign(p[t-1] / p[t-21] - 1) * np.minimum(1, .10/vol) / 4
    return w


def parse_code(text):
    text = text.strip()
    match = re.fullmatch(r'```(?:python)?\s*\n(.*?)\n```', text, re.DOTALL)
    return match.group(1) if match else text


def library_context():
    from fin_skills.algorithms import catalog
    names = {'inverse_volatility', 'cross_sectional_momentum', 'ma_crossover', 'vol_target_momentum'}
    cards = [c for c in catalog() if c['id'] in names]
    return dict(discovery_call='fin_skills.algorithms.catalog()', discovered_cards=cards,
        interface='from fin_skills.algorithms import catalog, run; '
            'run(algorithm_id, data_dictionary, **parameters) returns a Series (not a report). '
            'inverse_volatility uses asset_returns DataFrame and no parameters, returning one allocation; '
            'cross_sectional_momentum uses asset_returns, lookback=60, top_k=2, returning one allocation; '
            'ma_crossover uses prices Series, fast=5, slow=20, returning already one-bar-lagged exposures; '
            'vol_target_momentum uses prices Series, lookback=20,target_vol=0.10,periods_per_year=252,'
            'max_exposure=1.0, returning already one-bar-lagged exposures. Missing warmup rows must become 0. '
            'Static allocation methods use only the history you pass; apply to subsequent returns. '
            'From fin_skills.api import get; get("adjustment_check").run(close=Series,actions=[(date,ratio)]) '
            'checks a split convention. get("assert_causal").run(fn=callable,df=DataFrame,k=integer) '
            'perturbs future input and checks earlier outputs; result.passed is a bool. '
            'Never pass unadjusted split jumps into return-based algorithms. Guards detect defects; '
            'they do not automatically repair data. Use real library algorithms and audit your function.')


def messages(task, data, arm, context):
    packet = dict(task=task, definition=DEFINITIONS[task], data=data)
    if arm == 'library':
        packet['library_discovery'] = context
    system = ('Write one Python program defining solve(prices, split_factors). Both arguments are '
        'date-indexed pandas DataFrames, same shape, columns A,B,C,D. Return a finite float DataFrame '
        'of weights with exactly the same index and columns. prices are RAW quotes; cumulative '
        'split_factors become known on their event dates. The continuous forward-adjusted close is '
        'prices*split_factors. A position on row t earns return t-1 to t and MUST use information '
        'strictly before t. Implement the specified mathematical rule, including warmup and timing. '
        'Use numpy/pandas/scipy. No files, network, subprocesses, printed answers or hardcoded data. '
        'The function will be called on other data as well. Return Python only, optionally one python '
        'fence. You get one response capped at 2048 tokens; no execution feedback or repairs. ')
    system += ('Use the discovered fin_skills algorithms and guards where applicable.' if arm == 'library'
               else 'Implement from scratch. Do not import or use fin_skills or any other domain library.')
    return [dict(role='system', content=system), dict(role='user', content=json.dumps(packet))]


def execute(code, data, root):
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='codegen-', dir=root/'tmp') as td:
        box = Path(td)
        (box/'submission.py').write_text(code)
        (box/'input.json').write_text(json.dumps(data))
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        try:
            process = subprocess.run([sys.executable, '-B', '-m',
                'benchmarks.agent_study.codegen_worker', str(box)], capture_output=True,
                text=True, timeout=90, env=env)
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
    """Same post-hoc financial grader for both arms; failure never disappears from denominator."""
    result = dict(numerically_correct=False, future_invariant=None, split_invariant=None)
    if not original['executed']:
        return result
    base = np.array(original['weights'])
    result['max_abs_error'] = float(np.max(np.abs(base-reference(task, data))))
    result['numerically_correct'] = result['max_abs_error'] <= 1e-8
    canonical = dict(data, prices=(np.array(data['prices'])*np.array(data['factors'])).tolist(),
                     factors=np.ones_like(base).tolist())
    canonical_result = execute(code, canonical, root)
    result['split_probe'] = canonical_result
    result['split_invariant'] = bool(canonical_result['executed'] and
        np.allclose(base, canonical_result['weights'], atol=1e-8, rtol=0))
    probes = []
    for k in (32, 64):
        changed = np.array(data['prices'])
        changed[k:] *= np.linspace(1.15, 1.9, len(changed)-k)[:, None]
        probe = execute(code, dict(data, prices=changed.tolist()), root)
        probes.append(dict(k=k, result=probe, passed=bool(probe['executed'] and
            np.allclose(base[:k+1], np.array(probe['weights'])[:k+1], atol=1e-8, rtol=0))))
    result['future_probes'] = probes
    result['future_invariant'] = all(p['passed'] for p in probes)
    result['financially_valid'] = bool(result['numerically_correct'] and result['split_invariant']
                                       and result['future_invariant'])
    return result


def freeze(root):
    root.mkdir(parents=True, exist_ok=False)
    (root/'tmp').mkdir()
    context = library_context()
    rows = []
    for family, model, revision in MODELS:
        for task in TASKS:
            for seed in SEEDS:
                for arm in ('raw', 'library'):
                    rows.append(dict(id=f'{family}-{task}-{seed}-{arm}', family=family,
                        model=model, revision=revision, task=task, seed=seed, arm=arm,
                        messages=messages(task, dataset(seed), arm, context)))
    random.Random(20260928).shuffle(rows)
    write(root/'inputs.json', rows)
    write(root/'protocol.json', dict(version='codegen-utility-v1', created_utc=datetime.now(timezone.utc).isoformat(),
        tasks=TASKS, seeds=SEEDS, models=MODELS, planned=48, max_tokens=2048,
        max_responses=1, temperature=.1, top_p=1., tolerance=1e-8,
        endpoints=['first-submission execution','numerical correctness','future invariance including same-session',
                   'split-representation invariance','tokens','generation seconds','execution seconds'],
        input_sha256=sha(root/'inputs.json'),
        limits=['Synthetic author-designed tasks, not independent holdout.',
                'Treatment adds library documentation and guarded interfaces; not a guard-only ablation.',
                'First-submission success excludes repairs; audit failures remain failures.',
                'A passing sampled probe is not proof of universal causality.',
                'No financial-return or profitability endpoint.',
                'Timing measured on one shared allocation; no asymptotic scaling inference.']))


def run(root):
    from benchmarks.agent_study.transformers_chat import TransformersChat
    import torch
    protocol = json.loads((root/'protocol.json').read_text())
    assert sha(root/'inputs.json') == protocol['input_sha256']
    write(root/'inference-started.json', dict(protocol_sha256=sha(root/'protocol.json'),
        job_id=os.environ.get('SLURM_JOB_ID'), torch=torch.__version__, device=torch.cuda.get_device_name(0)))
    rows = json.loads((root/'inputs.json').read_text())
    inventory = []
    for family, model, revision in MODELS:
        backend = TransformersChat(model, revision, max_tokens=2048)
        for item in [r for r in rows if r['family'] == family]:
            backend.seed, backend.calls = item['seed'], 0
            response = backend(item['messages'])
            path = root/'responses'/(item['id']+'.json')
            write(path, dict(id=item['id'], response=response))
            inventory.append(dict(id=item['id'], path=path.relative_to(root).as_posix(), sha256=sha(path)))
            print(json.dumps(dict(phase='inference', completed=len(inventory), planned=48)), flush=True)
        del backend
        gc.collect()
        torch.cuda.empty_cache()
    write(root/'inference-receipt.json', dict(rows=inventory, planned=48,
                                            protocol_sha256=sha(root/'protocol.json')))


def qualify(root):
    probes = {
        'valid': 'import pandas as pd\ndef solve(prices, split_factors):\n return prices*0.0',
        'network': 'import socket\ndef solve(prices, split_factors):\n socket.socket()\n return prices*0.0',
        'private': 'def solve(prices, split_factors):\n open('+repr(str(root/'protocol.json'))+').read()\n return prices*0.0',
    }
    results = {name: execute(code, dataset(991), root) for name, code in probes.items()}
    passed = (results['valid']['executed'] and all(
        results[name].get('error',{}).get('type') == 'PermissionError' for name in ('network','private')))
    write(root/'qualification.json', dict(passed=passed, probes=results,
        scope='Software/sandbox qualification only, not model performance or task selection.'))
    if not passed:
        raise RuntimeError('Code execution boundary failed qualification')


def score(root):
    receipt = json.loads((root/'inference-receipt.json').read_text())
    assert receipt['protocol_sha256'] == sha(root/'protocol.json')
    assert len(receipt['rows']) == receipt['planned'] == 48
    rows = {r['id']: r for r in json.loads((root/'inputs.json').read_text())}
    assert set(rows) == {r['id'] for r in receipt['rows']}
    results = []
    for rec in receipt['rows']:
        path = root/rec['path']
        assert sha(path) == rec['sha256']
        item = rows[rec['id']]
        response = json.loads(path.read_text())['response']
        code = parse_code(response['choices'][0]['message']['content'])
        data = dataset(item['seed'])
        initial = execute(code, dict(data, audit_guard=True), root)
        if initial.get('infrastructure_error'):
            raise RuntimeError(f'Confinement/runtime unavailable: {initial}')
        check = audit(code, data, item['task'], initial, root)
        result = dict(id=item['id'], family=item['family'], arm=item['arm'], task=item['task'], seed=item['seed'],
                      execution=initial, audit=check, usage=response['usage'],
                      generation_seconds=response['generation_seconds'],
                      finish_reason=response['choices'][0]['finish_reason'])
        write(root/'audits'/(item['id']+'.json'), result)
        results.append(result)
        print(json.dumps(dict(phase='audit', completed=len(results), planned=48)), flush=True)
    summary = {}
    for family, _, _ in MODELS:
        for arm in ('raw', 'library'):
            group = [r for r in results if r['family'] == family and r['arm'] == arm]
            summary[family+'-'+arm] = dict(planned=len(group),
                executed=sum(r['execution']['executed'] for r in group),
                numerical=sum(r['audit']['numerically_correct'] for r in group),
                financial_valid=sum(r['audit'].get('financially_valid', False) for r in group),
                future_failures=sum(r['audit']['future_invariant'] is False for r in group),
                split_failures=sum(r['audit']['split_invariant'] is False for r in group),
                algorithm_users=sum(any(t['kind']=='algorithm' for t in r['execution'].get('trace',[])) for r in group),
                agent_guard_users=sum(any(t['kind']=='agent_guard' for t in r['execution'].get('trace',[])) for r in group),
                posthoc_guard_rejections=sum(r['execution'].get('posthoc_causality_guard',{}).get('passed') is False for r in group),
                errors=dict(Counter(r['execution'].get('error',{}).get('type','none') for r in group)),
                mean_input_tokens=float(np.mean([r['usage']['prompt_tokens'] for r in group])),
                mean_output_tokens=float(np.mean([r['usage']['completion_tokens'] for r in group])),
                mean_generation_seconds=float(np.mean([r['generation_seconds'] for r in group])),
                mean_worker_seconds=float(np.mean([r['execution']['worker_wall_seconds'] for r in group])))
    write(root/'scores.json', dict(aggregate=summary, inference_receipt_sha256=sha(root/'inference-receipt.json'),
        audits={r['id']:sha(root/'audits'/(r['id']+'.json')) for r in results}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['freeze', 'qualify', 'run', 'score'])
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    globals()[args.mode](args.root)
