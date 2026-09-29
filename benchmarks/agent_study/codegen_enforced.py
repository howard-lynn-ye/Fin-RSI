"""Frozen study: does enforcing library calls and guards change the validity of financial code?

Same four rules, three data seeds, two Qwen2.5-Coder models and hidden grader as the E4
first-submission study (codegen_utility). Every episode allows up to three attempts with
controller feedback, in three arms:

  raw_exec        implement from scratch; the controller reports execution errors only
  raw_guards      implement from scratch; acceptance also requires the lag and split checks
  library_guards  E4 library documentation; acceptance requires a traced library algorithm
                  call plus the same lag and split checks

The controller never computes reference weights. Numerical agreement, timing and split
validity are graded after all attempts are frozen, by the unchanged E4 grader.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time

import numpy as np

from benchmarks.agent_study.codegen_utility import (
    MODELS, SEEDS, TASKS, audit, dataset, execute, library_context, messages as e4_messages)
from benchmarks.agent_study.codegen_fence_reanalysis import EXTRACTOR, first_fence

ARMS = ('raw_exec', 'raw_guards', 'library_guards')
ATTEMPTS = 3
LAG_KS = (40, 72)
ONE_SHOT = 'You get one response capped at 2048 tokens; no execution feedback or repairs. '
MULTI = ('You may make up to 3 attempts, each capped at 2048 tokens. After each attempt a '
         'controller runs your program and replies with its findings; answer with the complete '
         'revised program. ')
CHECKS = ('the fin_skills assert_causal guard applied to your positions shifted up one row '
          '(a position must not depend on prices at or after its own session), and a '
          'split-representation check (the output must not change when the same prices are '
          'passed split-adjusted with unit split factors).')
ARM_TEXT = {
    'raw_exec': ' The controller reports execution errors only and accepts the first program '
                'that runs.',
    'raw_guards': ' Before accepting, the controller runs two checks: ' + CHECKS
                  + ' You must still not import fin_skills yourself.',
    'library_guards': ' The controller rejects any program that does not call '
                      'fin_skills.algorithms.run to compute the rule. Before accepting, it also '
                      'runs two checks: ' + CHECKS}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(text):
    return hashlib.sha256(text.encode('utf8')).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as out:
        json.dump(value, out, indent=2, allow_nan=False)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def strip(result):
    """Replace weight matrices by digests; keep every flag, error and guard summary."""
    if isinstance(result, dict):
        return {('weights_sha256' if k == 'weights' else k):
                (digest(json.dumps(v)) if k == 'weights' else strip(v))
                for k, v in result.items()}
    if isinstance(result, list):
        return [strip(v) for v in result]
    return result


def messages(task, data, arm, context):
    base = e4_messages(task, data, 'library' if arm == 'library_guards' else 'raw', context)
    system = base[0]['content']
    assert ONE_SHOT in system
    system = system.replace(ONE_SHOT, MULTI) + ARM_TEXT[arm]
    return [dict(role='system', content=system), base[1]]


def controller_execute(code, data, root, checks):
    """Run the submission in the confinement; a crash or timeout is the submission's failure."""
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='enforced-', dir=root/'tmp') as td:
        box = Path(td)
        (box/'submission.py').write_text(code)
        (box/'input.json').write_text(json.dumps(dict(data, checks=checks, lag_ks=LAG_KS)))
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                   MKL_NUM_THREADS='1')
        package_root = str(Path(__file__).resolve().parents[2])
        env['PYTHONPATH'] = os.pathsep.join(filter(None, [package_root,
                                                          env.get('PYTHONPATH')]))
        try:
            process = subprocess.run([sys.executable, '-B', '-m',
                'benchmarks.agent_study.codegen_controller_worker', str(box)],
                capture_output=True, text=True, timeout=120, env=env)
            if (box/'result.json').exists():
                result = json.loads((box/'result.json').read_text())
            else:
                result = dict(executed=False, library_calls=0, error=dict(
                    type='WorkerCrash', message=f'returncode {process.returncode}',
                    traceback=process.stderr[-3000:]))
        except subprocess.TimeoutExpired:
            result = dict(executed=False, library_calls=0,
                          error=dict(type='TimeoutExpired', message='120 s limit', traceback=''))
    result['controller_seconds'] = time.perf_counter() - started
    return result


def decide(arm, result, attempt):
    """Acceptance and feedback text. Never reveals reference weights or grades."""
    head = f'Controller findings for attempt {attempt} of {ATTEMPTS}:'
    tail = 'Return the complete corrected program.'
    if not result.get('executed'):
        error = result.get('error') or {}
        lines = [head, f"Execution failed: {error.get('type', 'Error')}: "
                       f"{error.get('message', '')}"[:700]]
        trace = [l for l in (error.get('traceback') or '').strip().splitlines() if l.strip()]
        if trace:
            lines.append('Traceback tail:\n' + '\n'.join(trace[-3:])[:900])
        return False, '\n'.join(lines + [tail])
    problems = []
    if arm == 'library_guards' and not result.get('library_calls'):
        problems.append('Library requirement failed: no fin_skills.algorithms.run call was '
                        'traced; compute the rule with the discovered library algorithm.')
    if arm != 'raw_exec':
        split = result['checks']['split']
        if 'error' in split:
            problems.append(f"Split-representation check raised {split['error']['type']}: "
                            f"{split['error']['message']}"[:500])
        elif not split['passed']:
            problems.append('Split-representation check failed: the output changed by up to '
                            f"{split['max_abs_diff']:.3g} when the same economic prices were "
                            'passed as split-adjusted closes with unit split factors.')
        for lag in result['checks']['lag']:
            if 'error' in lag:
                problems.append(f"Lag check (k={lag['k']}) raised {lag['error']['type']}: "
                                f"{lag['error']['message']}"[:500])
            elif not lag['passed']:
                report = ' '.join(lag['summary'].split())[:300]
                problems.append(f"Lag check failed at k={lag['k']}: a position at or before "
                                f"session {lag['k']} changed when only prices at or after "
                                f'session {lag["k"]} were perturbed. Guard report: {report}')
    if not problems:
        return True, None
    return False, '\n'.join([head] + problems + [tail])


def freeze(root, only=None):
    """Freeze all 72 episodes; `only` (a set of ids) exists for software tests."""
    root.mkdir(parents=True, exist_ok=False)
    (root/'tmp').mkdir()
    context = library_context()
    rows = []
    for family, model, revision in MODELS:
        for task in TASKS:
            for seed in SEEDS:
                for arm in ARMS:
                    rows.append(dict(id=f'{family}-{task}-{seed}-{arm}', family=family,
                        model=model, revision=revision, task=task, seed=seed, arm=arm,
                        messages=messages(task, dataset(seed), arm, context)))
    random.Random(20260929).shuffle(rows)
    if only is not None:
        rows = [r for r in rows if r['id'] in only]
    write(root/'inputs.json', rows)
    write(root/'protocol.json', dict(version='codegen-enforced-v1',
        created_utc=datetime.now(timezone.utc).isoformat(), tasks=TASKS, seeds=SEEDS,
        models=MODELS, arms=ARMS, planned=len(rows), attempts=ATTEMPTS, max_tokens=2048,
        temperature=.1, top_p=1., parser=EXTRACTOR, lag_ks=LAG_KS, tolerance=1e-8,
        input_sha256=sha(root/'inputs.json'),
        primary_endpoint='valid deliverable: accepted by the arm controller and valid under '
                         'the unchanged E4 grader (execution, 1e-8 agreement, timing probes '
                         'k=32,64 including the same session, split invariance)',
        secondary=['grader validity of the final attempt regardless of acceptance',
                   'grader validity of the first attempt', 'acceptance', 'attempts used',
                   'traced library calls', 'tokens', 'generation and controller seconds'],
        limits=['Same E4 tasks, seeds, models and grader; E4 failures informed the controller '
                'checks, so this is not an independent holdout.',
                'Controller checks overlap grader properties (timing, split); only the grader '
                'checks numerical agreement with hidden references, with other perturbations.',
                'The first-complete-fence parser was chosen after E4 and applies to all arms.',
                'library_guards changes documentation, enforced library use and checks '
                'together; raw_guards isolates the checks.',
                'Synthetic author-designed rules; three seeds on four fixed rules are not '
                'twelve independent problems; two sizes of one model family.',
                'No financial-return or profitability endpoint; one shared GPU allocation.']))


QUALIFY = {
    'library_correct': ('library_guards', None, '''import pandas as pd
from fin_skills.algorithms import run
def solve(prices, split_factors):
    adj = prices * split_factors
    out = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for c in adj:
        out[c] = run('ma_crossover', {'prices': adj[c]}, fast=5, slow=20).fillna(0) / 4
    return out
'''),
    'raw_without_library': ('library_guards', 'Library requirement failed',
                            '''import numpy as np, pandas as pd
def solve(prices, split_factors):
    adj = prices * split_factors
    fast = adj.rolling(5).mean().shift(1)
    slow = adj.rolling(20).mean().shift(1)
    return (np.sign(fast - slow) / 4).fillna(0.0)
'''),
    'same_session': ('raw_guards', 'Lag check failed', '''import numpy as np, pandas as pd
def solve(prices, split_factors):
    adj = prices * split_factors
    return (np.sign(adj.rolling(5).mean() - adj.rolling(20).mean()) / 4).fillna(0.0)
'''),
    'recumulated_split': ('raw_guards', 'Split-representation check failed',
                          '''import numpy as np, pandas as pd
def solve(prices, split_factors):
    adj = prices * split_factors.cumprod()
    fast = adj.rolling(5).mean().shift(1)
    slow = adj.rolling(20).mean().shift(1)
    return (np.sign(fast - slow) / 4).fillna(0.0)
''')}


def qualify(root):
    """Sandbox and controller qualification on a non-task seed; no model is involved."""
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
    for name, (arm, finding, code) in QUALIFY.items():
        out = controller_execute(code, data, root, checks=True)
        accepted, message = decide(arm, out, 1)
        # None means the program must be accepted; otherwise the named finding must appear.
        ok = accepted if finding is None else (not accepted and finding in message)
        results[name] = dict(arm=arm, expected_finding=finding, accepted=accepted,
                             feedback=message, controller=strip(out), ok=ok)
        passed &= ok
    graded = execute(QUALIFY['library_correct'][2], dict(data, audit_guard=True), root)
    check = audit(QUALIFY['library_correct'][2], data, 'ma_crossover', graded, root)
    results['grader_positive_control'] = strip(dict(execution=graded, audit=check))
    passed &= bool(check.get('financially_valid'))
    write(root/'qualification.json', dict(passed=passed, results=results,
        scope='Software/controller qualification on seed 991 only; not model performance.'))
    if not passed:
        raise RuntimeError('Controller qualification failed')


def run(root, backend_factory=None):
    protocol = load(root/'protocol.json')
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
        job_id=os.environ.get('SLURM_JOB_ID'), **device))
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
                result = controller_execute(code, data, root, checks=item['arm'] != 'raw_exec')
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


def grade(code, data, task, root):
    initial = execute(code, dict(data, audit_guard=True), root)
    if initial.get('infrastructure_error'):
        # The frozen worker only catches Exception; a SystemExit or crash is the program's.
        initial = dict(executed=False, worker_wall_seconds=initial.get('worker_wall_seconds'),
                       error=dict(type='WorkerCrash', message=initial.get('stderr', '')[-500:]))
    return initial, audit(code, data, task, initial, root)


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
        usage = [r['response']['usage'] for r in records]
        final_valid = bool(final_audit.get('financially_valid'))
        result = dict(id=item['id'], family=item['family'], arm=item['arm'], task=item['task'],
            seed=item['seed'], attempts=len(records), accepted=rec['accepted'],
            valid_deliverable=bool(rec['accepted'] and final_valid), final_valid=final_valid,
            first_valid=bool(first_audit.get('financially_valid')),
            final_execution=strip(final_exec), final_audit=strip(final_audit),
            first_execution=strip(first_exec), first_audit=strip(first_audit),
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
        for arm in ARMS:
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
    args = parser.parse_args()
    globals()[args.mode](args.root)
