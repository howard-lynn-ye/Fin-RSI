"""Small real-LLM paired test of financial computation, not a trading-alpha study.

Twelve episodes: 3 tasks x 2 seeds x 2 arms, one pinned model. Freeze before inference.
The private numerical reference uses vendor adjusted closes, not fin_skills output.
"""
import argparse
import json
from pathlib import Path
import random
import pandas as pd

from benchmarks.agent_study import market_data as md
from benchmarks.agent_study import trading_study as v3
from benchmarks.agent_study import trading_runtime_v4 as v4

TASKS = (
    dict(id='equal_weight_control', method='equal_weight', lookback=60,
         tickers=['SPY', 'TLT', 'GLD']),
    dict(id='inverse_vol_60', method='inverse_volatility', lookback=60,
         tickers=['SPY', 'NVDA', 'AVGO', 'TLT', 'GLD']),
    dict(id='inverse_vol_252', method='inverse_volatility', lookback=252,
         tickers=list(md.TICKERS)),
)
SEEDS = (11, 23)
DATE = '2024-07-15'


def source_hashes():
    files = ('trading_runtime_v4.py', 'trading_tools_v4.py', 'trading_worker_v4.py',
             'library_utility_pilot_v4.py', 'trading_study.py', 'trading_worker.py',
             'market_data.py', 'linux_sandbox.py', 'transformers_chat.py')
    return {name: v3.sha(Path(__file__).parent / name) for name in files}


def task_text(task):
    objective = ('equal weight over the specified tickers' if task['method'] == 'equal_weight'
                 else f"inverse daily volatility weights, using exactly the latest "
                 f"{task['lookback']} close-to-close returns (sample std, ddof=1), "
                 'normalized to sum to one')
    return (f"Financial calculation task as of {DATE}. Compute {objective}. "
            f"Tickers: {', '.join(task['tickers'])}. Use split- and cash-dividend-adjusted "
            'close returns; use multiplicative back-adjustment at corporate actions. '
            'No forecast, optimization of future return, or discretionary allocation is '
            'requested. Submit the numerical target weights; all other tickers must be zero. '
            'Numerical grading permits maximum absolute weight error 0.001. '
            'You may use the available tools; do not invent a result.')


def expected(task, total):
    if task['method'] == 'equal_weight':
        values = pd.Series(1 / len(task['tickers']), index=task['tickers'])
    else:
        returns = total.loc[:DATE, task['tickers']].tail(task['lookback'] + 1).pct_change()
        values = 1 / returns.iloc[1:].std(ddof=1)
        values /= values.sum()
    return {t: float(values.get(t, 0.)) for t in md.TICKERS}


def freeze(root, family='7b'):
    model = next((m for m in v3.MODELS if m[0] == family), None)
    if model is None or family not in ('7b', '14b'):
        raise ValueError('pilot supports only 7b or 14b')
    root.mkdir(parents=True, exist_ok=False)
    data_hashes = md.write_dataset(root / 'data')
    md.truncate(root / 'data', root / 'visible', DATE)
    rows = [dict(id=f"{task['id']}-{seed}-{arm}", task=task, seed=seed, arm=arm,
                 menu_seed=seed * 100 + i, prompt=task_text(task))
            for i, task in enumerate(TASKS) for seed in SEEDS for arm in ('raw', 'library')]
    random.Random(20261001).shuffle(rows)
    v3.write(root / 'inputs.json', rows)
    v3.write(root / 'protocol.json', dict(version='library-utility-pilot-v4-3',
        amendment='Pilot 1810695 stopped before grading: omitted arguments on zero-argument '
                  'tool calls were incorrectly rejected. Both dispatch paths now treat '
                  'missing/null arguments as an empty object. Tasks and scoring unchanged.',
        purpose='Real model numerical task utility, not trading returns or unseen markets',
        model=model, seeds=SEEDS, tasks=TASKS, date=DATE, episodes=len(rows),
        model_selection=('Original 7B development pilot' if family == '7b' else
            'Same-task 14B diagnostic selected after 7B basic controls failed to submit; '
            '7B numerical outcomes not inspected at selection. Retain both model reports.'),
        max_turns=8, max_tokens=1024, temperature=0.1, primary='correct numerical submission',
        tolerance=0.001, source_sha256=source_hashes(), data_sha256=data_hashes,
        input_sha256=v3.sha(root / 'inputs.json'),
        visible_sha256={n: v3.sha(root / 'visible' / n) for n in md.VISIBLE},
        common_prompt_sha256=__import__('hashlib').sha256(v4.COMMON.encode()).hexdigest(),
        library_prompt_sha256=__import__('hashlib').sha256(v4.LIBRARY.encode()).hexdigest(),
        oracle='vendor total-return close sample std or explicit equal weights; no library calls',
        limits=['Small development pilot, not a held-out general benchmark.',
                'Both arms know the requested algorithm. Library adds tools and documentation.',
                'Seeds share the same market history; no independence or significance claim.',
                'No investment-return claim; historical data may predate model training.']))


def verify(root):
    protocol = v3.load(root / 'protocol.json')
    if source_hashes() != protocol['source_sha256']:
        raise ValueError('source changed after freeze')
    assert v3.sha(root / 'inputs.json') == protocol['input_sha256']
    for name, digest in protocol['data_sha256'].items():
        assert v3.sha(root / 'data' / name) == digest
    for name, digest in protocol['visible_sha256'].items():
        assert v3.sha(root / 'visible' / name) == digest
    return protocol


def run(root):
    protocol = verify(root)
    if not (root / 'qualification' / 'qualification.json').exists():
        v4.qualify(root / 'qualification', root / 'data')
    qualification = v3.load(root / 'qualification' / 'qualification.json')
    if not qualification['passed']:
        raise ValueError('CPU qualification failed; inference prohibited')
    if (root / 'inference-receipt.json').exists():
        raise ValueError('completed inference exists')
    # Receipt is written before loading the model or inspecting any task scores.
    if not (root / 'inference-started.json').exists():
        import os
        v3.write(root / 'inference-started.json', dict(protocol_sha256=v3.sha(root/'protocol.json'),
                  qualification_sha256=v3.sha(root/'qualification'/'qualification.json'),
                  job_id=os.environ.get('SLURM_JOB_ID')))
    from benchmarks.agent_study.transformers_chat import TransformersChat
    _, model, revision = protocol['model']
    backend = TransformersChat(model, revision, max_tokens=protocol['max_tokens'])
    rows = v3.load(root / 'inputs.json')
    for row in rows:
        output = root / 'decisions' / (row['id'] + '.json')
        if output.exists():
            continue
        backend.seed, backend.calls = row['seed'], 0
        controller = v4.Controller(root, root / 'visible', row['arm'], row['menu_seed'])
        record = v4.decide(backend, controller, row['prompt'])
        v3.write(output, dict(id=row['id'], **record))
        print(json.dumps(dict(id=row['id'], submitted=record['submitted'])), flush=True)
    v3.write(root / 'inference-receipt.json', dict(protocol_sha256=v3.sha(root/'protocol.json'),
        decisions={r['id']: v3.sha(root/'decisions'/(r['id']+'.json')) for r in rows}))


def score(root):
    protocol = verify(root)
    receipt = v3.load(root / 'inference-receipt.json')
    assert receipt['protocol_sha256'] == v3.sha(root / 'protocol.json')
    total = pd.read_csv(root / 'data' / md.HIDDEN, index_col=0)
    scored = []
    for row in v3.load(root / 'inputs.json'):
        file = root / 'decisions' / (row['id'] + '.json')
        assert v3.sha(file) == receipt['decisions'][row['id']]
        record = v3.load(file)
        oracle = expected(row['task'], total)
        target = record['target']
        gap = max(abs(target[t] - oracle[t]) for t in md.TICKERS) if target else None
        scored.append(dict(id=row['id'], arm=row['arm'], task=row['task']['id'], seed=row['seed'],
            submitted=target is not None, max_weight_error=gap,
            correct=gap is not None and gap <= protocol['tolerance'],
            tool_errors=sum(not c['ok'] for c in record['tool_calls']),
            parse_errors=sum(t['parse'] == 'unparsed' for t in record['turns']),
            turns=len(record['turns']),
            generation_seconds=sum(t.get('generation_seconds', 0.) for t in record['turns']),
            completion_tokens=sum((t.get('usage') or {}).get('completion_tokens', 0)
                                  for t in record['turns']),
            prompt_tokens=sum((t.get('usage') or {}).get('prompt_tokens', 0)
                              for t in record['turns'])))
    summary = {arm: dict(correct=sum(r['correct'] for r in scored if r['arm'] == arm),
                        total=sum(r['arm'] == arm for r in scored),
                        submitted=sum(r['submitted'] for r in scored if r['arm'] == arm),
                        tool_errors=sum(r['tool_errors'] for r in scored if r['arm'] == arm))
               for arm in ('raw', 'library')}
    result = dict(summary=summary, rows=scored,
                  inference_receipt_sha256=v3.sha(root/'inference-receipt.json'),
                  limits=protocol['limits'])
    v3.write(root / 'scores.json', result)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['freeze', 'run', 'score'])
    parser.add_argument('root', type=Path)
    parser.add_argument('--family', choices=('7b', '14b'),
                        help='model family, only valid for freeze; default 7b')
    args = parser.parse_args()
    if args.mode == 'freeze':
        freeze(args.root.resolve(), args.family or '7b')
    elif args.family is not None:
        parser.error('--family is only valid for freeze; run/score use the frozen protocol')
    else:
        globals()[args.mode](args.root.resolve())
