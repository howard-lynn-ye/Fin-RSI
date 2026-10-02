"""Frozen repair rerun: 7B/14B, original three seeds and market window, Return Rate (%).

Each independently scheduled job runs one model/seed with both arms. No future returns
are reported to the agent. Completed paired jobs can be aggregated only as a full batch.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import pandas as pd

from benchmarks.agent_study import market_data as md
from benchmarks.agent_study import trading_runtime_v4 as runtime
from benchmarks.agent_study import trading_study as old

FAMILIES = ('7b', '14b')
SEEDS = (11, 23, 37)
REPO = Path(__file__).resolve().parents[2]


def sources():
    names = ('trading_study_v4.py', 'trading_runtime_v4.py', 'trading_tools_v4.py',
             'trading_worker_v4.py', 'trading_study.py', 'trading_worker.py',
             'market_data.py', 'linux_sandbox.py', 'transformers_chat.py')
    paths = [Path(__file__).parent / n for n in names]
    paths += [p for p in (REPO / 'fin_skills').rglob('*')
              if p.is_file() and p.suffix in ('.py', '.json', '.md')]
    return {p.relative_to(REPO).as_posix(): old.sha(p) for p in sorted(paths)}


def task(date, index, holdings):
    return (f"You manage a long-only portfolio of these instruments: {', '.join(md.TICKERS)}. "
        f"Today is {date} (session {index}). Current holdings after price drift: "
        f"{json.dumps(holdings)}. Target weights execute at the NEXT session's close, "
        f"after that session's return, and are held for {old.STEP} sessions. Trading costs "
        f"{old.COST_BPS:g} bps per side on traded notional. Objective: maximize net "
        'risk-adjusted return over the holding period; there is no benchmark to track and '
        'cash earns zero. Decide the target weights for the next holding period. '
        'Use the tools, then submit. A failed or missing submission retains existing holdings.')


def freeze(root, family, seed):
    if family not in FAMILIES or seed not in SEEDS:
        raise ValueError('only the declared 7B/14B, seeds 11/23/37 repair rerun is supported')
    root.mkdir(parents=True, exist_ok=False)
    hashes = md.write_dataset(root / 'data')
    total = pd.read_csv(root / 'data' / md.HIDDEN, index_col=0)
    picks = old.schedule(list(total.index))
    model = next(m for m in old.MODELS if m[0] == family)
    arms = list(old.ARMS)
    random.Random(20261001 + seed).shuffle(arms)
    old.write(root / 'inputs.json', dict(arms=arms, decision_sessions=picks,
        decision_dates=[total.index[i] for i in picks]))
    old.write(root / 'protocol.json', dict(version='trading-repair-v4-1',
        created_utc=datetime.now(timezone.utc).isoformat(), model=model, seed=seed,
        universe=md.TICKERS, cost_bps=old.COST_BPS, step=old.STEP,
        max_turns=old.MAX_TURNS, max_tokens=old.MAX_TOKENS, temperature=.1,
        planned_decisions=len(picks) * len(arms), decisions_per_arm=len(picks),
        input_sha256=old.sha(root / 'inputs.json'), data_sha256=hashes,
        source_sha256=sources(), raw_download_sha256=old.sha(md.RAW),
        window=[total.index[picks[0] + 1], total.index[-1]],
        primary_endpoint='Cumulative net Return Rate (%) from initial capital before first '
            'trade, paired library minus raw in percentage points; mean of three seeds.',
        objective='Original net risk-adjusted trading objective retained to avoid changing '
            'the optimization goal while repairing the interface. Return Rate is the '
            'primary reported endpoint; Sharpe and drawdown are secondary.',
        treatment='Library tools, method documentation and adjusted-data adapters together.',
        changes=['Common JSON/Python functions and Python submit; explicit API instructions.',
                 'Single Python fences accepted with surrounding prose; no code repair.',
                 'HRP adapter linkage default; seeded menu order; rounded weights normalized.',
                 'Future-source and private-ledger read denial; first fee included.'],
        limits=['Original window is now development data, not a new unseen holdout.',
                'Three sampling seeds share one market path; no independent-market or '
                'statistical-significance claim.',
                'This measures the combined repair bundle, not each individual fix.',
                'Do not compare corrected returns with uncorrected v3 as a causal estimate.',
                'Long-only, zero cash yield, 5 bps per side; no extra slippage or capacity.',
                'Do not tune strategy, prompts, tasks or models after observing this batch.']))


def verify(root):
    p = old.load(root / 'protocol.json')
    if p['source_sha256'] != sources():
        raise ValueError('source/package/documentation changed after freeze')
    assert old.sha(root / 'inputs.json') == p['input_sha256']
    assert old.sha(md.RAW) == p['raw_download_sha256']
    for name, digest in p['data_sha256'].items():
        assert old.sha(root / 'data' / name) == digest, name
    return p


def qualification(root):
    q = old.load(root / 'qualification' / 'qualification.json')
    assert q['passed'] and all(r['passed'] for r in q['results'].values())
    current = {p.name: old.sha(p) for p in Path(__file__).parent.glob('trading_*v4.py')}
    assert q['source_sha256'] == current, 'qualification source changed'
    return q


def run(root, backend_factory=None):
    p = verify(root)
    qualification(root)
    if (root / 'inference-receipt.json').exists():
        raise FileExistsError('completed pair already exists')
    started = root / 'inference-started.json'
    if started.exists():
        assert old.load(started)['protocol_sha256'] == old.sha(root / 'protocol.json')
    else:
        old.write(started, dict(protocol_sha256=old.sha(root / 'protocol.json'),
            qualification_sha256=old.sha(root / 'qualification' / 'qualification.json'),
            job_id=os.environ.get('SLURM_JOB_ID'),
            started_utc=datetime.now(timezone.utc).isoformat()))
    if backend_factory is None:
        from benchmarks.agent_study.transformers_chat import TransformersChat
        backend_factory = TransformersChat
    _, model, revision = p['model']
    backend = backend_factory(model, revision, max_tokens=p['max_tokens'])
    inputs = old.load(root / 'inputs.json')
    total = pd.read_csv(root / 'data' / md.HIDDEN, index_col=0)
    returns = total.pct_change().fillna(0.)
    picks = inputs['decision_sessions']
    receipts = {}
    (root / 'tmp').mkdir(exist_ok=True)
    for arm in inputs['arms']:
        holdings = dict.fromkeys(md.TICKERS, 0.)
        for i, pick in enumerate(picks):
            file = root / 'decisions' / arm / f'{i:02d}.json'
            date = total.index[pick]
            if file.exists():
                record = old.load(file)
                assert record['date'] == date and record['index'] == pick
                assert max(abs(record['holdings_before'][t] - holdings[t])
                           for t in md.TICKERS) < 1e-12
            else:
                backend.seed, backend.calls = p['seed'] * 1000 + i, 0
                with tempfile.TemporaryDirectory(dir=root / 'tmp') as td:
                    workspace = Path(td) / 'visible'
                    md.truncate(root / 'data', workspace, date)
                    controller = runtime.Controller(root, workspace, arm, p['seed'] * 1000 + i)
                    record = runtime.decide(backend, controller, task(date, pick, holdings))
                record.update(date=date, index=pick, holdings_before=dict(holdings))
                old.write(file, record)
                print(json.dumps(dict(family=p['model'][0], seed=p['seed'], arm=arm,
                    decision=i + 1, planned=len(picks), submitted=record['submitted'])), flush=True)
            receipts[file.relative_to(root).as_posix()] = old.sha(file)
            end = picks[i + 1] + 1 if i + 1 < len(picks) else pick + 1
            for k in range(pick + 1, end):
                holdings, _ = old.drift(holdings, returns.iloc[k])
                if k == pick + 1 and record['target'] is not None:
                    holdings = dict(record['target'])
    assert len(receipts) == p['planned_decisions']
    old.write(root / 'inference-receipt.json', dict(protocol_sha256=old.sha(root / 'protocol.json'),
                                                  decisions=receipts))


def score(root):
    p = verify(root)
    receipt = old.load(root / 'inference-receipt.json')
    assert receipt['protocol_sha256'] == old.sha(root / 'protocol.json')
    inputs = old.load(root / 'inputs.json')
    total = pd.read_csv(root / 'data' / md.HIDDEN, index_col=0)
    picks = inputs['decision_sessions']
    expected = {f'decisions/{a}/{i:02d}.json' for a in inputs['arms'] for i in range(len(picks))}
    assert expected == set(receipt['decisions'])
    assert expected == {f.relative_to(root).as_posix() for f in (root / 'decisions').glob('*/*.json')}
    paths = {}
    for arm in inputs['arms']:
        records = []
        for i in range(len(picks)):
            name = f'decisions/{arm}/{i:02d}.json'
            assert old.sha(root / name) == receipt['decisions'][name]
            records.append(old.load(root / name))
        nav, trades = runtime.ledger(total, picks, [r['target'] for r in records])
        metrics = runtime.metrics(nav)
        paths[arm] = dict(return_rate_pct=100 * metrics['cumulative_return'], metrics=metrics,
            submitted=sum(r['submitted'] for r in records), decisions=len(records),
            failed_turns=sum(not t['result'].get('ok') for r in records for t in r['turns']),
            total_turns=sum(len(r['turns']) for r in records),
            prompt_tokens=sum((t.get('usage') or {}).get('prompt_tokens', 0)
                             for r in records for t in r['turns']),
            completion_tokens=sum((t.get('usage') or {}).get('completion_tokens', 0)
                                 for r in records for t in r['turns']))
        old.write(root / 'nav' / f'{arm}.json', dict(nav=nav.to_dict(), trades=trades))
    ew = dict.fromkeys(md.TICKERS, 1 / len(md.TICKERS))
    balanced = dict.fromkeys(md.TICKERS, 0.)
    balanced.update(SPY=.6, IEF=.4)
    inv = []
    for pick in picks:
        ret = total.iloc[:pick + 1].tail(253).pct_change().iloc[1:]
        w = 1 / ret.std(ddof=1)
        inv.append((w / w.sum()).to_dict())
    base = {}
    for name, targets in dict(cash=[None] * len(picks), equal_weight_rebalanced=[ew] * len(picks),
            equal_weight_buy_hold=[ew] + [None] * (len(picks) - 1),
            sixty_forty_rebalanced=[balanced] * len(picks), inverse_vol_252=inv).items():
        nav, _ = runtime.ledger(total, picks, targets)
        base[name] = 100 * runtime.metrics(nav)['cumulative_return']
    result = dict(family=p['model'][0], seed=p['seed'], window=p['window'], paths=paths,
        return_difference_pp=paths['library']['return_rate_pct'] - paths['raw']['return_rate_pct'],
        baselines_return_pct=base, inference_receipt_sha256=old.sha(root / 'inference-receipt.json'),
        limits=p['limits'])
    old.write(root / 'scores.json', result)
    old.write(root / 'completed.json', dict(scores_sha256=old.sha(root / 'scores.json'),
        completed_utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(result), flush=True)
    return result


def aggregate(batch):
    roots = [batch / f'{f}-{s}' for f in FAMILIES for s in SEEDS]
    missing = [r.name for r in roots if not (r / 'completed.json').exists()]
    if missing:
        return dict(status='pending', missing=missing)
    rows = []
    for root in roots:
        assert old.load(root / 'completed.json')['scores_sha256'] == old.sha(root / 'scores.json')
        rows.append(old.load(root / 'scores.json'))
    summary = {}
    for family in FAMILIES:
        group = [r for r in rows if r['family'] == family]
        assert sorted(r['seed'] for r in group) == list(SEEDS)
        summary[family] = {a: float(np.mean([r['paths'][a]['return_rate_pct'] for r in group]))
                           for a in ('raw', 'library')}
        summary[family]['difference_pp'] = summary[family]['library'] - summary[family]['raw']
    result = dict(status='complete', return_rate_pct=summary, seed_results=rows,
                  interpretation='Three seeds on one shared development market window; '
                                 'no significance or new-market claim.')
    try:
        old.write(batch / 'aggregate.json', result)
    except FileExistsError:
        pass
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'qualify', 'run', 'score', 'aggregate'))
    parser.add_argument('root', type=Path)
    parser.add_argument('--family', choices=FAMILIES)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == 'freeze':
        freeze(root, args.family, args.seed)
    elif args.mode == 'qualify':
        verify(root)
        report = runtime.qualify(root / 'qualification', root / 'data')
        assert report['passed']
    elif args.mode == 'aggregate':
        print(json.dumps(aggregate(root)))
    else:
        globals()[args.mode](root)
