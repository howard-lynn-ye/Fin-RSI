"""Publish decision ledgers from frozen runs after a read-only accounting replay.

Normally run on a Slurm compute node. On October 7 the user also authorized a local
publication check using copied, completed records while Beacon's job quota was full.
Each worker imports the original archived source, not the new experiment source.
Never export prompts or evidence text, and never invoke inference or change decisions.
"""
import json
import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys

BASE = Path(os.environ.get('FIN_LEDGER_ARCHIVE_BASE',
                          '/beacon-projects/radfm/wy891/fin-multisource-models-20261005'))
REFERENCE_CAPITAL = 100_000.0


def worker(source, root, output):
    sys.path.insert(0, str(source))
    import numpy as np
    import pandas as pd
    from benchmarks.agent_study import multisource_model_study as study
    from benchmarks.agent_study.audit_manual_multisource import replay

    # Relocate the byte-identical source download; verify() still checks its
    # original frozen SHA256. This does not rebuild market data or alter hashes.
    study.md.RAW = BASE / 'r2/source/benchmarks/agent_study/data/market-us-20260929/yahoo_raw_20260929.json'
    load, sha = study.old.load, study.old.sha
    p, inputs = study.verify(root), study.validate_receipt(root)
    done, scores, audit = [load(root / n) for n in
        ('completed.json', 'scores.json', 'independent-model-audit.json')]
    assert done['scores_sha256'] == sha(root / 'scores.json')
    assert done['independent_audit_sha256'] == sha(root / 'independent-model-audit.json')
    assert audit['passed'] and audit['protocol_sha256'] == sha(root / 'protocol.json')
    assert audit['scores_sha256'] == done['scores_sha256']
    assert audit['inference_receipt_sha256'] == sha(root / 'inference-receipt.json')
    assert scores['inference_receipt_sha256'] == audit['inference_receipt_sha256']
    prices = pd.read_csv(root / 'data' / study.md.HIDDEN, index_col=0)[p['universe']]
    decisions, replay_checks, daily_nav = {}, {}, {}
    for arm in inputs['arms']:
        targets, rows = {}, []
        original_account = load(root / 'nav' / f'{arm}.json')
        original_nav = pd.Series(original_account['nav'], dtype=float)
        trades = original_account['trades']
        assert len(trades) == len(inputs['picks'])
        for i, pick in enumerate(inputs['picks']):
            name = f'decisions/{arm}/{i:02d}.json'
            record = load(root / name)
            execution = str(prices.index[pick + 1])
            assert (record['index'], record['date']) == (pick, str(prices.index[pick]))
            trade = trades[i]
            assert (trade['session'], trade['date']) == (pick + 1, execution)
            targets[execution] = record['target']
            row = dict(decision_number=i + 1, date=record['date'], execution_date=execution,
                target_weights=record['target'], submitted=record['submitted'],
                turn_count=len(record['turns']),
                failed_turns=sum(not t.get('turn_ok', t['result'].get('ok')) for t in record['turns']),
                attempted_tools=[t.get('tool') for t in record['turns']],
                turnover=trade['turnover'], fee_fraction=trade['cost'],
                decision_close_nav=float(original_nav.loc[record['date']]),
                execution_close_nav=float(original_nav.loc[execution]),
                execution_equity_reference_usd=REFERENCE_CAPITAL * float(original_nav.loc[execution]),
                cumulative_return_rate_pct=100 * (float(original_nav.loc[execution]) - 1),
                original_record_sha256=sha(root / name))
            if p.get('batch') == 'personal':
                row['authored_action_notes'] = [load((root / item['request']).with_name('response.json'))['rationale']
                                                for item in record['personal_response_receipts']]
            rows.append(row)
        assert len(rows) == scores['paths'][arm]['decisions'] == 44
        assert sum(r['submitted'] for r in rows) == scores['paths'][arm]['submitted']
        nav, detail = replay(prices.loc[next(iter(targets)):], targets, cost_bps=p['cost_bps'])
        assert original_nav.index[0] == str(prices.index[inputs['picks'][0]]) and original_nav.iloc[0] == 1.
        assert list(nav.index) == list(original_nav.index[1:])
        np.testing.assert_allclose(nav, original_nav.iloc[1:], atol=1e-12, rtol=0)
        np.testing.assert_allclose(detail['return_rate_pct'], scores['paths'][arm]['return_rate_pct'],
                                   atol=1e-10, rtol=0)
        decisions[arm] = rows
        daily_nav[arm] = original_nav.to_dict()
        replay_checks[arm] = dict(return_rate_pct=detail['return_rate_pct'],
            max_daily_nav_difference=float((nav - original_nav.iloc[1:]).abs().max()))
    public = dict(schema='fin-skills-frozen-pair-ledger-v1', interface=p['interface'],
        model=p['model'], seed=p['seed'], window=p['window'], paths=scores['paths'],
        return_difference_pp=scores['return_difference_pp'], decisions=decisions, daily_nav=daily_nav,
        reference_capital=dict(initial_usd=REFERENCE_CAPITAL,
            ending_usd={arm: REFERENCE_CAPITAL * list(values.values())[-1] for arm, values in daily_nav.items()},
            note='Presentation scaling of original unit NAV, not a new capital amount frozen in the v7 protocol.'),
        independent_replay=replay_checks, protocol_sha256=sha(root / 'protocol.json'),
        inference_receipt_sha256=sha(root / 'inference-receipt.json'),
        scores_sha256=sha(root / 'scores.json'), audit_sha256=sha(root / 'independent-model-audit.json'),
        limits=p['limits'], counts_as_additional_completed_model=False,
        scope=('Previously reported non-blind personal v7 case; now publishing its decision ledger.'
               if p.get('batch') == 'personal' else
               'One complete v7 seed pair; the other declared seeds are incomplete. Not v8/v9 repair results.'),
        omissions='Original prompts, raw evidence and per-turn response text are excluded from the public ledger.')
    with output.open('x') as stream:
        json.dump(public, stream, indent=2, allow_nan=False)
    fields = ['arm', 'decision_number', 'date', 'execution_date', 'submitted', 'turn_count', 'failed_turns',
              'turnover', 'fee_fraction', 'decision_close_nav', 'execution_close_nav',
              'execution_equity_reference_usd', 'cumulative_return_rate_pct', *p['universe']]
    with output.with_suffix('.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for arm, rows in decisions.items():
            for row in rows:
                item = {k: row[k] for k in fields if k in row}
                item.update(arm=arm)
                item.update(row['target_weights'] or {ticker: '' for ticker in p['universe']})
                writer.writerow(item)
    print(json.dumps(dict(output=output.name, sha256=sha(output),
                          return_rate_pct={a: scores['paths'][a]['return_rate_pct'] for a in inputs['arms']})))


def main(output, base=BASE):
    output.mkdir(exist_ok=False)
    cases = [
        ('personal-v7-v3/source', 'personal-v7-v3/pair', 'personal-v7-ledger.json'),
        ('matrix-replay-v2/source', 'matrix-replay-v2/pairs/mistral-small-24b-11', 'mistral-small-24b-seed11.json'),
    ]
    for source_name, root_name, destination in cases:
        source, root = base / source_name, base / root_name
        env = dict(os.environ, PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE='1',
                   FIN_LEDGER_ARCHIVE_BASE=str(base))
        subprocess.run([sys.executable, str(Path(__file__).resolve()), 'worker', str(source),
                        str(root), str(output / destination)], cwd=source, env=env, check=True)


if __name__ == '__main__':
    if len(sys.argv) == 5 and sys.argv[1] == 'worker':
        worker(*(Path(x) for x in sys.argv[2:]))
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('output', type=Path)
        parser.add_argument('--base', type=Path, default=BASE)
        args = parser.parse_args()
        main(args.output, args.base)
