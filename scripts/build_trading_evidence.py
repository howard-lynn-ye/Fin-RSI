"""Generate manuscript macros for the out-of-sample trading study from hash-checked evidence.

Reads each fetched job folder under benchmarks/agent_study/evidence/20260929-trading-study/
results-v3/<family>/, re-checks fetch receipt -> protocol -> inputs -> every decision file ->
inference receipt -> scores, re-prices every path with the study's own ledger, and writes
paper/latex_naacl/trading_numbers.tex. `--check` fails if that file is stale.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.agent_study import trading_study as ts  # noqa: E402

RESULTS = ROOT/'benchmarks/agent_study/evidence/20260929-trading-study/results-v3'
OUTPUT = ROOT/'paper/latex_naacl/trading_numbers.tex'
FAMILY = {'7b': 'Seven', '14b': 'Fourteen', '32b': 'ThirtyTwo'}
ARM = {'raw': 'Raw', 'library': 'Lib'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verified(folder, verify_fetch=True):
    """Return scores after re-checking the hash chain and re-pricing every path."""
    if verify_fetch:
        fetch = load(folder/'fetch-receipt.json')
        assert fetch['complete'] and fetch['all_match'], folder
    protocol = load(folder/'protocol.json')
    assert sha(folder/'inputs.json') == protocol['input_sha256']
    for name, digest in protocol['data_sha256'].items():
        assert sha(folder/'data'/name) == digest, name
    receipt = load(folder/'inference-receipt.json')
    assert receipt['protocol_sha256'] == sha(folder/'protocol.json')
    assert len(receipt['rows']) == receipt['planned_paths'] == protocol['planned_paths']
    scores = load(folder/'scores.json')
    assert scores['inference_receipt_sha256'] == sha(folder/'inference-receipt.json')
    total = pd.read_csv(folder/'data'/ts.HIDDEN, index_col=0)
    picks = load(folder/'inputs.json')['decision_sessions']
    for row in receipt['rows']:
        assert len(row['decisions']) == receipt['decisions_per_path'] == len(picks)
        targets = []
        for item in row['decisions']:
            assert sha(folder/item['path']) == item['sha256'], item['path']
            targets.append(load(folder/item['path'])['target'])
        nav, _ = ts.ledger(total, picks, targets)
        again = ts.metrics(nav)
        stored = scores['paths'][row['id']]['metrics']
        gap = abs(again['cumulative_return'] - stored['cumulative_return'])
        assert gap < 1e-9, row['id']
    return protocol, scores


def bootstrap(diffs, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs, dtype=float)
    means = rng.choice(diffs, (n, len(diffs)), replace=True).mean(axis=1)
    return (float(diffs.mean()), float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)))


def macros(results=RESULTS, verify_fetch=True):
    out, pooled, done = {}, [], []
    base = None
    for family, label in FAMILY.items():
        folder = results/family
        if not (folder/'scores.json').exists():
            continue
        protocol, scores = verified(folder, verify_fetch)
        done.append(family)
        base = base or scores
        paths = scores['paths']
        for arm, arm_label in ARM.items():
            group = [p for p in paths.values() if p['arm'] == arm]
            name = 'TR' + label + arm_label
            mean = lambda key: float(np.mean([p['metrics'][key] for p in group]))
            out[name+'Return'] = f"{100*mean('cumulative_return'):.1f}"
            out[name+'Sharpe'] = f"{mean('sharpe'):.2f}"
            out[name+'MaxDD'] = f"{100*mean('max_drawdown'):.1f}"
            rate = np.mean([p['submitted'] / p['decisions'] for p in group])
            calls = np.mean([p['library_calls'] / p['decisions'] for p in group])
            out[name+'SubRate'] = f'{100*rate:.1f}'
            out[name+'LibCalls'] = f'{calls:.2f}'
            out[name+'Turnover'] = f"{100*np.mean([p['mean_turnover'] for p in group]):.1f}"
            out[name+'Turns'] = f"{np.mean([p['mean_turns'] for p in group]):.1f}"
        diffs, wins = [], 0
        for seed in protocol['seeds']:
            lib, raw = paths[f'{family}-{seed}-library'], paths[f'{family}-{seed}-raw']
            diffs += list(np.array(lib['blocks']) - np.array(raw['blocks']))
            wins += lib['metrics']['cumulative_return'] > raw['metrics']['cumulative_return']
        pooled += diffs
        mean, low, high = bootstrap(diffs)
        name = 'TR' + label
        out[name+'Wins'] = wins
        out[name+'Seeds'] = len(protocol['seeds'])
        gap = float(out[name+'LibReturn']) - float(out[name+'RawReturn'])
        out[name+'ReturnDiff'] = f'{gap:+.1f}'
        gap = float(out[name+'LibSharpe']) - float(out[name+'RawSharpe'])
        out[name+'SharpeDiff'] = f'{gap:+.2f}'
        out[name+'PeriodDiffBps'] = f'{1e4*mean:+.1f}'
        out[name+'PeriodDiffLowBps'] = f'{1e4*low:+.1f}'
        out[name+'PeriodDiffHighBps'] = f'{1e4*high:+.1f}'
    if base is not None:
        labels = {'cash': 'Cash', 'equal_weight_buy_and_hold': 'EwHold',
                  'equal_weight_rebalanced': 'EwReb', 'sixty_forty_rebalanced': 'SixtyForty'}
        for key, label in labels.items():
            row = base['baselines'][key]
            out['TRBase'+label+'Return'] = f"{100*row['cumulative_return']:.1f}"
            out['TRBase'+label+'Sharpe'] = f"{row['sharpe']:.2f}"
        out['TRWindowStart'], out['TRWindowEnd'] = base['window']
        out['TRDecisions'] = len(json.loads((results/done[0]/'inputs.json').read_text())
                                 ['decision_sessions'])
    if pooled:
        mean, low, high = bootstrap(pooled)
        out['TRPooledPeriodDiffBps'] = f'{1e4*mean:+.1f}'
        out['TRPooledPeriodDiffLowBps'] = f'{1e4*low:+.1f}'
        out['TRPooledPeriodDiffHighBps'] = f'{1e4*high:+.1f}'
        out['TRFamiliesDone'] = len(done)
    return out


def render(results=RESULTS):
    lines = ['% GENERATED by scripts/build_trading_evidence.py from hash-checked evidence.']
    lines += [f'\\newcommand{{\\{k}}}{{{v}}}' for k, v in macros(results).items()]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    text = render()
    if args.check:
        current = OUTPUT.read_text(encoding='utf-8') if OUTPUT.exists() else ''
        if current != text:
            sys.exit('trading_numbers.tex is stale; run scripts/build_trading_evidence.py')
        print('trading evidence macros up to date')
        return
    OUTPUT.write_text(text, encoding='utf-8', newline='\n')
    print(f'wrote {OUTPUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
