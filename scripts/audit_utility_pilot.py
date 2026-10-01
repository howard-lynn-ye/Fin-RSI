"""Independently verify a completed utility pilot without importing its scorer or library.

The input directory contains the frozen pilot plus source_snapshot/<filename>.
Uses stdlib CSV and sample standard deviation on vendor total-return closes.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def audit(root):
    protocol = read(root / 'protocol.json')
    receipt = read(root / 'inference-receipt.json')
    started = read(root / 'inference-started.json')
    scores = read(root / 'scores.json')
    assert digest(root / 'protocol.json') == receipt['protocol_sha256']
    assert receipt['protocol_sha256'] == started['protocol_sha256']
    assert digest(root / 'inference-receipt.json') == scores['inference_receipt_sha256']
    assert digest(root / 'inputs.json') == protocol['input_sha256']
    q = root / 'qualification' / 'qualification.json'
    assert read(q)['passed'] and digest(q) == started['qualification_sha256']
    for filename, expected in protocol['source_sha256'].items():
        assert digest(root / 'source_snapshot' / filename) == expected, filename
    for section, directory in (('data_sha256', 'data'), ('visible_sha256', 'visible')):
        for filename, expected in protocol[section].items():
            assert digest(root / directory / filename) == expected, filename
    inputs = read(root / 'inputs.json')
    ids = {r['id'] for r in inputs}
    assert len(inputs) == len(ids) == protocol['episodes'] == 12
    assert ids == set(receipt['decisions']) == {r['id'] for r in scores['rows']}
    assert ids == {p.stem for p in (root / 'decisions').glob('*.json')}
    with (root / 'data' / 'total_return_close.csv').open(newline='') as stream:
        reader = csv.DictReader(stream)
        date_column = reader.fieldnames[0]
        prices = [r for r in reader if r[date_column] <= protocol['date']]
    score_by_id = {r['id']: r for r in scores['rows']}
    rows = []
    for item in inputs:
        decision_path = root / 'decisions' / (item['id'] + '.json')
        assert digest(decision_path) == receipt['decisions'][item['id']]
        record = read(decision_path)
        task, target = item['task'], record['target']
        peer = [p for p in inputs if p['task'] == task and p['seed'] == item['seed']]
        assert len(peer) == 2 and {p['arm'] for p in peer} == {'raw', 'library'}
        assert peer[0]['prompt'] == peer[1]['prompt']
        assert peer[0]['menu_seed'] == peer[1]['menu_seed']
        if task['method'] == 'equal_weight':
            expected = dict.fromkeys(task['tickers'], 1 / len(task['tickers']))
        else:
            assert task['method'] == 'inverse_volatility'
            window = prices[-(task['lookback'] + 1):]
            assert len(window) == task['lookback'] + 1
            inv = {}
            for ticker in task['tickers']:
                p = [float(r[ticker]) for r in window]
                assert all(math.isfinite(v) and v > 0 for v in p)
                inv[ticker] = 1 / statistics.stdev(b / a - 1 for a, b in zip(p, p[1:]))
            expected = {ticker: v / sum(inv.values()) for ticker, v in inv.items()}
        if target is None:
            gap = None
        else:
            assert all(math.isfinite(v) and v >= 0 for v in target.values())
            assert sum(target.values()) <= 1 + 1e-12
            gap = max(abs(target.get(t, 0) - expected.get(t, 0))
                      for t in set(target) | set(expected))
        correct = gap is not None and gap <= protocol['tolerance']
        scored = score_by_id[item['id']]
        assert correct == scored['correct']
        assert (gap is None) == (scored['max_weight_error'] is None)
        if gap is not None:
            assert abs(gap - scored['max_weight_error']) < 1e-12
        assert scored['submitted'] == record['submitted'] == (target is not None)
        rows.append(dict(id=item['id'], arm=item['arm'], task=task['id'],
                         submitted=target is not None, correct=correct, max_weight_error=gap))
    summary = {a: dict(correct=sum(r['correct'] for r in rows if r['arm'] == a),
                       submitted=sum(r['submitted'] for r in rows if r['arm'] == a),
                       total=sum(r['arm'] == a for r in rows)) for a in ('raw', 'library')}
    for arm, values in summary.items():
        assert all(scores['summary'][arm][k] == v for k, v in values.items())
    return dict(verified=True, method='stdlib csv + statistics.stdev, no scorer/library imports',
                model=protocol['model'], summary=summary, rows=rows,
                protocol_sha256=digest(root / 'protocol.json'),
                inference_receipt_sha256=digest(root / 'inference-receipt.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.root)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report['summary']))
