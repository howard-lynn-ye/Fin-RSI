import json
from pathlib import Path

import pandas as pd
import pytest

from benchmarks.agent_study import trading_study_v4 as study
from benchmarks.agent_study import trading_study as old
from benchmarks.agent_study import market_data as md


def test_paired_trading_receipts_returns_and_resume(tmp_path, monkeypatch):
    """Exercise the entire inference/receipt/scoring chain with a known cash/share outcome."""
    root = tmp_path / 'pair'
    study.freeze(root, '7b', 11)
    monkeypatch.setattr(study, 'qualification', lambda root: {'passed': True})
    q = root / 'qualification' / 'qualification.json'
    old.write(q, {'test_only': True})
    seen = []

    class Backend:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, history):
            seen.append(history)
            return {'choices': [{'message': {'content':
                '{"tool":"submit","arguments":{"weights":{"SPY":1}}}'},
                'finish_reason': 'stop'}]}

    study.run(root, backend_factory=Backend)
    p = study.verify(root)
    assert len(seen) == p['planned_decisions'] == 88
    assert all('risk-adjusted return' in h[1]['content'] for h in seen)
    # Simulate interruption immediately before the inference receipt, not a rerun of decisions.
    (root / 'inference-receipt.json').unlink()
    seen.clear()
    study.run(root, backend_factory=Backend)
    assert not seen
    result = study.score(root)
    prices = pd.read_csv(root / 'data' / md.HIDDEN, index_col=0)
    picks = old.load(root / 'inputs.json')['decision_sessions']
    expected = ((1 - .0005) * prices.SPY.iloc[-1] / prices.SPY.iloc[picks[0] + 1] - 1) * 100
    for arm in ('raw', 'library'):
        assert result['paths'][arm]['return_rate_pct'] == pytest.approx(expected, abs=1e-10)
    assert result['return_difference_pp'] == pytest.approx(0)
    # The frozen protocol must reject changes to market data before any new inference.
    with (root / 'data' / 'quotes.csv').open('a') as stream:
        stream.write('\n')
    with pytest.raises(AssertionError):
        study.verify(root)


def test_incomplete_pairs_and_batches_never_get_a_return_score(tmp_path):
    root = tmp_path / 'pair'
    study.freeze(root, '14b', 23)
    with pytest.raises(FileNotFoundError):
        study.score(root)
    assert not (root / 'scores.json').exists()
    report = study.aggregate(tmp_path / 'batch')
    assert report['status'] == 'pending' and len(report['missing']) == 6
    assert not (tmp_path / 'batch' / 'aggregate.json').exists()


def test_aggregate_uses_same_three_seeds_and_reports_percentage_points(tmp_path):
    for family in study.FAMILIES:
        for seed in study.SEEDS:
            root = tmp_path / f'{family}-{seed}'
            old.write(root / 'scores.json', dict(family=family, seed=seed,
                paths={'raw': {'return_rate_pct': 10.}, 'library': {'return_rate_pct': 12.}}))
            old.write(root / 'completed.json', dict(scores_sha256=old.sha(root / 'scores.json')))
    report = study.aggregate(tmp_path)
    assert report['return_rate_pct']['7b'] == {'raw': 10., 'library': 12., 'difference_pp': 2.}
    assert report['return_rate_pct']['14b']['difference_pp'] == 2.
