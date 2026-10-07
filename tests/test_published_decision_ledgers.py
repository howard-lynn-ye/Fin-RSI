"""Keep public decision rows, CSV exports and original return receipts consistent."""
import csv
import hashlib
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / 'benchmarks/agent_study/reports/20261007-additional-records'
HASHES = {
    'personal-v7-ledger': '1e34c9ffa9b422e3c311e9273a18943b72412c0a7facf345d46d1c1e1ca4d19e',
    'mistral-small-24b-seed11': '359350f983840100b88af2363286d1bc21f1f8774b4b6b9394a6f5e2e01e5488',
}


@pytest.mark.parametrize('name', HASHES)
def test_original_88_decisions_survive_publication(name):
    path = ROOT / (name + '.json')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == HASHES[name]
    data = json.loads(path.read_text(encoding='utf-8'))
    assert data['interface'] == 'v7' and not data['counts_as_additional_completed_model']
    assert set(data['decisions']) == {'raw', 'library'}
    with path.with_suffix('.csv').open(encoding='utf-8', newline='') as stream:
        table = list(csv.DictReader(stream))
    assert len(table) == 88
    for arm, rows in data['decisions'].items():
        nav = data['daily_nav'][arm]
        score = data['paths'][arm]
        assert len(rows) == score['decisions'] == 44
        assert [r['decision_number'] for r in rows] == list(range(1, 45))
        assert len({r['date'] for r in rows}) == 44
        assert sum(r['submitted'] for r in rows) == score['submitted']
        assert sum(r['failed_turns'] for r in rows) == score['failed_turns']
        assert all(math.isfinite(v) and v > 0 for v in nav.values())
        assert next(iter(nav.values())) == 1.
        assert 100 * (list(nav.values())[-1] - 1) == pytest.approx(score['return_rate_pct'], abs=1e-10)
        assert data['independent_replay'][arm]['return_rate_pct'] == pytest.approx(score['return_rate_pct'], abs=1e-10)
        assert data['reference_capital']['ending_usd'][arm] == pytest.approx(100000 * list(nav.values())[-1])
        exported = [r for r in table if r['arm'] == arm]
        for row, flat in zip(rows, exported):
            assert flat['date'] == row['date'] < row['execution_date'] == flat['execution_date']
            assert row['decision_close_nav'] == nav[row['date']]
            assert row['execution_close_nav'] == nav[row['execution_date']]
            assert row['execution_equity_reference_usd'] == 100000 * row['execution_close_nav']
            assert row['fee_fraction'] == pytest.approx(row['turnover'] * .0005, abs=1e-15)
            assert flat['submitted'] == str(row['submitted'])
            assert float(flat['execution_close_nav']) == row['execution_close_nav']
            if row['target_weights'] is not None:
                assert row['submitted'] and sum(row['target_weights'].values()) <= 1 + 1e-12
                for ticker, value in row['target_weights'].items():
                    assert math.isfinite(value) and value >= 0 and float(flat[ticker]) == value
            else:
                assert not row['submitted'] and row['turnover'] == row['fee_fraction'] == 0.
            if name == 'personal-v7-ledger':
                assert len(row['authored_action_notes']) == row['turn_count']
    assert data['return_difference_pp'] == pytest.approx(
        data['paths']['library']['return_rate_pct'] - data['paths']['raw']['return_rate_pct'], abs=1e-10)
