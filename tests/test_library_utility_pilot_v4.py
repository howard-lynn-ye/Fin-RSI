import json

import pandas as pd
import pytest

from benchmarks.agent_study import library_utility_pilot_v4 as pilot
from benchmarks.agent_study import market_data as md
from benchmarks.agent_study.trading_tools_v4 import Tools


def test_freeze_pairs_identical_tasks_and_detects_input_changes(tmp_path):
    root = tmp_path / 'pilot'
    pilot.freeze(root)
    protocol = pilot.verify(root)
    rows = json.loads((root / 'inputs.json').read_text())
    assert protocol['episodes'] == len(rows) == 12
    for task in pilot.TASKS:
        for seed in pilot.SEEDS:
            pair = [r for r in rows if r['task'] == task and r['seed'] == seed]
            assert {r['arm'] for r in pair} == {'raw', 'library'}
            assert pair[0]['prompt'] == pair[1]['prompt']
            assert pair[0]['menu_seed'] == pair[1]['menu_seed']
    (root / 'visible' / 'README.md').write_text('changed')
    with pytest.raises(AssertionError):
        pilot.verify(root)


def test_library_adapter_matches_independent_vendor_price_oracle(tmp_path):
    md.write_dataset(tmp_path / 'data')
    md.truncate(tmp_path / 'data', tmp_path / 'visible', pilot.DATE)
    total = pd.read_csv(tmp_path / 'data' / md.HIDDEN, index_col=0)
    tools = Tools(tmp_path / 'visible', 'library')
    for task in pilot.TASKS:
        result = tools.run_algorithm(task['method'], task['tickers'], task['lookback'])
        assert result['ok']
        oracle = pilot.expected(task, total)
        weights = result['result']['weights']
        assert max(abs(weights.get(t, 0) - oracle[t]) for t in md.TICKERS) < 1e-5
    # Unadjusted split returns are meaningfully distinguishable by the scoring tolerance.
    quotes, _ = md.load_visible(tmp_path / 'visible')
    close = quotes.pivot(index='date', columns='ticker', values='close')
    task = pilot.TASKS[1]
    raw = close[task['tickers']].tail(task['lookback'] + 1).pct_change().iloc[1:]
    wrong = 1 / raw.std(ddof=1)
    wrong /= wrong.sum()
    oracle = pilot.expected(task, total)
    assert max(abs(wrong[t] - oracle[t]) for t in task['tickers']) > 0.001
