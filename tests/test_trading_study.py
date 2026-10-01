import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmarks.agent_study import market_data as md
from benchmarks.agent_study import trading_study as ts
from benchmarks.agent_study.linux_sandbox import confinement_available

ROOT = Path(__file__).resolve().parents[1]
linux_only = pytest.mark.skipif(not confinement_available(),
                                reason='real confinement needs Linux Landlock and libseccomp')


@pytest.fixture
def package_path(monkeypatch):
    monkeypatch.setenv('PYTHONPATH', str(ROOT))


def test_visible_files_reproduce_total_return_from_the_raw_download():
    quotes, actions = md.load_visible(md.DATA_DIR)
    hidden = pd.read_csv(md.DATA_DIR/md.HIDDEN, index_col=0)
    rebuilt = md.adjust(quotes, actions)
    diff = (rebuilt.pct_change().iloc[1:] - hidden.pct_change().iloc[1:]).abs()
    assert float(diff.max().max()) < 5e-6
    orly = quotes[quotes['ticker'] == 'ORLY'].set_index('date')['close']
    assert orly['2025-06-09'] > 1000 and orly['2025-06-10'] < 100  # raw quotes keep the jump
    splits = actions[actions['kind'] == 'split']
    assert set(splits['ticker']) == {'NVDA', 'AVGO', 'ORLY', 'FAST'}


def test_dataset_files_are_deterministic(tmp_path):
    first = md.write_dataset(tmp_path/'a')
    second = md.write_dataset(tmp_path/'b')
    assert first == second


def test_json_extraction_is_the_same_rule_for_any_response():
    plain = '{"tool": "submit", "arguments": {"weights": {"SPY": 1}}}'
    fenced = 'Sure.\n```json\n{"tool": "read_file", "arguments": {}}\n```'
    assert ts.extract_json(plain)[1] == 'json' and ts.extract_json(fenced)[1] == 'json'
    assert ts.extract_json('I will read the file first.')[0] is None
    weights, problem = ts.validate_weights({'SPY': 0.6, 'IEF': 0.5})
    assert weights is None and 'sum' in problem
    weights, problem = ts.validate_weights({'SPY': 0.6, 'IEF': 0.4})
    assert problem is None and weights['GLD'] == 0.0 and abs(sum(weights.values()) - 1) < 1e-9


def test_ledger_matches_a_hand_computation():
    dates = pd.date_range('2025-01-01', periods=6).strftime('%Y-%m-%d')
    total = pd.DataFrame(1.0, index=dates, columns=list(md.TICKERS))
    total['SPY'] = [1, 1, 1.1, 1.21, 1.21, 1.331]
    picks = [0, 3]
    target = {t: 0.0 for t in md.TICKERS}
    target['SPY'] = 1.0
    nav, trades = ts.ledger(total, picks, [target, None])
    # Trade at session 1 (close 1.0), pay 5 bps on turnover 1, then compound SPY.
    assert abs(nav.iloc[0] - (1 - 5e-4)) < 1e-12
    assert abs(nav.iloc[-1] - (1 - 5e-4) * 1.331) < 1e-9
    assert trades[1].get('unchanged') is True and trades[1]['cost'] == 0.0


class Scripted:
    """Raw arm submits equal weight after reading a file; library arm runs an algorithm."""

    def __init__(self, *_):
        self.seed = self.calls = 0

    def __call__(self, history):
        library = 'fin-skills financial library' in history[0]['content']
        turn = (len(history) - 2) // 2
        if turn == 0:
            call = ({'tool': 'run_algorithm', 'arguments': {'algorithm_id': 'inverse_volatility',
                                                            'tickers': ['SPY', 'TLT', 'GLD'],
                                                            'lookback': 60}}
                    if library else {'tool': 'read_file', 'arguments': {'path': 'README.md'}})
        elif library:
            weights = json.loads(history[-1]['content'])['result']['weights']
            call = {'tool': 'submit', 'arguments': {'weights': weights}}
        else:
            call = {'tool': 'submit', 'arguments': {'weights': {t: 1/12 for t in md.TICKERS}}}
        self.calls += 1
        return {'choices': [{'message': {'content': json.dumps(call)}, 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15},
                'generation_seconds': 0.0}


@pytest.mark.slow
@linux_only
def test_scripted_paths_run_through_controller_ledger_and_scorer(tmp_path, package_path,
                                                                 monkeypatch):
    monkeypatch.setattr(ts, 'SEEDS', (11,))
    root = tmp_path/'7b'
    ts.freeze(root, ['7b'])
    ts.qualify(root)
    ts.run(root, ['7b'], backend_factory=Scripted)
    ts.score(root)
    scores = json.loads((root/'scores.json').read_text())
    raw, lib = scores['paths']['7b-11-raw'], scores['paths']['7b-11-library']
    assert raw['submitted'] == raw['decisions'] == lib['submitted']
    assert lib['library_calls'] == lib['decisions'] and raw['library_calls'] == 0
    # Equal weight rebalanced every decision must reproduce the baseline exactly.
    base = scores['baselines']['equal_weight_rebalanced']
    assert abs(raw['metrics']['cumulative_return'] - base['cumulative_return']) < 1e-9
    assert '7b-11' in scores['paired'] and scores['paired']['7b-11']['block_bootstrap']
    # The manuscript generator re-checks hashes and re-prices every path from the decisions.
    from scripts.build_trading_evidence import macros
    numbers = macros(tmp_path, verify_fetch=False)
    assert numbers['TRSevenRawReturn'] == f"{100*base['cumulative_return']:.1f}"
    assert numbers['TRSevenLibLibCalls'] == '1.00' and numbers['TRFamiliesDone'] == 1


def test_code_can_arrive_in_a_fence_or_python_triple_quotes():
    fenced = '{"tool": "run_python"}\n```python\nprint(1)\n```'
    assert ts.extract_call(fenced) == ({'tool': 'run_python',
                                        'arguments': {'code': 'print(1)'}}, 'json+fence')
    triple = ('```json\n{\n  "tool": "run_python",\n  "arguments": {\n    "code": """\n'
              'print(2)\n"""\n  }\n}\n```')
    call, status = ts.extract_call(triple)
    assert status == 'lenient' and call['arguments']['code'].strip() == 'print(2)'
    # Only run_python code is recovered; malformed submissions are not repaired.
    assert ts.extract_call('{"tool": "submit", "arguments": {"weights": {SPY: 1}}}')[0] is None
    assert ts.extract_call('I would buy SPY.') == (None, 'unparsed')


def test_every_aborted_v2_turn_parses_the_same_way_or_better():
    folder = ROOT/'benchmarks/agent_study/evidence/20260929-trading-study/aborted-v2-20260930'
    for path in folder.glob('*/decisions/*/*.json'):
        for turn in json.loads(path.read_text())['turns']:
            call, status = ts.extract_call(turn['response'])
            if turn['parse'] == 'json':
                assert status in ('json', 'json+fence')
            else:
                assert status == 'lenient' and call['tool'] == 'run_python'


def test_a_malformed_submission_is_rejected_without_stopping_the_run():
    class Backend:
        seed = calls = 0
        replies = ['{"tool": "submit", "arguments": "SPY 0.5, TLT 0.5"}',
                   '{"tool": "submit", "arguments": {"weights": {"SPY": 0.5, "TLT": 0.5}}}']

        def __call__(self, history):
            text = self.replies[min(self.calls, 1)]
            self.calls += 1
            return {'choices': [{'message': {'content': text}, 'finish_reason': 'stop'}],
                    'usage': {'prompt_tokens': 1, 'completion_tokens': 1},
                    'generation_seconds': 0.0}

    record = ts.decide(Backend(), ts.Controller(ROOT, 'raw', md.DATA_DIR), 'raw', '2025-01-02',
                       0, {t: 0.0 for t in md.TICKERS})
    assert record['turns'][0]['accepted'] is False
    assert 'non-empty object' in record['turns'][0]['result']['error']
    assert record['submitted'] and record['target']['SPY'] == 0.5
