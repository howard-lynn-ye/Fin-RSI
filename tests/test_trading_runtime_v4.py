import json
from pathlib import Path

import pandas as pd
import pytest

from benchmarks.agent_study import trading_runtime_v4 as v4
from benchmarks.agent_study import trading_study as v3
from benchmarks.agent_study import market_data as md
from benchmarks.agent_study.linux_sandbox import confinement_available
from benchmarks.agent_study.trading_tools_v4 import Tools, validate_weights
from benchmarks.agent_study.trading_worker_v4 import run_python, readonly_paths


def test_first_fee_counts_in_return_drawdown_and_volatility():
    prices = pd.DataFrame(100., index=pd.date_range('2024-01-01', periods=5),
                          columns=md.TICKERS)
    weights = dict.fromkeys(md.TICKERS, 0.)
    weights['SPY'] = 1.
    nav, _ = v4.ledger(prices, [0], [weights])
    report = v4.metrics(nav)
    assert nav.iloc[0] == 1.
    assert report['cumulative_return'] == pytest.approx(-0.0005)
    assert report['max_drawdown'] == pytest.approx(-0.0005)
    assert report['annualized_volatility'] > 0
    # The archived scorer deliberately retains its old interpretation.
    old, _ = v3.ledger(prices, [0], [weights])
    assert v3.metrics(old)['cumulative_return'] == 0.


def test_rounding_normalizes_without_leverage_and_preserves_cash():
    clean, error = validate_weights({'SPY': .5001, 'IEF': .5001})
    assert error is None and sum(clean.values()) == pytest.approx(1)
    assert clean['SPY'] == .5
    assert validate_weights({'SPY': .8})[0]['SPY'] == .8
    for bad in ({'SPY': 1.01}, {'SPY': -1e-10}, {'SPY': float('nan')}, {}):
        assert validate_weights(bad)[0] is None


def test_bare_fence_is_code_not_an_embedded_json_tool_call():
    code = 'print({"tool":"submit","arguments":{"SPY":1}})'
    call, status = v4.extract_call('```python\n' + code + '\n```')
    assert call['tool'] == 'run_python' and call['arguments']['code'] == code
    assert status == 'python-fence'
    assert v4.extract_call('{"tool":"submit","arguments":{"SPY":1}}')[0]['tool'] == 'submit'


def test_python_submit_ends_execution_and_invalid_submit_is_an_error(tmp_path):
    tools = Tools(tmp_path, 'raw')
    result = run_python('submit({"SPY": 0.5001, "IEF": 0.5001})\nraise Exception()', tools)
    assert result['ok'] and result['submission']['SPY'] == .5
    result = run_python('submit({"SPY": 1.2})', tools)
    assert not result['ok'] and result['submission'] is None
    assert 'run_algorithm' not in tools.python_bindings()


def test_python_and_json_adapter_results_match_and_menu_is_seeded(tmp_path):
    (tmp_path / 'README.md').write_text('known input')
    tools = Tools(tmp_path, 'library', menu_seed=7)
    direct = tools.call('read_file', {'path': 'README.md'})
    python = tools.python_bindings()['read_file']('README.md')
    assert direct == python
    a = tools.list_algorithms()
    assert a == Tools(tmp_path, 'library', menu_seed=7).list_algorithms()
    assert a != Tools(tmp_path, 'library', menu_seed=8).list_algorithms()
    assert Tools(tmp_path, 'raw').call('run_algorithm', {})['ok'] is False


def test_read_grants_exclude_study_data_even_with_bad_shared_runtime(tmp_path, monkeypatch):
    source = Path(v4.__file__).resolve().parent
    package = source.parents[1] / 'fin_skills'
    monkeypatch.delenv('FIN_STUDY_SHARED_RUNTIME', raising=False)
    grants = readonly_paths(tmp_path, 'library', package)
    assert not any(source.is_relative_to(p.resolve()) for p in grants)
    monkeypatch.setenv('FIN_STUDY_SHARED_RUNTIME', str(source.parent))
    with pytest.raises(RuntimeError, match='contains study data'):
        readonly_paths(tmp_path, 'library', package)


def test_controller_consumes_python_submission(tmp_path):
    class Backend:
        def __call__(self, history):
            return {'choices': [{'message': {'content': '```python\nsubmit({"SPY":1})\n```'},
                                  'finish_reason': 'stop'}]}
    class Controller:
        arm, menu_seed, calls = 'raw', 0, []
        def call(self, tool, arguments):
            return run_python(arguments['code'], Tools(tmp_path, 'raw'))
    record = v4.decide(Backend(), Controller(), 'submit a target')
    assert record['submitted'] and len(record['turns']) == 1
    assert record['target']['SPY'] == 1.


def test_no_argument_json_call_reaches_the_same_catalog_as_python(tmp_path):
    controller = v4.Controller(tmp_path, tmp_path, 'library', menu_seed=3)
    call, _ = v4.extract_call('{"tool":"list_algorithms"}')
    result = controller.call(call['tool'], call.get('arguments'))
    assert result['ok']
    assert result == controller.tools.python_bindings()['list_algorithms']()
    assert controller.tools.call('list_algorithms', None) == result
    assert controller.call('list_algorithms', 'bad')['ok'] is False


@pytest.mark.skipif(not confinement_available(), reason='requires real Linux confinement')
def test_real_worker_denies_future_data_and_accepts_both_interfaces(tmp_path):
    md.write_dataset(tmp_path / 'data')
    result = v4.qualify(tmp_path / 'qualification', tmp_path / 'data')
    assert result['passed'], json.dumps(result['results'], indent=2)
