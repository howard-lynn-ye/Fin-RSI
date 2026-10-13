"""Actual retained probe wrappers and negative-CASH correction feedback."""
import json

import pytest

from benchmarks.agent_study.trading_submission_v8 import extract_decision, submission
from benchmarks.agent_study import trading_runtime_v8 as v8
from test_trading_terminal_v8 import Controller, response


@pytest.mark.parametrize('label', ['py', 'python', 'json', 'JSON'])
def test_language_label_only_unwraps_one_complete_json_call(label):
    call = {'tool': 'research_context', 'arguments': {'query': 'risk'}}
    assert extract_decision(label + '\n' + json.dumps(call)) == (call, 'json')


@pytest.mark.parametrize('body', [
    'read_snapshot(tickers=None)\n{"tool":"hold","arguments":{}}',
    '{"tool":"hold","arguments":{}}\n{"tool":"read_snapshot","arguments":{}}',
    '{"tool":"hold","arguments":{}} explanation',
    '```python\nprint(123)\n```',
    'print(123)',
    '{"tool":"hold","tool":"submit","arguments":{}}',
])
def test_label_never_selects_between_actions_or_executes_unfenced_code(body):
    assert extract_decision('py\n' + body)[0] is None


def test_negative_cash_feedback_identifies_field_without_inventing_replacement():
    chosen = {'SPY': 0.2, 'CASH': -0.05}
    result = submission({'allocation': chosen})
    assert not result['ok'] and result['submission'] is None
    assert result['invalid_parts'] == ['CASH']
    assert 'need not sum to 1' in result['hint']
    assert chosen == {'SPY': 0.2, 'CASH': -0.05}


def test_backend_receives_actionable_error_and_makes_its_own_correction():
    calls = []
    def backend(messages):
        calls.append(messages[-1]['content'])
        return (response('submit', {'allocation': {'CASH': -0.05, 'SPY': 0.2}})
                if len(calls) == 1 else response('submit', {'allocation': {'CASH': 2, 'GLD': 7}}))
    result = v8.decide(backend, Controller('raw'), 'fixture')
    assert 'CASH' in calls[1] and 'need not sum to 1' in calls[1]
    assert result['target']['GLD'] == pytest.approx(7 / 9)
    assert result['turns'][1]['result']['allocation_input'] == {'CASH': 2, 'GLD': 7}


def test_invalid_part_diagnostic_does_not_crash_on_non_string_python_keys():
    result = submission({'allocation': {123: -1, 'CASH': 1}})
    assert not result['ok'] and result['submission'] is None


@pytest.mark.parametrize('amount', [1, -1])
def test_unknown_instrument_feedback_identifies_the_name_and_allowed_universe(amount):
    result = submission({'allocation': {'invented_ticker': amount, 'CASH': 1}})
    assert not result['ok'] and result['submission'] is None
    assert 'unknown instrument: invented_ticker' in result['error']
    assert 'Allowed instruments:' in result['error'] and 'CASH' in result['error']


def test_all_invalid_actions_remain_failure_not_a_corrected_portfolio():
    result = v8.decide(lambda _: response('submit', {'allocation': {'SPY': 1, 'CASH': -1}}),
                       Controller('library'), 'fixture')
    assert result['decision_action'] == 'failed' and result['target'] is None
