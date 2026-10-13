"""Regression for actual general7B py-prefixed research requests."""
import json
import pytest
from benchmarks.agent_study.trading_submission_v8 import extract_decision
from benchmarks.agent_study import trading_runtime_v8 as v8
from benchmarks.agent_study.trading_worker_v6 import run_python
from test_trading_terminal_v8 import Controller, response, workspace


@pytest.mark.parametrize('label', ['py', 'python', 'PYTHON'])
def test_explicit_label_preserves_complete_program(label):
    code = 'quotes = read_snapshot()\nprint(quotes)'
    call, status = extract_decision(label + '\n' + code)
    assert status == 'python-label'
    assert call == {'tool': 'run_python', 'arguments': {'code': code}}


@pytest.mark.parametrize('body', [
    'read_snapshot()\n{"tool":"hold","arguments":{}}',
    '{"tool":"hold","arguments":{}}\nread_snapshot()',
    '{"tool":"hold","arguments":{}}\n{"tool":"read_snapshot","arguments":{}}',
    'print(read_snapshot(',
    'Here is my investment analysis:\nprint(1)',
    '```python\nprint(1)\n```',
    '# No executable request',
    '"explanation"\nprint(1)',
])
def test_label_does_not_repair_or_mine_mixed_and_incomplete_responses(body):
    assert extract_decision('py\n' + body)[0] is None


@pytest.mark.parametrize('text', ['print(123)', 'json\nprint(123)', 'javascript\nprint(123)'])
def test_executable_python_still_requires_explicit_python_transport(text):
    assert extract_decision(text)[0] is None


def test_json_literals_inside_assigned_python_data_are_not_orders():
    code = 'example = {"tool":"hold","arguments":{}}\nprint(example)'
    assert extract_decision('py\n' + code) == (
        {'tool': 'run_python', 'arguments': {'code': code}}, 'python-label')


def test_prefixed_reader_uses_existing_worker_and_returns_real_snapshot(workspace):
    call, _ = extract_decision('py\nprint(read_snapshot())')
    result = run_python(call['arguments']['code'], v8.Tools(workspace, 'raw'))
    assert result['ok'] and 'SPY' in result['output']


@pytest.mark.parametrize('arm', ['raw', 'library'])
def test_label_executes_in_research_only_with_unchanged_model_decision(arm):
    c = Controller(arm)
    replies = iter(['py\nprint(read_snapshot())', '{"tool":"hold","arguments":{}}'])
    def backend(_):
        return {'choices': [{'message': {'content': next(replies)}, 'finish_reason': 'stop'}]}
    record = v8.decide(backend, c, 'fixture')
    v8.validate_decision(record)
    assert c.calls == [{'tool': 'run_python', 'arguments': {'code': 'print(read_snapshot())'}}]
    assert record['decision_action'] == 'hold' and record['target'] is None


def test_truncated_label_is_not_executed_even_when_prefix_is_valid_python():
    c = Controller('raw')
    def backend(_):
        return {'choices': [{'message': {'content': 'py\nprint(123)'}, 'finish_reason': 'length'}]}
    record = v8.decide(backend, c, 'fixture')
    assert not c.calls and record['decision_action'] == 'failed'


def test_final_stage_rejects_label_without_inserting_a_hold():
    c, count = Controller('library'), 0
    def backend(_):
        nonlocal count
        count += 1
        if count <= v8.RESEARCH_RESPONSES:
            return response('read_snapshot')
        return {'choices': [{'message': {'content': 'py\nprint(read_snapshot())'}, 'finish_reason': 'stop'}]}
    record = v8.decide(backend, c, 'fixture')
    assert len(c.calls) == v8.RESEARCH_RESPONSES
    assert record['decision_action'] == 'failed' and record['target'] is None
