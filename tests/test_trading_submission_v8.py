"""Execution-contract regressions: explicit exposure, arithmetic and model consent."""
import json

import pytest

from benchmarks.agent_study import trading_runtime_v8 as v8
from benchmarks.agent_study.trading_submission_v8 import extract_decision, submission
from test_trading_terminal_v8 import Controller, response
from test_trading_terminal_v8 import workspace  # noqa: F401
from benchmarks.agent_study.trading_worker_v6 import run_python


def text_response(text):
    return {'choices': [{'message': {'content': text}, 'finish_reason': 'stop'}]}


@pytest.mark.parametrize('arm', ['raw', 'library'])
def test_model_parts_determine_cash_exposure_and_asset_ratios(arm):
    parts = {'SPY': 40, 'QQQ': 30, 'CASH': 40}
    record = v8.decide(lambda _: text_response(json.dumps({'allocation': parts})), Controller(arm), 'fixture')
    v8.validate_decision(record)
    assert record['target']['SPY'] == pytest.approx(40 / 110)
    assert record['target']['QQQ'] == pytest.approx(30 / 110)
    assert record['target']['IEF'] == 0
    assert 1 - sum(record['target'].values()) == pytest.approx(40 / 110)
    assert record['turns'][0]['result']['allocation_input'] == parts
    record['turns'][0]['result']['cash_weight'] = 0
    with pytest.raises(ValueError, match='receipt changed'):
        v8.validate_decision(record)


def test_original_overbudget_weights_are_not_reinterpreted_as_relative_parts():
    # Exact arithmetic failure seen in four of the seven inspected final responses.
    weights = {'SPY': .4, 'QQQ': .3, 'EFA': .2, 'EEM': .1, 'TLT': .05, 'IEF': .05}
    record = v8.decide(lambda _: text_response(json.dumps({'weights': weights})), Controller('raw'), 'fixture')
    assert record['decision_action'] == 'failed' and record['target'] is None
    assert all('1.100000' in t['result']['error'] for t in record['turns'])


@pytest.mark.parametrize('parts', [
    {'SPY': 1}, {'SPY': 1, 'CASH': -1}, {'SPY': float('nan'), 'CASH': 0},
    {'SPY': float('inf'), 'CASH': 0}, {'SPY': True, 'CASH': 0},
    {'SPY': '40', 'CASH': 60}, {'SPY': 0, 'CASH': 0},
    {'INVENTED': 1, 'CASH': 0}, {'SPY': 10**1000, 'CASH': 0}, [],
])
def test_incomplete_or_invalid_orders_never_get_default_investments(parts):
    result = submission({'allocation': parts})
    assert not result['ok'] and result['submission'] is None


def test_large_finite_parts_do_not_overflow_and_all_cash_is_an_explicit_liquidation():
    result = submission({'allocation': {'SPY': 1e308, 'CASH': 1e308}})
    assert result['submission']['SPY'] == .5 and result['cash_weight'] == .5
    cash = submission({'allocation': {'CASH': 1}})
    assert cash['ok'] and sum(cash['submission'].values()) == 0 and cash['cash_weight'] == 1


@pytest.mark.parametrize('text', ['HOLD', 'hold', '{"action":"hold"}', '{"action":"HOLD"}',
                                '```json\n{"action":"hold"}\n```'])
def test_simple_hold_is_model_authored_and_preserves_units(text):
    record = v8.decide(lambda _: text_response(text), Controller('raw'), 'fixture')
    v8.validate_decision(record)
    assert record['decision_action'] == 'hold' and record['target'] is None


@pytest.mark.parametrize('text', [
    'Maybe hold, or buy SPY.',
    '{"allocation":{"SPY":1,"CASH":0}}\n{"action":"hold"}',
    '{"tool":"hold","arguments":{}}\n{"weights":{"SPY":1}}',
    '{"allocation":{"SPY":1,"SPY":2,"CASH":0}}',
    '{"tool":"submit","arguments":{"allocation":{"SPY":1,"SPY":2,"CASH":0}}}',
    '<think>{"allocation":{"SPY":1,"CASH":0}}</think>',
    '<think>HOLD',
])
def test_ambiguity_duplicate_keys_and_reasoning_are_not_orders(text):
    call, _ = extract_decision(text)
    assert call is None


@pytest.mark.parametrize('arm', ['raw', 'library'])
def test_overbudget_draft_gets_one_model_authored_correction_within_eight_calls(arm):
    calls = []
    def backend(messages):
        calls.append([dict(m) for m in messages])
        if len(calls) <= 6:
            return response('read_market')
        if len(calls) == 7:
            assert 'FINAL DECISION' in messages[0]['content']
            return text_response('{"weights":{"SPY":0.7,"QQQ":0.4}}')
        assert '1.100000' in messages[-1]['content']
        assert '0 correction responses' in messages[-1]['content']
        return text_response('{"allocation":{"SPY":7,"QQQ":4,"CASH":2}}')
    controller = Controller(arm)
    record = v8.decide(backend, controller, 'fixture')
    v8.validate_decision(record)
    assert len(calls) == 8 and len(controller.calls) == 6
    assert record['decision_action'] == 'submit'
    assert record['target']['SPY'] == pytest.approx(7 / 13)
    assert 1 - sum(record['target'].values()) == pytest.approx(2 / 13)


@pytest.mark.parametrize('text', [
    '{"tool":"hold","arguments":{},"allocation":{"SPY":1,"CASH":0}}',
    '{"tool":"submit","arguments":{"weights":{"SPY":1}},"leverage":2}',
    '{"allocation":{"SPY":1,"CASH":0}}\n```python\nsubmit({"IEF":1})\n```',
    '{"tool":"hold","arguments":{}}\n{"allocation":',
    '{"plan":{"tool":"submit","arguments":{"weights":{"SPY":1}}}}',
    '{"tool":"hold","arguments":{}}<think>Reconsider.</think>{"weights":{"SPY":1}}',
    '```json\n{"action":"hold"}\n```\n```python\nsubmit({"SPY":1})\n```',
])
def test_entire_order_is_validated_without_discarding_competing_instructions(text):
    controller = Controller('library')
    record = v8.decide(lambda _: text_response(text), controller, 'fixture')
    assert record['decision_action'] == 'failed' and record['target'] is None
    assert not controller.calls


@pytest.mark.parametrize('label', ['', 'json', 'python', 'py'])
def test_fence_label_does_not_change_complete_json_order_semantics(label):
    text = '```' + label + '\n{"allocation":{"SPY":3,"CASH":2}}\n```'
    call, status = extract_decision(text)
    assert status == 'json' and call['tool'] == 'submit'
    assert submission(call['arguments'])['submission']['SPY'] == pytest.approx(.6)


def test_python_json_example_is_data_and_reasoning_is_never_an_order():
    code = 'example = {"tool":"submit","arguments":{"weights":{"SPY":1}}}\nprint(example)'
    call, status = extract_decision('```python\n' + code + '\n```')
    assert status == 'python-fence' and call['arguments']['code'] == code
    call, status = extract_decision('<think>{"weights":{"SPY":1}}</think>{"action":"hold"}')
    assert status == 'json' and call['tool'] == 'hold'


@pytest.mark.parametrize('value', [True, '0.5', 10**1000, float('inf')])
def test_invalid_absolute_numbers_are_errors_not_casts_or_worker_crashes(value):
    reply = submission({'weights': {'SPY': value}})
    assert not reply['ok'] and reply['submission'] is None


@pytest.mark.parametrize('arm', ['raw', 'library'])
def test_python_and_json_allocation_have_identical_validation_and_receipts(arm, workspace):
    tools = v8.Tools(workspace, arm)
    for parts in ({'SPY': 7, 'CASH': 3}, {'SPY': 1}, {'CASH': 0}, {'CASH': 1}):
        expected = submission({'allocation': parts})
        actual = run_python('submit(allocation=' + repr(parts) + ')', tools)
        assert actual['ok'] == expected['ok']
        assert actual['submission'] == expected['submission']
        if expected['ok']:
            for key in ('allocation_input', 'cash_weight', 'conversion'):
                assert actual[key] == expected[key]


def test_contracts_are_visible_before_first_call_and_do_not_suggest_an_allocation(workspace):
    for arm in ('raw', 'library'):
        tools = v8.Tools(workspace, arm)
        prompt = v8.system_prompt(arm)
        for name in ('read_market', 'read_snapshot', 'read_evidence', 'read_file'):
            assert tools.help_tool(name)['signature'] in prompt
        assert not any(ticker in prompt for ticker in v8.TICKERS)
        assert '"allocation"' in prompt and '"hold"' in prompt
        assert 'example' not in tools.help_tool('submit')


@pytest.mark.parametrize('text', ['unanswered', '{"tool":"read_market","arguments":{}}'])
def test_audit_rejects_success_receipt_without_a_matching_model_order(text):
    record = v8.decide(lambda _: text_response('{"weights":{"SPY":0.6}}'),
                       Controller('raw'), 'fixture')
    record['turns'][-1]['response'] = text
    with pytest.raises(ValueError, match='model response|model submit'):
        v8.validate_decision(record)


def test_rounding_tolerance_is_explicit_in_receipt_and_cannot_be_tampered():
    weights = {'SPY': .5001, 'IEF': .5001}
    record = v8.decide(lambda _: response('submit', {'weights': weights}), Controller('raw'), 'fixture')
    assert record['turns'][0]['result']['weights_input'] == weights
    assert record['turns'][0]['result']['budget_rounding_applied']
    assert record['target']['SPY'] == .5
    v8.validate_decision(record)
    record['turns'][0]['result']['weights_input'] = {'SPY': .5, 'IEF': .5}
    with pytest.raises(ValueError, match='receipt changed'):
        v8.validate_decision(record)


def test_research_parse_feedback_does_not_push_model_to_stop_research():
    record = v8.decide(lambda _: text_response('incomplete'), Controller('raw'), 'fixture')
    for turn in record['turns'][:5]:
        assert 'HOLD' not in turn['model_feedback'] and 'allocation' not in turn['model_feedback']
        assert 'tool JSON' in turn['model_feedback']


def test_non_text_backend_is_an_infrastructure_error_not_a_trading_outcome():
    assert extract_decision(None) == (None, 'non-text-response')
    with pytest.raises(RuntimeError, match='qualified adapter'):
        v8.decide(lambda _: text_response(None), Controller('raw'), 'fixture')


@pytest.mark.parametrize('label', ['JSON', 'Python', 'py', ''])
def test_crlf_fences_are_equivalent_json_transports(label):
    call, status = extract_decision('```' + label + '\r\n{"allocation":{"SPY":1,"CASH":0}}\r\n```')
    assert status == 'json' and call['tool'] == 'submit'


def test_hold_rationale_does_not_ignore_extra_trade_parameters():
    assert extract_decision('{"action":"hold","rationale":"keep units"}')[0]['tool'] == 'hold'
    assert extract_decision('{"action":"hold","rationale":"keep units","weights":{"SPY":1}}')[0]['tool'] == 'submit'
    record = v8.decide(lambda _: text_response('{"action":"hold","weights":{"SPY":1}}'), Controller('raw'), 'fixture')
    assert record['decision_action'] == 'failed'


def test_excessive_json_nesting_is_a_failed_turn_not_a_runner_crash():
    call, status = extract_decision('[' * 2000 + '0' + ']' * 2000)
    assert call is None and status in ('unparsed', 'json-too-deep')


def test_library_signatures_are_available_without_discovery_calls():
    from benchmarks.agent_study import trading_runtime_v9 as v9
    prompt = v9.system_prompt('library')
    for name in ('research_context', 'run_algorithm', 'read_skill', 'describe_algorithm'):
        assert name + '(' in prompt
    assert 'run_algorithm(' not in v9.system_prompt('raw')
