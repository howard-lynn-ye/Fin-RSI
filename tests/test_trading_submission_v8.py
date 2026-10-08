"""Execution-contract regressions: explicit exposure, arithmetic and model consent."""
import json

import pytest

from benchmarks.agent_study import trading_runtime_v8 as v8
from benchmarks.agent_study.trading_submission_v8 import extract_decision, submission
from test_trading_terminal_v8 import Controller, response


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


@pytest.mark.parametrize('text', ['HOLD', 'hold', '{"action":"hold"}',
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
