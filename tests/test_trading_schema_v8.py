"""Output transport tests, including the two real CASH-nesting regressions."""
import copy
import json

import pytest

from benchmarks.agent_study import trading_runtime_v8 as v8, trading_schema_v8 as schema
from test_trading_terminal_v8 import Controller, response


class StructuredBackend:
    def __init__(self, finals):
        self.finals, self.research, self.final_calls = iter(finals), 0, 0

    def prepare_final_decision(self):
        return schema.output_contract(True)

    def __call__(self, messages):
        self.research += 1
        return response('read_market')

    def final_decision(self, messages):
        self.final_calls += 1
        assert 'FINAL DECISION' in messages[0]['content']
        reply = next(self.finals)
        reply['backend_metadata'] = {'final_output': schema.output_contract(True)}
        return reply


@pytest.mark.parametrize('arm', ['raw', 'library'])
@pytest.mark.parametrize('answer', [response('hold'), response('submit', {'allocation': {'CASH': 5}}),
    response('submit', {'allocation': {'SPY': 7, 'CASH': 0}}),
    response('submit', {'allocation': {'SPY': 3, 'GLD': 8, 'CASH': 2}, 'rationale': 'chosen'})])
def test_final_transport_preserves_model_action_parts_and_equal_budget(arm, answer):
    backend, controller = StructuredBackend([copy.deepcopy(answer)]), Controller(arm)
    record = v8.decide(backend, controller, 'fixture', final_output=schema.output_contract(True))
    v8.validate_decision(record, expected_final_output=schema.output_contract(True))
    assert backend.research == 6 and backend.final_calls == 1 and len(controller.calls) == 6
    assert record['turns'][-1]['response'] == answer['choices'][0]['message']['content']
    original = json.loads(record['turns'][-1]['response'])
    if original['tool'] == 'submit':
        parts = original['arguments']['allocation']
        assert record['turns'][-1]['result']['allocation_input'] == parts
        assert record['target']['SPY'] == pytest.approx(parts.get('SPY', 0) / sum(parts.values()))
    else:
        assert record['decision_action'] == 'hold' and record['target'] is None


@pytest.mark.parametrize('first', [response('submit', {'allocation': {'SPY': -2, 'CASH': 1}}),
    response('submit', {'allocation': {'CASH': 0}}),
    response('submit', {'allocation': {'SPY': 1, 'CASH': 0}}, finish='length')])
def test_grammar_does_not_bypass_semantic_or_truncation_checks(first):
    backend = StructuredBackend([copy.deepcopy(first), response('hold')])
    record = v8.decide(backend, Controller('raw'), 'fixture', final_output=schema.output_contract(True))
    v8.validate_decision(record)
    assert backend.research == 6 and backend.final_calls == 2
    assert not record['turns'][-2]['result']['ok']
    assert record['decision_action'] == 'hold' and record['target'] is None


def test_exhausted_semantic_failures_are_not_default_hold_or_portfolio():
    backend = StructuredBackend([response('submit', {'allocation': {'CASH': 0}}) for _ in range(2)])
    record = v8.decide(backend, Controller('library'), 'fixture', final_output=schema.output_contract(True))
    v8.validate_decision(record)
    assert record['decision_action'] == 'failed' and not record['decision_completed']
    assert record['target'] is None and backend.final_calls == 2


def test_missing_decoder_fails_before_any_model_or_research_call():
    def backend(messages):
        pytest.fail('unqualified backend should never be called')
    with pytest.raises(RuntimeError, match='structured final backend required'):
        v8.decide(backend, Controller('raw'), 'fixture', final_output=schema.output_contract(True))


@pytest.mark.parametrize('bad', [
    {'tool': 'submit', 'arguments': {'allocation': {'SPY': .3}, 'CASH': 0}},
    {'tool': 'submit', 'arguments': {'allocation': {'SPY': .35}, 'CASH': 0}},
    {'tool': 'submit', 'arguments': {'allocation': {'SPY': 1}}},
    {'tool': 'hold', 'arguments': {'allocation': {'CASH': 1}}},
    {'tool': 'research_context', 'arguments': {'query': 'news'}},
])
def test_broken_decoder_is_infrastructure_failure_never_a_scored_hold(bad):
    reply = response('hold')
    reply['choices'][0]['message']['content'] = json.dumps(bad)
    with pytest.raises(RuntimeError, match='broken structured decoder'):
        v8.decide(StructuredBackend([reply]), Controller('raw'), 'fixture',
                  final_output=schema.output_contract(True))


@pytest.mark.parametrize('tamper', ['version', 'schema', 'missing', 'downgrade', 'wrong_stage'])
def test_auditor_rejects_different_transport_or_missing_decoder_receipt(tamper):
    record = v8.decide(StructuredBackend([response('hold')]), Controller('raw'), 'fixture',
                       final_output=schema.output_contract(True))
    metadata = record['turns'][-1]['backend_metadata']
    if tamper == 'version':
        metadata['final_output']['version'] = 'other'
    elif tamper == 'schema':
        metadata['final_output']['schema_sha256'] = '0' * 64
    elif tamper == 'missing':
        record['turns'][-1]['backend_metadata'] = None
    elif tamper == 'downgrade':
        record['final_output'] = schema.output_contract()
    else:
        record['turns'][0]['backend_metadata'] = metadata
    with pytest.raises(ValueError):
        v8.validate_decision(record, expected_final_output=schema.output_contract(True))


def grammar_accepts(text):
    lmfe = pytest.importorskip('lmformatenforcer', reason='pinned decoder tested in Slurm qualification')
    parser = schema.parser_factory()
    for character in text:
        if character not in parser.get_allowed_characters():
            return False
        parser = parser.add_character(character)
    return parser.can_end()


@pytest.mark.parametrize('text', [
    '{"tool":"hold","arguments":{}}',
    '{"arguments":{},"tool":"hold"}',
    '{"tool":"submit","arguments":{"allocation":{"CASH":1}}}',
    '{"arguments":{"allocation":{"SPY":2,"CASH":0},"rationale":"own choice"},"tool":"submit"}',
    '{"tool":"submit","arguments":{"allocation":{"CASH":0,"QQQ":2,"GLD":1}}}',
])
def test_pinned_grammar_allows_either_action_order_and_model_chosen_assets(text):
    assert grammar_accepts(text)
    schema.check_format(text)


@pytest.mark.parametrize('text', [
    '{"tool":"submit","arguments":{"allocation":{"SPY":.3},"CASH":0}}',
    '{"tool":"submit","arguments":{"allocation":{"SPY":0.35},"CASH":0}}',
    '{"tool":"submit","arguments":{"allocation":{"SPY":1}}}',
    '{"tool":"hold","arguments":{}} extra explanation',
    '{"tool":"hold","arguments":{}} {"tool":"hold","arguments":{}}',
    '{"tool":"hold","arguments":{"allocation":{"CASH":1}}}',
    '{"tool":"submit","arguments":{"allocation":{"UNKNOWN":1,"CASH":0}}}',
    '{"tool":"submit","arguments":{"allocation":{"CASH":0,"CASH":1}}}',
    '{"tool":"submit","arguments":{"allocation":{"CASH":0,"SPY":true}}}',
    '{"tool":"rebalance","arguments":{"allocation":{"SPY":1,"CASH":0}}}',
    '{"tool":"read_market","arguments":{}}',
])
def test_pinned_grammar_excludes_observed_format_regressions(text):
    assert not grammar_accepts(text)
    with pytest.raises(ValueError):
        schema.check_format(text)
