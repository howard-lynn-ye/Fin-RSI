"""Regressions from the retained three-model preflight, without portfolio defaults."""
import pytest

from benchmarks.agent_study import trading_runtime_v8 as v8, trading_runtime_v9 as v9
from benchmarks.agent_study.trading_submission_v8 import submission
from test_trading_terminal_v8 import workspace, response, Controller  # noqa: F401


def test_task_separates_observation_cutoff_from_valuation_deadline():
    text = v8.task('2025-01-02', {}, '2026-09-25')
    assert text.startswith('OBSERVATION CUTOFF: 2025-01-02.')
    assert 'INVESTMENT DEADLINE: 2026-09-25.' in text
    for arm in ('raw', 'library'):
        text = v8.system_prompt(arm)
        assert 'omit read_market start/end' in text
        assert 'CURRENT request or decision' in text
        assert 'Response 1/8' not in text


@pytest.mark.parametrize('arm', ['raw', 'library'])
def test_future_queries_still_rejected_with_actionable_contract(arm, workspace):
    tools = v8.Tools(workspace, arm)
    bad = tools.call('read_market', {'end': '2026-09-25'})
    assert not bad['ok'] and 'exceeds visible as_of=' in bad['error']
    assert 'Omit read_market start/end' in bad['contract']['note']
    good = tools.call('read_market', {})
    assert good['ok'] and good['rows']
    assert all(r['date'] <= good['as_of'] for r in good['rows'])


def test_conflicting_hold_and_allocation_are_rejected_without_choosing():
    result = submission({'allocation': {'SPY': 80, 'CASH': 20}, 'action': 'hold'})
    assert not result['ok'] and result['submission'] is None
    assert 'Conflicting decision' in result['error'] and 'Nothing was executed' in result['error']


def test_format_recovery_mentions_all_supported_choices_and_does_not_execute_dialogue():
    replies = iter(['Response 2/8: {"action":"hold"}', '{"action":"hold"}'])
    def backend(messages):
        reply = response('hold')
        reply['choices'][0]['message']['content'] = next(replies)
        return reply
    c = Controller('raw')
    record = v8.decide(backend, c, 'fixture')
    v8.validate_decision(record)
    assert record['decision_action'] == 'hold' and len(record['turns']) == 2 and not c.calls
    assert 'including submit or hold' in record['turns'][0]['model_feedback']


@pytest.mark.parametrize('arm', ['raw', 'library'])
def test_package_route_hint_does_not_execute_or_grant_raw_access(arm, workspace):
    controller = v8.Controller(workspace / ('runtime-' + arm), workspace, arm)
    result = controller.call('search_quant_methods', {'query': 'momentum'})
    assert not result['ok'] and controller.calls == [dict(tool='search_quant_methods', ok=False)]
    assert ('contract_lookup' in result) == (arm == 'library')
    if arm == 'library':
        assert result['contract_lookup'] == {'tool':'describe_library_tool',
                                             'arguments':{'name':'search_quant_methods'}}
        assert 'has not executed' in result['hint']


def test_rag_shortcut_rejects_package_arguments_without_discarding_them(workspace):
    tools = v9.Tools(workspace, 'library')
    result = tools.call('research_context', {'query': 'news', 'evidence': []})
    assert not result['ok']
    assert result['contract']['signature'] == 'research_context(query, offset=0, limit=3500)'
    assert 'already loads the visible evidence' in v9.system_prompt('library')


def test_current_cash_is_visible_without_choosing_the_next_allocation():
    assert '"CASH": 1.0' in v8.task('2025-01-02', {}, '2026-09-25')
    assert '"CASH": 0.25' in v8.task('2025-01-02', {'SPY': .75}, '2026-09-25')


def test_abandoned_or_shadowed_failure_cannot_be_reported_as_a_full_decision():
    record = v8.decide(lambda _: response('read_market'), Controller('raw'), 'fixture')
    record['turns'].pop()
    with pytest.raises(ValueError, match='complete response budget'):
        v8.validate_decision(record)
    record = v8.decide(lambda _: response('read_market'), Controller('raw'), 'fixture')
    record['turns'][0]['result']['decision_action'] = 'hold'
    with pytest.raises(ValueError, match='shadowed'):
        v8.validate_decision(record)
    record['turns'][0]['result'].pop('decision_action')
    record['turns'][0]['phase'] = 'final'
    with pytest.raises(ValueError, match='sequence or phase'):
        v8.validate_decision(record)


def test_rag_returns_same_day_evidence_and_routes_its_templates(workspace):
    import json
    path = workspace / 'evidence.json'
    packet = json.loads(path.read_text())
    packet['records'][0]['eligible_date'] = packet['as_of']
    path.write_text(json.dumps(packet))
    tools = v9.Tools(workspace, 'library')
    result = tools.call('research_context', {'query': 'synthetic observation'})
    assert result['ok'] and not result['evidence']['no_evidence']
    assert 'synthetic observation' in result['evidence']['context']
    assert result['tool_contracts'] or result.get('deferred_contracts')
    for contract in result['tool_contracts']:
        assert contract['call_template']['tool'] == 'call_library_tool'
        assert contract['call_template']['arguments']['name'] == contract['name']
        assert contract['template_requires_values'] is True


def test_decision_instructions_use_real_tool_names_in_both_phases():
    for arm in ('raw', 'library'):
        for index in (0, 6, 7):
            prompt = v8.system_prompt(arm, index)
            assert '"tool":"submit"' in prompt and '"tool":"hold"' in prompt
            assert 'REBALANCE' not in prompt
            assert 'arguments.rationale' in prompt
            assert 'No text, explanation, headings or second call outside it' in prompt


def test_unavailable_final_tool_gets_the_actual_decision_route():
    counter = []
    def backend(messages):
        counter.append(1)
        return response('read_market') if len(counter) < 7 else response('rebalance', {
            'allocation': {'SPY': 2, 'CASH': 1}})
    c = Controller('raw')
    record = v8.decide(backend, c, 'fixture')
    assert record['decision_action'] == 'failed' and len(c.calls) == 6
    assert 'tool="submit" with arguments.allocation' in record['turns'][-1]['result']['error']
