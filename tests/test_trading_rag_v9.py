"""RAG access, output delivery, final-decision rules and raw-arm isolation."""
import json

import pytest

from benchmarks.agent_study import trading_runtime_v8 as v8, trading_runtime_v9 as v9
from benchmarks.agent_study.linux_sandbox import confinement_available
from benchmarks.agent_study.trading_tools_v5 import OUTPUT_LIMIT, wire
from benchmarks.agent_study.trading_worker_v6 import run_python
from test_trading_terminal_v8 import workspace, response, Controller  # noqa: F401


def test_rag_is_directly_discoverable_but_raw_prompt_is_unchanged():
    assert 'research_context(query=' in v9.system_prompt('library')
    assert v9.system_prompt('raw') == v8.system_prompt('raw')
    assert v9.system_prompt('library', 7).endswith('No research calls.')


def test_direct_query_delivers_evidence_contracts_and_bound_python(workspace):
    tools = v9.Tools(workspace, 'library')
    result = tools.call('research_context', {'query': 'observation'})
    assert result['ok'] and 'knowledge' in result and 'evidence' in result, result
    assert 'synthetic observation' in result['evidence']['context']
    assert all('(' in c['signature'] for c in result['study_contracts'])
    assert len(wire(result)) <= OUTPUT_LIMIT and not json.loads(wire(result)).get('output_omitted')
    for example in result['study_examples']:
        assert tools.call(example['tool'], example['arguments'])['ok']
    assert run_python('print(research_context("observation")["ok"])', tools)['ok']
    assert not v9.Tools(workspace, 'raw').call('research_context', {'query': 'observation'})['ok']


def test_snapshot_future_record_is_rejected_before_rag(workspace):
    path = workspace / 'evidence.json'
    packet = json.loads(path.read_text())
    packet['records'][0]['eligible_date'] = '2099-01-01'
    path.write_text(json.dumps(packet))
    result = v9.Tools(workspace, 'library').call('research_context', {'query': 'observation'})
    assert not result['ok'] and 'future' in result['error']


def test_rag_does_not_choose_a_portfolio_or_add_decision_turns():
    calls = []
    def backend(messages):
        calls.append(messages[0]['content'])
        return response('research_context', {'query': 'risk'})
    record = v9.decide(backend, Controller('library'), 'fixture')
    v9.validate_decision(record)
    assert len(calls) == 8 and record['interface'] == 'v9'
    assert record['decision_action'] == 'failed' and record['target'] is None
    assert 'research was not executed' in record['turns'][-1]['result']['error']


@pytest.mark.skipif(not confinement_available(), reason='requires Linux confinement')
def test_direct_rag_runs_inside_real_worker_and_raw_remains_isolated(workspace):
    for arm in ('raw', 'library'):
        controller = v9.Controller(workspace / ('runtime-' + arm), workspace, arm)
        result = controller.call('research_context', {'query': 'observation'})
        assert bool(result.get('ok')) == (arm == 'library'), result
        if arm == 'library':
            assert v8.execution_status(result)['turn_ok'] and 'evidence' in result
        code = 'print(research_context("observation")["ok"])'
        result = controller.call('run_python', {'code': code})
        assert bool(result.get('ok')) == (arm == 'library'), result
        assert not controller.call('run_python', {'code': 'open("../future.json").read()'})['ok']
