"""Replay observed discovery mistakes without inventing or executing a strategy."""
import json
import pytest

import fin_skills
from fin_skills.algorithms import catalog, TASKS
from fin_skills.algorithms.core import ALIASES, task_name
from fin_skills.tools import call_tool, list_tools
from benchmarks.agent_study import trading_runtime_v8 as v8
from benchmarks.agent_study.trading_tools_v5 import wire, OUTPUT_LIMIT
from test_trading_terminal_v8 import workspace  # noqa: F401


def test_public_task_schema_matches_runtime_names_and_aliases():
    specs = {row['name']: row['input_schema'] for row in list_tools()}
    for name in ('list_algorithms', 'auto_algorithm', 'profile_algorithm_data', 'research_algorithms'):
        declared = specs[name]['properties']['task']['enum']
        assert set(declared) == set(TASKS) | set(ALIASES)
        assert all(task_name(value) in TASKS for value in declared)
        assert 'investment' not in declared
    assert specs['recommend_algorithms']['properties']['request']['properties']['task']['enum'] == declared


@pytest.mark.parametrize('task', [*TASKS, *ALIASES])
def test_every_advertised_task_can_query_the_actual_registry(task):
    result = call_tool('list_algorithms', {'task': task})
    assert all(row['task'] == task_name(task) for row in result['algorithms'])


def test_invalid_task_recovers_to_neutral_catalog_without_automatic_execution(workspace):
    tools = v8.Tools(workspace, 'library')
    failed = tools.call('list_algorithms', {'task': 'investment'})
    assert not failed['ok'] and 'investment' in failed['error']
    assert set(failed['contract']['allowed_tasks']) == set(TASKS)
    recovery = failed['recovery_call']
    assert recovery == {'tool': 'list_algorithms', 'arguments': {}}
    assert tools.calls == [{'tool': 'list_algorithms', 'ok': False}]
    page = tools.call(recovery['tool'], recovery['arguments'])
    assert page['ok'] and page['algorithms']
    assert {row['id'] for row in page['algorithms']} <= {row['id'] for row in catalog()}
    assert all(row['tool'] == 'list_algorithms' for row in tools.calls)
    assert len(wire(failed)) <= OUTPUT_LIMIT and json.loads(wire(failed))['recovery_call'] == recovery


def test_skill_name_routes_to_document_only_after_unknown_algorithm_error(workspace):
    assert 'signal-reconciler' in fin_skills.names()
    tools = v8.Tools(workspace, 'library')
    failed = tools.call('describe_algorithm', {'algorithm_id': 'signal-reconciler'})
    assert not failed['ok'] and failed['identifier_kind'] == 'skill_document'
    recovery = failed['recovery_call']
    assert recovery == {'tool': 'read_skill', 'arguments': {'name': 'signal-reconciler'}}
    assert tools.calls == [{'tool': 'describe_algorithm', 'ok': False}]
    document = tools.call(recovery['tool'], recovery['arguments'])
    assert document['ok'] and document['text']
    assert not any(call['tool'] in ('run_algorithm', 'submit') for call in tools.calls)
    assert json.loads(wire(failed))['identifier_kind'] == 'skill_document'


def test_raw_condition_gets_no_algorithm_or_skill_access(workspace):
    tools = v8.Tools(workspace, 'raw')
    failed = tools.call('describe_algorithm', {'algorithm_id': 'signal-reconciler'})
    assert not failed['ok'] and 'recovery_call' not in failed and 'identifier_kind' not in failed
    for name in ('list_algorithms', 'read_skill'):
        assert not tools.call(name, {})['ok']
    assert 'valid task filters:' not in v8.system_prompt('raw')
    assert all(task in v8.system_prompt('library') for task in TASKS)
