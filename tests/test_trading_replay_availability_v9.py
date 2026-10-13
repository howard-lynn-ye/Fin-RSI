"""Retrieved package contracts must obey the same policy as execution."""
import pytest

from benchmarks.agent_study import trading_runtime_v9 as v9
from test_trading_terminal_v8 import workspace  # noqa: F401


@pytest.mark.parametrize('name,status', [('fetch_market_data', 'outside_replay'),
                                       ('future_unreviewed_tool', 'unreviewed'),
                                       ('search_skills', 'offline_allowed')])
def test_retrieval_does_not_offer_an_unexecutable_call_template(workspace, monkeypatch, name, status):
    import fin_skills.rag.research as rag
    monkeypatch.setattr(rag, 'research_context', lambda *a, **kw: dict(
        knowledge={}, evidence={}, tool_contracts=[dict(name=name,
        input_schema={'type': 'object'}, call_template=dict(tool=name, arguments={}))]))
    result = v9.Tools(workspace, 'library').research_context('visible market information')
    card, = result['tool_contracts']
    assert card['study_status'] == status
    if status == 'offline_allowed':
        assert card['call_template']['tool'] == 'call_library_tool'
        assert card['call_template']['arguments']['name'] == name
    else:
        assert card['call_template'] is None
        assert 'Unavailable' in card['transport']
    assert 'read_snapshot/read_market' in result['replay_policy']


def test_rejected_live_tool_has_no_misleading_retry_route(workspace):
    tools = v9.Tools(workspace, 'library')
    result = tools.call('fetch_market_data', {'start': '2015-01-01'})
    assert not result['ok'] and result['study_status'] == 'outside_replay'
    assert 'contract_lookup' not in result
    assert 'No wrapper can enable' in result['hint']
    assert result['available_readers'] == ['read_snapshot', 'read_market', 'read_evidence']


def test_allowed_package_tool_still_has_its_explicit_discovery_route(workspace):
    result = v9.Tools(workspace, 'library').call('search_skills', {'query': 'risk'})
    assert not result['ok'] and result['study_status'] == 'offline_allowed'
    assert result['contract_lookup'] == {'tool': 'describe_library_tool',
                                         'arguments': {'name': 'search_skills'}}
