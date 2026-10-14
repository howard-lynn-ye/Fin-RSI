"""Real wrapper regressions: Python exit success is not nested tool success."""
import pytest

from benchmarks.agent_study import trading_runtime_v8 as v8, trading_runtime_v9 as v9
from benchmarks.agent_study import trading_preflight_v8 as gate
from benchmarks.agent_study.linux_sandbox import confinement_available
from benchmarks.agent_study.trading_tools_v5 import text_page, rows_page
from test_trading_terminal_v8 import workspace, response  # noqa: F401


@pytest.mark.skipif(not confinement_available(), reason='requires Linux confinement')
@pytest.mark.parametrize('recover', [False, True])
def test_nested_failure_is_visible_and_successful_recovery_still_counts(workspace, recover):
    controller = v9.Controller(workspace/('nested-'+str(recover)), workspace, 'library')
    code = 'print(describe_algorithm("not-an-algorithm")["ok"])'
    if recover:
        code += '\nprint(read_snapshot()["ok"])'
    replies = iter([response('run_python', {'code': code}), response('hold')])
    record = v9.decide(lambda _: next(replies), controller, 'fixture')
    v9.validate_decision(record)
    result = record['turns'][0]['result']
    assert result['ok'] and record['decision_action'] == 'hold'
    summary = result['nested_tool_summary']
    assert summary['failed_receipts'] == 1
    assert summary['failed_tools'] == ['describe_algorithm']
    assert summary['all_failed'] is (not recover)
    observed = gate.observations(record)
    assert observed['research_successes'] == int(recover)
    assert observed['research_errors'] == int(not recover)
    assert observed['nested_tool_failed_receipts'] == 1


@pytest.mark.skipif(not confinement_available(), reason='requires Linux confinement')
def test_negative_guard_finding_is_a_successful_tool_result(workspace):
    controller = v9.Controller(workspace/'negative-guard', workspace, 'library')
    code = ('r=call_library_tool("check_rf_convention", '
            '{"returns":[0.01,-0.02,0.005,0.012],"rf":5}); '
            'assert r["ok"] and not r["result"]["passed"]; print("flag delivered")')
    result = controller.call('run_python', {'code': code})
    assert result['ok'] and 'flag delivered' in result['output']
    assert result['nested_tool_summary']['successful_receipts'] > 0
    assert result['nested_tool_summary']['failed_receipts'] == 0
    assert not result['nested_tool_summary']['all_failed']


@pytest.mark.parametrize('value', [True, 1.5, '1', None])
@pytest.mark.parametrize('field', ['offset', 'limit'])
def test_paging_does_not_silently_coerce_model_arguments(value, field):
    with pytest.raises((TypeError, ValueError), match='integer'):
        text_page('abc', **{field: value})
    args = dict(offset=0, limit=1)
    args[field] = value
    with pytest.raises((TypeError, ValueError), match='integer'):
        rows_page([1, 2], key='rows', **args)


def test_existing_failed_preflight_rejects_before_loading_model_or_writing_start(tmp_path, monkeypatch):
    pytest.importorskip('bs4', reason='study integration requires the study extra')
    from benchmarks.agent_study import multisource_model_study as study
    p = dict(interface='v9', batch='matrix_hf', model=['32b','unused','unused'])
    monkeypatch.setattr(study, 'verify', lambda _: p)
    (tmp_path/'decision-preflight.json').write_text('{}')
    calls = []
    def reject(root, **kwargs):
        calls.append(root)
        raise ValueError('existing pilot failed: reject before backend')
    monkeypatch.setattr(gate, 'require_campaign_ready', reject)
    with pytest.raises(ValueError, match='reject before backend'):
        study.run(tmp_path)
    assert calls == [tmp_path]
    assert not (tmp_path/'inference-started.json').exists()
