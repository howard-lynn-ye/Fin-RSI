"""Diagnose observed model shape mistakes without executing a fallback strategy."""
import json
import pytest

from benchmarks.agent_study import trading_runtime_v8 as v8
from benchmarks.agent_study.linux_sandbox import confinement_available
from test_reader_contract_v8 import workspace  # noqa: F401


@pytest.mark.parametrize('arm', ['raw', 'library'])
@pytest.mark.parametrize('expression,exception', [
    ('q["close"].pct_change().cov()', 'TypeError'),
    ('q["close"].pct_change().mean().index', 'AttributeError'),
    ('q["SPY"]', 'KeyError'),
])
@pytest.mark.skipif(not confinement_available(), reason='requires Linux confinement')
def test_failed_shape_call_has_executable_neutral_recovery(arm, expression, exception, workspace):
    controller = v8.Controller(workspace/('shape-'+arm+'-'+exception), workspace, arm)
    failed = controller.call('run_python', {'code':
        'import pandas as pd; q=pd.read_csv("quotes.csv"); print('+expression+')'})
    assert not failed['ok'] and exception in failed['error']
    assert not failed.get('submission')
    assert len(controller.calls) == 1  # No hidden call or strategy substitution.
    assert 'long format' in failed['hint']
    assert 'Study tools are already bound' not in failed['hint']
    call = failed['recovery_call']
    assert call['tool'] == 'run_python'
    assert call['arguments']['code'] == controller.tools.help_tool('run_python')['raw_matrix_example']
    recovered = controller.call(call['tool'], call['arguments'])
    assert recovered['ok'], recovered
    assert not recovered.get('submission')
    last = json.loads(recovered['output'].strip().splitlines()[-1])
    assert last['type'] == 'DataFrame' and last['shape'] == [2, 12]
    assert 'SPY' in last['columns']
    other = v8.Tools(workspace, 'library' if arm == 'raw' else 'raw')
    assert other.help_tool('run_python')['raw_matrix_example'] == call['arguments']['code']


@pytest.mark.skipif(not confinement_available(), reason='requires Linux confinement')
def test_unrelated_type_error_is_not_diagnosed_as_a_tool_or_table_failure(workspace):
    controller = v8.Controller(workspace/'unrelated-type', workspace, 'raw')
    failed = controller.call('run_python', {'code': 'print(1 + "x")'})
    assert not failed['ok'] and 'TypeError' in failed['error']
    assert 'Study tools are already bound' not in failed['hint']
    assert 'recovery_call' not in failed
    assert not failed.get('submission')
