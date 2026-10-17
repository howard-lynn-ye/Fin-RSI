"""Observed full-cycle errors, repaired equally without supplying a strategy."""
import json
import pytest
from benchmarks.agent_study import trading_runtime_v8 as v8
from benchmarks.agent_study import trading_runtime_v9 as v9
from benchmarks.agent_study.linux_sandbox import confinement_available
from test_reader_contract_v8 import workspace  # noqa: F401


@pytest.mark.parametrize('arm', ['raw', 'library'])
@pytest.mark.skipif(not confinement_available(), reason='requires Linux confinement')
def test_scientific_aliases_work_in_real_worker_without_imports(arm, workspace):
    c = v9.Controller(workspace/('aliases-'+arm), workspace, arm)
    result = c.call('run_python', {'code':
        'q=pd.read_csv("quotes.csv")\n'
        'def objective(x): return np.sum(x*x)\n'
        'r=sco.minimize(objective, [2.0, 3.0]); '
        'print(json.dumps({"success":bool(r.success), "value":float(r.fun), '
        '"columns":q.columns.tolist()}))'})
    assert result['ok'], result
    data = json.loads(result['output'])
    assert data['success'] and data['value'] < 1e-8
    assert 'ticker' in data['columns']
    assert result['submission'] is None
    assert len(c.calls) == 1
    # User-defined state still does not persist; no raw access to library.
    assert c.call('run_python', {'code': 'reply=123; print(reply)'})['ok']
    missing = c.call('run_python', {'code': 'print(reply)'})
    assert not missing['ok'] and 'starts fresh' in missing['hint']
    if arm == 'raw':
        assert not c.call('run_python', {'code': 'import fin_skills'})['ok']


@pytest.mark.parametrize('arm', ['raw', 'library'])
@pytest.mark.parametrize('code,phrase', [
    ('q=pd.read_csv("quotes.csv"); print(q["CASH"])', 'account balance'),
    ('q=pd.read_csv("quotes.csv"); q.append(q)', 'pd.concat'),
    ('from datetime import datetime; print("2025-01-02" < datetime.now())', 'pd.to_datetime'),
    ('sco.minimize(lambda x: np.sum(x*x), [1., 2.], bounds=[(0., 1.)]*3)', 'len(x0)'),
])
@pytest.mark.skipif(not confinement_available(), reason='requires Linux confinement')
def test_exact_failures_remain_failures_with_actionable_feedback(arm, code, phrase, workspace):
    c = v9.Controller(workspace/('feedback-'+arm), workspace, arm)
    result = c.call('run_python', {'code': code})
    assert not result['ok'], result
    assert phrase in result['hint']
    assert result['submission'] is None and len(c.calls) == 1
    assert 'recovery_call' not in result  # No alternate analysis silently run.


def test_environment_contract_matches_both_arms(workspace):
    raw, library = (v9.Tools(workspace, arm) for arm in ('raw', 'library'))
    assert raw.help_tool('run_python') == library.help_tool('run_python')
    for alias in raw.help_tool('run_python')['prebound_modules']:
        assert raw.python_bindings()[alias] is library.python_bindings()[alias]
    assert 'CASH' not in raw.python_bindings()
    assert 'CASH' not in library.load_history(lookback=1)['returns'].columns
    assert 'actual' in v8.COMMON and 'CASH is an account balance' in v8.COMMON
