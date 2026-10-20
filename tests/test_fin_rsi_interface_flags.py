"""Fin-RSI opt-in flags: each one OFF is exactly the frozen default path; each one ON does its job.

Flags: --interface strict/tolerant, --algorithm-hint, --skill-search ranked, --skill-overlay,
--persistent-python, --python-only-hint, --decision-channel structured, --snapshot-tool-log
(and --skill-name-hint). The end-to-end check is a slow replay with every flag passed at its
default value, which must reproduce the frozen decisions of tests/test_fin_rsi_run.py.
"""
import json
import os
import subprocess

import pytest

from benchmarks.agent_study import fin_rsi_interface as fi
from benchmarks.agent_study import trading_capabilities as caps
from benchmarks.agent_study import trading_worker_v6

ENV = ('FIN_RSI_ALGORITHM_HINT', 'FIN_RSI_SKILL_SEARCH', 'FIN_RSI_SKILL_NAME_HINT',
       'FIN_RSI_SKILL_OVERLAY', 'FIN_RSI_SNAPSHOT_TOOL_LOG', 'FIN_RSI_PERSISTENT_PYTHON')


# ---------------------------------------------------------------- OFF = frozen default path

def test_tools_off_is_the_frozen_tools_class():
    assert fi.tools_class() is caps.Tools
    for off in (dict(algorithm_hint=False), dict(skill_search='substring'), dict(skill_name_hint=False),
                dict(skill_overlay=None), dict(skill_overlay=''), dict(snapshot_tool_log=False)):
        assert fi.tools_class(**off) is caps.Tools, off


def test_env_off_values_set_by_the_runner_give_frozen_tools(monkeypatch):
    for name in ENV:
        monkeypatch.delenv(name, raising=False)
    assert fi.env_tools_class() is caps.Tools
    for name, value in (('FIN_RSI_ALGORITHM_HINT', '0'), ('FIN_RSI_SKILL_SEARCH', 'substring'),
                        ('FIN_RSI_SKILL_NAME_HINT', '0'), ('FIN_RSI_SKILL_OVERLAY', ''),
                        ('FIN_RSI_SNAPSHOT_TOOL_LOG', '0')):
        monkeypatch.setenv(name, value)
    assert fi.env_tools_class() is caps.Tools


def test_decision_channel_off_leaves_bindings_untouched():
    bindings = {'submit': object(), 'run_algorithm': object()}
    assert fi.channel_bindings(bindings, None, None) is bindings
    assert fi.channel_bindings(bindings, None, dict(channel='free', holdings={})) is bindings


def test_worker_off_runs_plain_tolerant_python(monkeypatch, tmp_path):
    (tmp_path / 'request.json').write_text(json.dumps(dict(tool='run_python', arguments={'code': 'x=1'})))
    monkeypatch.setattr('sys.argv', ['worker', str(tmp_path)])
    monkeypatch.delenv('FIN_RSI_PERSISTENT_PYTHON', raising=False)
    assert fi.worker_run_python() is fi.run_python_tolerant


def test_strict_worker_is_untouched_without_tolerant(monkeypatch):
    # The worker entry swaps run_python only when FIN_RSI_INTERFACE == 'tolerant'.
    source = open(fi.__file__, encoding='utf-8').read()
    entry = source[source.index("if __name__ == '__main__':"):]
    assert "if os.environ.get('FIN_RSI_INTERFACE') == 'tolerant':" in entry
    assert trading_worker_v6.run_python.__module__ == 'benchmarks.agent_study.trading_worker_v6'


class _Parent:
    def __init__(self):
        self.seen, self.calls = [], []

    def call(self, tool, arguments):
        self.seen.append((tool, arguments))
        return dict(ok=False, error=f'unknown tool {tool!r}')


def _controller(**flags):
    class C(fi.InterfaceControllerMixin, _Parent):
        pass
    c = C()
    for k, v in flags.items():
        setattr(c, k, v)
    c.tools = type('T', (), {'python_bindings': lambda self: {'load_history': None}})()
    return c


def test_controller_mixin_off_passes_calls_through():
    c = _controller()
    args = {'code': 'print(1)'}
    assert c.call('run_python', args) == dict(ok=False, error="unknown tool 'run_python'")
    assert c.seen == [('run_python', args)]
    assert c.call('load_history', {}) == dict(ok=False, error="unknown tool 'load_history'")


# ---------------------------------------------------------------- ON = the documented change

def test_algorithm_hint_names_ready_ids():
    message = fi.unknown_algorithm_message('min_varience')
    assert 'Did you mean: min_variance' in message and 'Ready portfolio-weight ids:' in message
    assert fi.tools_class(algorithm_hint=True) is fi.HintTools


def test_ranked_skill_search():
    cards = [dict(name='alpha', description='nothing here'),
             dict(name='portfolio-optimizers', description='min variance and risk parity'),
             dict(name='risk', description='portfolio risk')]
    rows = fi.rank_skills(cards, 'portfolio optimizer')
    assert [r['name'] for r in rows] == ['portfolio-optimizers', 'risk']
    assert fi.rank_skills(cards, '') and fi.tools_class(skill_search='ranked') is fi.RankedSkillTools


def test_skill_overlay(tmp_path):
    import fin_skills
    name = sorted(fin_skills.names())[0]
    path = tmp_path / 'overlay.md'
    path.write_text('candidate text', encoding='utf-8')
    spec = f'{name}={path}'
    assert fi.load_skill_overlay(spec) == (name, 'candidate text')
    with pytest.raises(ValueError):
        fi.load_skill_overlay('no-such-skill=' + str(path))
    tools = object.__new__(fi.tools_class(skill_overlay=spec))
    page = tools.read_skill(name)
    assert page['text'] == 'candidate text'


def test_persistent_python_round_trip_and_controller_carry():
    import math
    state = fi.capture_namespace({'a': 1, 'b': [1, 2], 'm': math, 'f': lambda: 0, '__x': 1, 'bound': 3},
                                 {'bound'})
    assert fi.restore_namespace(state) == {'a': 1, 'b': [1, 2], 'm': math}
    c = _controller(persistent_python=True)
    c.call('run_python', {'code': 'x=1'})
    assert c.seen[-1][1][fi.STATE_IN] == ''


def test_python_only_hint():
    c = _controller(python_only_hint=True)
    error = c.call('load_history', {})['error']
    assert 'Python-only function' in error and 'load_history(lookback=252)' in error
    assert c.call('read_nothing', {})['error'] == "unknown tool 'read_nothing'"  # not a binding


def test_structured_channel_math_and_redirect():
    holdings = {'A': 0.5, 'B': 0.5, 'C': 0.0, 'CASH': 0.0}
    w, stats = fi.allocate_weights({'A': 0.7, 'B': 0.2, 'C': 0.1}, holdings, 0.5, 1.0)
    assert abs(sum(w.values()) - 1) < 1e-5 and max(w.values()) <= 0.5 + 1e-6
    w, stats = fi.allocate_weights({'A': 0.0, 'B': 0.0, 'C': 1.0}, holdings, 1.0, 0.25)
    assert abs(stats['turnover_achieved'] - 0.25) < 1e-6
    with pytest.raises(ValueError):
        fi.check_allocation('momentum')
    bound = fi.channel_bindings({'submit': None, 'run_algorithm': None}, None,
                                dict(channel='structured', holdings=holdings))
    with pytest.raises(ValueError, match='not available'):
        bound['submit']({'A': 1})
    assert 'allocate' in bound


def test_snapshot_tool_log_keeps_the_logged_reply(monkeypatch):
    shared = {'weights': {'SPY': 0.5}}

    def parent_call(self, name, arguments):
        self.calls.append(dict(tool=name, ok=True, weights=shared['weights']))
        return dict(ok=True, result=shared)
    monkeypatch.setattr(caps.Tools, 'call', parent_call)
    tools = object.__new__(fi.tools_class(snapshot_tool_log=True))
    tools.calls = []
    reply = tools.call('run_algorithm', {})
    reply['result']['weights']['SPY'] = 9.0          # agent edits the reply in place
    assert tools.calls[-1]['weights'] == {'SPY': 0.5}


def test_exec_line_is_pinned():
    import inspect
    assert inspect.getsource(fi).splitlines()[138].strip().startswith('exec(compile(code')


# ---------------------------------------------------------------- end to end, flags at defaults

@pytest.mark.slow
def test_replay_with_every_flag_at_its_default(tmp_path):
    import test_fin_rsi_run as run_test
    if not (run_test.DATA / run_test.ROWS).exists() or not run_test.PYTHON.exists():
        pytest.skip('needs the Beacon Fin-RSI result files and environment')
    out = tmp_path / 'defaults'
    env = dict(os.environ, PYTHONPATH=str(run_test.REPO), PYTHONNOUSERSITE='1',
               PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1')
    subprocess.run([str(run_test.PYTHON), str(run_test.RUNNER), '--rows', run_test.ROWS,
                    '--condition', 'C_unchanged', '--model', 'general14b=x', '--model', 'general32b=x',
                    '--interface', 'tolerant', '--backend', 'replay:' + run_test.ATTEMPT,
                    '--system-variant', 'sealed', '--task-variant', 'sealed', '--skill-search', 'substring',
                    '--decision-channel', '', '--sampling', 'default', '--flag-arms', '',
                    '--out', str(out)],
                   cwd=run_test.DATA, env=env, check=True, capture_output=True, text=True, timeout=1800)
    got = {p.name: run_test.projection_digest(json.loads((p / 'decision.json').read_text()))
           for p in sorted((out / 'rows').iterdir())}
    assert got == run_test.FROZEN['tolerant']
