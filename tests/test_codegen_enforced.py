import ctypes
import json
from pathlib import Path
import platform
import sys

import pytest

from benchmarks.agent_study import codegen_enforced as ce
from benchmarks.agent_study.codegen_utility import dataset, library_context


def _confinement_available():
    if sys.platform != 'linux' or platform.machine() != 'x86_64':
        return False
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        libc.syscall.restype = ctypes.c_long
        ctypes.CDLL('libseccomp.so.2')
        return libc.syscall(444, 0, 0, 1) >= 3
    except OSError:
        return False


linux_only = pytest.mark.skipif(not _confinement_available(),
                                reason='real confinement needs Linux Landlock and libseccomp')
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def package_path(monkeypatch):
    # conftest moves the working directory; the frozen E4 grader's worker needs the package.
    monkeypatch.setenv('PYTHONPATH', str(ROOT))


def test_arms_share_the_task_packet_and_announce_the_attempt_budget():
    data, context = dataset(11), library_context()
    packets = {arm: ce.messages('ma_crossover', data, arm, context) for arm in ce.ARMS}
    raw = json.loads(packets['raw_exec'][1]['content'])
    assert json.loads(packets['raw_guards'][1]['content']) == raw
    library = json.loads(packets['library_guards'][1]['content'])
    assert library.pop('library_discovery') and library == raw
    for arm, (system, _) in packets.items():
        assert ce.ONE_SHOT not in system['content'] and ce.MULTI in system['content']
    assert ce.CHECKS in packets['raw_guards'][0]['content']
    assert ce.CHECKS in packets['library_guards'][0]['content']
    assert ce.CHECKS not in packets['raw_exec'][0]['content']


def test_controller_decisions_follow_the_arm():
    ok = {'executed': True, 'library_calls': 0, 'checks': {
        'split': {'passed': True, 'max_abs_diff': 0.0},
        'lag': [{'k': 40, 'passed': True, 'summary': 'PASS'}]}}
    assert ce.decide('raw_exec', {'executed': True}, 1) == (True, None)
    assert ce.decide('raw_guards', ok, 1) == (True, None)
    accepted, message = ce.decide('library_guards', ok, 2)
    assert not accepted and 'Library requirement failed' in message
    assert 'attempt 2 of 3' in message
    lagged = json.loads(json.dumps(ok))
    lagged['checks']['lag'][0].update(passed=False, summary='FAIL assert_causal')
    assert 'Lag check failed at k=40' in ce.decide('raw_guards', lagged, 1)[1]
    crashed = {'executed': False, 'error': {'type': 'ValueError', 'message': 'shape',
                                            'traceback': 'a\nb\nValueError: shape'}}
    accepted, message = ce.decide('raw_exec', crashed, 1)
    assert not accepted and 'ValueError: shape' in message


@linux_only
def test_controller_qualification_passes(tmp_path, package_path):
    root = tmp_path/'root'
    ce.freeze(root)
    assert len(json.loads((root/'inputs.json').read_text())) == 72
    ce.qualify(root)
    assert json.loads((root/'qualification.json').read_text())['passed']


class Scripted:
    """Deterministic stand-in for the model: a fixed program per arm and attempt."""
    programs = {
        'raw_exec': ['def solve(prices, split_factors):\n    return prices.\n',
                     ce.QUALIFY['same_session'][2]],
        'raw_guards': [ce.QUALIFY['same_session'][2], ce.QUALIFY['raw_without_library'][2]],
        'library_guards': [ce.QUALIFY['raw_without_library'][2],
                           ce.QUALIFY['library_correct'][2]]}

    def __init__(self, *_):
        self.seed = self.calls = 0

    def __call__(self, history):
        system = history[0]['content']
        arm = ('library_guards' if 'rejects any program' in system else
               'raw_guards' if ce.CHECKS in system else 'raw_exec')
        attempt = (len(history) - 2) // 2
        self.calls += 1
        text = '```python\n' + self.programs[arm][attempt] + '```\nExplanation.'
        return {'choices': [{'message': {'content': text}, 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15},
                'generation_seconds': 0.0}


@pytest.mark.slow
@linux_only
def test_scripted_episodes_are_scored_by_the_unchanged_grader(tmp_path, package_path):
    root = tmp_path/'root'
    ce.freeze(root, only={f'7b-ma_crossover-11-{arm}' for arm in ce.ARMS})
    ce.qualify(root)
    ce.run(root, backend_factory=Scripted)
    ce.score(root)
    summary = json.loads((root/'scores.json').read_text())['aggregate']
    raw_exec, raw_guards, library = (summary['7b-'+arm] for arm in ce.ARMS)
    assert raw_exec['accepted'] == 1 and raw_exec['valid_deliverable'] == 0
    assert raw_exec['final_timing_failures'] == 1
    assert raw_guards['valid_deliverable'] == 1 and raw_guards['mean_attempts'] == 2
    assert library['valid_deliverable'] == 1 and library['library_users'] == 1
    assert library['first_valid'] == 1  # the rejected raw program was itself valid
