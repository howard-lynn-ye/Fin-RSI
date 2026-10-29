import ctypes
import inspect
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import pytest

from benchmarks.agent_study import codegen_enforced as v1
from benchmarks.agent_study import codegen_enforced_v2 as v2
from benchmarks.agent_study.codegen_utility import (
    SEEDS, TASKS, dataset, library_context, reference)
import fin_skills.algorithms as algorithms
from fin_skills.api.rules import card


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
OK = {'executed': True, 'library_calls': 0, 'checks': {
    'split': {'passed': True, 'max_abs_diff': 0.0},
    'lag': [{'k': 40, 'passed': True, 'summary': 'PASS'}]}}


@pytest.fixture
def package_path(monkeypatch):
    monkeypatch.setenv('PYTHONPATH', str(ROOT))


def _failing(split=False, lag=False, calls=1):
    out = json.loads(json.dumps(OK))
    out['library_calls'] = calls
    if split:
        out['checks']['split'].update(passed=False, max_abs_diff=0.5)
    if lag:
        out['checks']['lag'][0].update(passed=False, summary='FAIL assert_causal')
    return out


@pytest.mark.parametrize('task', TASKS)
@pytest.mark.parametrize('arm', ['raw_exec', 'raw_guards'])
def test_raw_arm_prompts_are_byte_identical_to_v1(task, arm):
    data = dataset(SEEDS[0])
    assert (v2.messages(task, data, arm, v2.library_context_v2())
            == v1.messages(task, data, arm, library_context()))


def test_library_and_rules_arms_get_the_task_packet_rules_and_budget():
    data, context = dataset(11), v2.library_context_v2()
    raw = json.loads(v2.messages('ma_crossover', data, 'raw_exec', context)[1]['content'])
    library = v2.messages('ma_crossover', data, 'library_guards', context)
    rules = v2.messages('ma_crossover', data, 'rules_guards', context)
    packet = json.loads(library[1]['content'])
    assert packet.pop('library_discovery')['interface'] == v2.INTERFACE and packet == raw
    assert json.loads(rules[1]['content']) == raw
    for system in (library[0]['content'], rules[0]['content']):
        assert v1.ONE_SHOT not in system and v1.MULTI in system and v1.CHECKS in system
        assert 'Rules that apply to this task:' in system and '[execution_lag]' in system
    block = rules[0]['content'].split('Rules that apply to this task:')[1]
    assert 'fin_skills' not in block                    # the rules arm may not import it
    assert 'fin_skills.algorithms.positions()' in library[0]['content']
    assert v2.cards('ma_crossover') == ['execution_lag', 'feature_causality', 'split_once',
                                        'same_convention', 'warmup']
    assert 'first_return' in v2.cards('diagonal_risk_parity')


def test_interface_names_real_signatures_and_input_keys():
    assert 'data_dictionary' not in v2.INTERFACE
    assert list(inspect.signature(algorithms.run).parameters)[:2] == ['algorithm_id', 'data']
    p = inspect.signature(algorithms.positions).parameters
    assert {'window', 'parameters', 'lag', 'split_factors'} <= set(p)
    inputs = {c['id']: c['inputs'] for c in algorithms.catalog()}
    for algorithm, key in [('inverse_volatility', 'asset_returns'),
                           ('cross_sectional_momentum', 'asset_returns'),
                           ('ma_crossover', 'prices'), ('vol_target_momentum', 'prices')]:
        assert inputs[algorithm] == (key,) and algorithm in v2.INTERFACE
        assert f'"{key}"' in v2.INTERFACE


DOCUMENTED = {
    'diagonal_risk_parity': ('inverse_volatility', dict(window=60)),
    'cross_sectional_momentum': ('cross_sectional_momentum',
                                 dict(parameters=dict(lookback=60, top_k=2))),
    'ma_crossover': ('ma_crossover', dict(parameters=dict(fast=5, slow=20))),
    'vol_target_momentum': ('vol_target_momentum', dict(parameters=dict(
        lookback=20, target_vol=0.10, periods_per_year=252, max_exposure=1.0)))}


@pytest.mark.parametrize('task', TASKS)
def test_calls_written_from_the_interface_text_run_and_match_the_reference(task):
    data = dataset(11)
    index = pd.to_datetime(data['dates'])
    prices = pd.DataFrame(data['prices'], index=index, columns=data['assets'])
    factors = pd.DataFrame(data['factors'], index=index, columns=data['assets'])
    algorithm, kwargs = DOCUMENTED[task]
    w = algorithms.positions(algorithm, prices, factors, **kwargs)
    np.testing.assert_allclose(w.to_numpy(), reference(task, data), atol=1e-12, rtol=0)


def test_raw_arm_decisions_are_v1_decisions():
    cases = [OK, _failing(split=True), _failing(lag=True), _failing(split=True, lag=True),
             {'executed': False, 'error': {'type': 'ValueError', 'message': 'shape',
                                           'traceback': 'a\nValueError: shape'}}]
    for arm in ('raw_exec', 'raw_guards'):
        for result in cases:
            assert v2.decide(arm, result, 2) == v1.decide(arm, result, 2)


def test_library_and_rules_feedback_adds_cause_and_rule_without_answers():
    accepted, message = v2.decide('library_guards', _failing(split=True), 1)
    assert not accepted and 'Split-representation check failed' in message
    assert 'Check split_representation failed' in message and '[split_once]' in message
    assert 'adjust_prices' in message and message.endswith(
        'Return the complete corrected program.')
    assert 'weight' not in message.lower() and 'reference' not in message.lower()
    accepted, message = v2.decide('rules_guards', _failing(lag=True, calls=0), 1)
    assert not accepted and '[execution_lag]' in message and 'fin_skills' not in message
    accepted, message = v2.decide('library_guards', _failing(calls=0), 1)
    assert not accepted and message.count('\n') == 2 and v2.V2_LIBRARY in message
    assert v2.decide('library_guards', _failing(), 1) == (True, None)
    assert v2.decide('rules_guards', _failing(calls=0), 1) == (True, None)


def test_failed_checks_ignore_checks_that_raised():
    raised = json.loads(json.dumps(OK))
    raised['checks']['split'] = {'passed': False, 'error': {'type': 'TypeError',
                                                            'message': 'x'}}
    assert v2.failed_checks(raised) == []
    assert v2.failed_checks(_failing(split=True, lag=True)) == ['split_representation',
                                                                'execution_lag']
    assert v2.failed_checks({'executed': False}) == []


def test_rule_cards_named_in_protocol_exist(tmp_path):
    v2.freeze(tmp_path/'root', only=set())
    protocol = json.loads((tmp_path/'root'/'protocol.json').read_text())
    assert protocol['version'] == 'codegen-enforced-v2'
    assert protocol['arms'] == list(v2.DEFAULT_ARMS)
    for ids in protocol['rule_cards'].values():
        assert [card(i).id for i in ids] == ids
    with pytest.raises(ValueError, match='arms'):
        v2.freeze(tmp_path/'other', arms=('raw_exec', 'library'))


def test_default_freeze_has_72_episodes_and_rules_arm_is_optional(tmp_path):
    v2.freeze(tmp_path/'a')
    rows = json.loads((tmp_path/'a'/'inputs.json').read_text())
    assert len(rows) == 72 and {r['arm'] for r in rows} == set(v2.DEFAULT_ARMS)
    assert all(r['rule_cards'] for r in rows if r['arm'] == 'library_guards')
    assert not any(r['rule_cards'] for r in rows if r['arm'].startswith('raw'))
    v2.freeze(tmp_path/'b', arms=v2.ARMS)
    assert len(json.loads((tmp_path/'b'/'inputs.json').read_text())) == 96


KEYWORD_RUN = v2.QUALIFY['library_keyword_run'][2]


@linux_only
def test_v1_tracer_rejects_a_correct_keyword_call_and_v2_runs_it(tmp_path, package_path):
    root = tmp_path/'root'
    (root/'tmp').mkdir(parents=True)
    data = dataset(991)
    old = v1.controller_execute(KEYWORD_RUN, data, root, checks=True)
    assert not old['executed'] and old['error']['type'] == 'TypeError'
    new = v2.controller_execute(KEYWORD_RUN, data, root, checks=True)
    assert new['executed'] and new['library_calls'] > 0
    assert v2.decide('library_guards', new, 1) == (True, None)
    graded, audit = v2.grade(KEYWORD_RUN, data, 'ma_crossover', root)
    assert graded['executed'] and audit['financially_valid']


@linux_only
def test_controller_qualification_passes(tmp_path, package_path):
    root = tmp_path/'root'
    v2.freeze(root)
    v2.qualify(root)
    assert json.loads((root/'qualification.json').read_text())['passed']


class Scripted:
    """Deterministic stand-in for the model: a fixed program per arm and attempt."""
    programs = {
        'raw_exec': ['def solve(prices, split_factors):\n    return prices.\n',
                     v1.QUALIFY['same_session'][2]],
        'raw_guards': [v1.QUALIFY['same_session'][2], v1.QUALIFY['raw_without_library'][2]],
        'library_guards': [v1.QUALIFY['raw_without_library'][2],
                           v2.QUALIFY['library_positions'][2]]}

    def __init__(self, *_):
        self.seed = self.calls = 0

    def __call__(self, history):
        system = history[0]['content']
        arm = ('library_guards' if 'rejects any program' in system else
               'raw_guards' if v1.CHECKS in system else 'raw_exec')
        attempt = (len(history) - 2) // 2
        self.calls += 1
        text = '```python\n' + self.programs[arm][attempt] + '```\nExplanation.'
        return {'choices': [{'message': {'content': text}, 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15},
                'generation_seconds': 0.0}


@pytest.mark.slow
@linux_only
def test_scripted_episodes_are_scored(tmp_path, package_path):
    root = tmp_path/'root'
    v2.freeze(root, only={f'7b-ma_crossover-11-{arm}' for arm in v2.DEFAULT_ARMS})
    v2.qualify(root)
    v2.run(root, backend_factory=Scripted)
    v2.score(root)
    summary = json.loads((root/'scores.json').read_text())['aggregate']
    raw_exec, raw_guards, library = (summary['7b-'+arm] for arm in v2.DEFAULT_ARMS)
    assert raw_exec['accepted'] == 1 and raw_exec['valid_deliverable'] == 0
    assert raw_exec['final_timing_failures'] == 1 and raw_exec['final_strict_lag_failures'] == 1
    assert raw_guards['valid_deliverable'] == 1 and raw_guards['mean_attempts'] == 2
    assert library['valid_deliverable'] == 1 and library['library_users'] == 1
    assert library['final_strict_lag_failures'] == 0


@pytest.mark.slow
@linux_only
def test_rsi_loop_learns_from_real_controller_checks_on_cpu(tmp_path, package_path):
    """A small RSI evolution on real sandbox checks and the real grader (L7 acceptance).

    The model is scripted: without the split lesson in its playbook it re-cumulates the
    split factors; with it, it writes the correct rule. One validation task always trades
    on the same close, so some case always fails: the all_pass gate can never accept, the
    no_regression gate accepts the lesson, and the frozen playbook does not change again.
    """
    from functools import lru_cache
    from fin_skills.rsi import Attempt, Policy, loop_from_checks, observe
    root = tmp_path/'root'
    (root/'tmp').mkdir(parents=True)
    lesson = 'split_factors are already cumulative; multiply once and never cumprod them'
    splits = {'development': ['ma-11', 'ma-23'], 'validation': ['ma-37', 'hard-991']}

    @lru_cache(maxsize=None)
    def run_attempt(task_id, playbook):
        kind, seed = task_id.split('-')
        code = (v1.QUALIFY['same_session'][2] if kind == 'hard' else
                v1.QUALIFY['raw_without_library'][2] if lesson in playbook else
                v1.QUALIFY['recumulated_split'][2])
        data = dataset(int(seed))
        out = v2.controller_execute(code, data, root, checks=True)
        checks = {'split_representation': bool(out['checks']['split']['passed']),
                  'execution_lag': all(l['passed'] for l in out['checks']['lag'])}
        _, audit = v2.grade(code, data, 'ma_crossover', root)
        valid = bool(audit.get('financially_valid')) and all(checks.values())
        return Attempt(task_id=task_id, code=code, executed=bool(out['executed']),
                       checks=checks, valid=valid)

    prompts = []

    def proposer(prompt):
        prompts.append(prompt)
        assert 'ma-37' not in prompt and 'hard-991' not in prompt and '"valid"' not in prompt
        assert 'split_representation' in prompt and '[split_once]' in prompt
        return json.dumps({'key': 'split_factors', 'surface': 'experience', 'text': lesson,
                           'rationale': 'development traces fail the split check'})

    def make(name, policy):
        loop = loop_from_checks(tmp_path/name, proposer, run_attempt, splits,
                                model_id='scripted@1', protocol_id='rsi-real-checks',
                                available_at='2026-11-01T00:00:00Z', policy=policy,
                                evidence_dir=tmp_path/name/'evidence')
        observe(loop, [run_attempt(t, '') for t in splits['development']],
                available_at='2026-11-01T00:00:00Z', source_version='test')
        return loop

    strict = make('all-pass', Policy(enabled=True))
    assert strict.improve(as_of='2026-11-02T00:00:00Z').status == 'rejected'
    loop = make('no-regression', Policy(enabled=True, guard_rule='no_regression'))
    decision = loop.improve(as_of='2026-11-02T00:00:00Z')
    assert decision.status == 'accepted', decision.reason
    book = loop.instructions(as_of='2026-11-02T00:00:00Z')
    assert lesson in book
    loop.freeze(as_of='2026-11-03T00:00:00Z')
    assert loop.improve(as_of='2026-11-04T00:00:00Z').status == 'frozen'
    assert loop.instructions(as_of='2026-11-04T00:00:00Z') == book
    assert run_attempt('ma-37', book).valid and not run_attempt('ma-37', '').valid
