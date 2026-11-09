"""Fault injection against real Meter/benchmark control flow; no model inference."""
from dataclasses import dataclass
import json
from types import SimpleNamespace

import pytest

from scripts.study_continuation import (OperationStore, ReplayBackend, UnresolvedOperation,
                                        checkpointed_method, digest, restore_fit)
from fin_skills.rsi.methods import BaseRSI, PlaybookLearning, Evolve, Step
from fin_skills.rsi.meter import Budget, FitBudget
from fin_skills.rsi.bench import benchmark
from fin_skills.tasks.judge import Visible


class Stop(BaseException):
    pass


class Backend:
    def __init__(self):
        self.calls = []
        self.fail = False

    def describe(self):
        return dict(model_id='fixture', model_revision='a'*40, tokenizer_revision='a'*40,
                    precision='torch.bfloat16', context_tokens=32768, max_new_tokens=2048,
                    temperature=.1, top_p=1.)

    def generate_bounded(self, messages, *, max_new_tokens, seed):
        self.calls.append((messages, max_new_tokens, seed))
        if self.fail:
            raise Stop()
        return dict(choices=[dict(message=dict(content='output = 3'))],
                    usage=dict(completion_tokens=3, prompt_tokens=7))


@dataclass
class Task:
    id: str
    algorithm: str = 'heldout'

    def manifest(self):
        return dict(id=self.id, algorithm=self.algorithm)

    @property
    def sha256(self):
        from fin_skills.rsi.run_record import digest as task_digest
        return task_digest(self.manifest())


class Judge:
    executions = 0

    def __init__(self, meter):
        self.meter = meter

    def check(self, code):
        self.meter.charge_attempt()
        Judge.executions += 1
        return Visible(code, True, checks={'ok': True}, attempt_index=0)

    def grade_attempt(self, index):
        assert index == 0
        return True


class Method(BaseRSI):
    name = 'fixture'
    fits = 0
    solves = 0
    stop_after_response = False
    stop_before_solve = None

    def fit(self, *args, **kwargs):
        Method.fits += 1
        return super().fit(*args, **kwargs)

    def make_judge(self, task, *, seed, meter):
        return Judge(meter)

    def _solve(self, meter, judge, task, seed, episode):
        Method.solves += 1
        if Method.stop_before_solve == task.id:
            raise Stop()
        text = meter([dict(role='user', content=task.id)])
        if Method.stop_after_response:
            raise Stop()
        seen = judge.check(text)
        episode.steps.append(Step(1, text, seen.executed, seen.accepted, dict(seen.checks),
                                  attempt_index=seen.attempt_index))
        episode.status = 'accepted'


@pytest.fixture
def setup(tmp_path):
    Method.fits = Method.solves = Judge.executions = 0
    Method.stop_after_response = False
    Method.stop_before_solve = None
    binding = dict(plan='frozen-plan', source='qualified-source', model='fixture')
    store = OperationStore(tmp_path/'store', binding, create=True)
    raw = Backend()
    return store, raw, ReplayBackend(raw, store)


def run(tmp_path, store, backend, name):
    return benchmark([checkpointed_method(Method, store)], backend,
        SimpleNamespace(dev=(), heldout=(Task('one'), Task('two'))),
        seeds=(101,), library=(False,), budget=Budget(attempts=1, calls=1, generated_tokens=10),
        out=tmp_path/(name+'.jsonl'), model_id='fixture', execution_mode='trusted-fixtures')


def test_response_committed_before_algorithm_return_is_replayed_once(tmp_path, setup):
    store, raw, backend = setup
    Method.stop_after_response = True
    with pytest.raises(Stop):
        run(tmp_path, store, backend, 'parent')
    old_bytes = {p: p.read_bytes() for p in tmp_path.glob('parent.jsonl*') if p.is_file()}
    assert len(raw.calls) == Method.fits == 1
    Method.stop_after_response = False
    report = run(tmp_path, store, backend, 'child')
    assert report.complete and len(report.rows) == 2
    assert len(raw.calls) == 2 and Method.fits == 1
    assert all(p.read_bytes() == data for p, data in old_bytes.items())
    assert store.physical_accounting() == dict(physical_requests=2, known_responses=2,
        known_generated_tokens=6, unknown_cost_requests=0, unknown_reserved_output_tokens=0)
    assert sum(row['usage']['generated_tokens'] for row in report.rows) == 6


def test_completed_episode_is_not_solved_or_executed_again(tmp_path, setup):
    store, raw, backend = setup
    Method.stop_before_solve = 'two'
    with pytest.raises(Stop):
        run(tmp_path, store, backend, 'parent')
    assert Judge.executions == len(raw.calls) == 1
    Method.stop_before_solve = None
    report = run(tmp_path, store, backend, 'child')
    assert report.complete and Judge.executions == len(raw.calls) == 2
    assert Method.fits == 1 and Method.solves == 3  # one, interrupted two, resumed two
    report._validate()


def test_unresolved_generation_remains_unknown_and_never_reissues(tmp_path, setup):
    store, raw, backend = setup
    raw.fail = True
    with pytest.raises(Stop):
        run(tmp_path, store, backend, 'parent')
    raw.fail = False
    with pytest.raises(UnresolvedOperation):
        run(tmp_path, store, backend, 'child')
    assert len(raw.calls) == 1
    assert store.physical_accounting()['unknown_cost_requests'] == 1
    assert store.physical_accounting()['unknown_reserved_output_tokens'] == 10


def test_checkpoint_commit_survives_crash_before_benchmark_row(tmp_path, setup, monkeypatch):
    from fin_skills.rsi.run_record import RunRecord
    store, raw, backend = setup
    original = RunRecord.append
    monkeypatch.setattr(RunRecord, 'append', lambda *a: (_ for _ in ()).throw(Stop()))
    with pytest.raises(Stop):
        run(tmp_path, store, backend, 'parent')
    assert len(raw.calls) == Judge.executions == 1
    monkeypatch.setattr(RunRecord, 'append', original)
    report = run(tmp_path, store, backend, 'child')
    assert report.complete and len(raw.calls) == Judge.executions == 2


@pytest.mark.parametrize('change', ['seed', 'messages', 'max_new_tokens'])
def test_changed_request_is_rejected_before_provider(setup, change):
    store, raw, backend = setup
    request = dict(messages=[dict(role='user', content='one')], seed=101, max_new_tokens=10)
    with backend.phase(['episode', 'fixed']):
        backend.generate_bounded(**request)
    request[change] = ([dict(role='user', content='different')] if change == 'messages' else 9)
    with backend.phase(['episode', 'fixed']), pytest.raises(ValueError, match='identity changed'):
        backend.generate_bounded(**request)
    assert len(raw.calls) == 1


def test_frozen_binding_and_checkpoint_value_cannot_change(tmp_path, setup):
    store, _, _ = setup
    with pytest.raises(ValueError, match='binding changed'):
        OperationStore(store.root, dict(plan='other'))
    store.completed(['fit', 'x'], {}, dict(state={}))
    with pytest.raises(ValueError, match='must not be replaced'):
        store.completed(['fit', 'x'], {}, dict(state={'changed': True}))


@pytest.mark.parametrize('method,state', [
    (PlaybookLearning, dict(playbook='frozen instructions')),
    (Evolve, dict(addendum='frozen instructions'))])
def test_completed_learning_state_has_explicit_restore(method, state):
    restored = method(library=True)
    usage = dict(calls=21, generated_tokens=300, episodes=4)
    restore_fit(restored, dict(state=state, usage=usage))
    assert restored.fitted and restored.state() == state and restored.fit_usage == usage


def test_receipt_damage_fails_closed(setup):
    store, raw, backend = setup
    with backend.phase(['episode', 'one']):
        backend.generate_bounded([], max_new_tokens=10, seed=1)
    key = ['provider', ['episode', 'one'], 0]
    path = store.path(key)/'complete.json'
    value = json.loads(path.read_bytes())
    value['value']['response']['usage']['completion_tokens'] = 0
    path.write_text(json.dumps(value))
    with backend.phase(['episode', 'one']), pytest.raises(ValueError, match='receipt identity'):
        backend.generate_bounded([], max_new_tokens=10, seed=1)
    assert len(raw.calls) == 1


def test_fsynced_response_before_meter_receipt_needs_no_second_generation(tmp_path, setup, monkeypatch):
    import fin_skills.rsi.meter as meter
    from scripts.seven_study import partial_accounting
    store, raw, backend = setup
    original = meter.event
    def interrupt(kind, **payload):
        if kind == 'continuation_provider_reference':
            raise Stop()
        original(kind, **payload)
    monkeypatch.setattr(meter, 'event', interrupt)
    with pytest.raises(Stop):
        run(tmp_path, store, backend, 'parent')
    assert partial_accounting(tmp_path/'parent.jsonl.events.jsonl')['unknown_cost_requests'] == 1
    assert store.physical_accounting()['unknown_cost_requests'] == 0
    monkeypatch.setattr(meter, 'event', original)
    assert run(tmp_path, store, backend, 'child').complete
    assert len(raw.calls) == 2
    # The separate recovery proof does not rewrite the parent's original unknown ledger.
    assert partial_accounting(tmp_path/'parent.jsonl.events.jsonl')['unknown_cost_requests'] == 1


def test_fsynced_checkpoint_before_publication_is_explicitly_recoverable(setup, monkeypatch):
    import scripts.study_continuation as continuation
    import hashlib
    store, _, _ = setup
    link = continuation.os.link
    def interrupt(source, target):
        if target.name == 'complete.json':
            raise Stop()
        return link(source, target)
    monkeypatch.setattr(continuation.os, 'link', interrupt)
    key, request, value = ['fit', 'fixed'], {}, dict(state={}, usage={}, events=[])
    with pytest.raises(Stop):
        store.completed(key, request, value)
    monkeypatch.setattr(continuation.os, 'link', link)
    pending, = store.path(key).glob('complete.json.*.pending')
    body = pending.read_bytes()
    with pytest.raises(UnresolvedOperation):
        store.get(key, request)
    with pytest.raises(UnresolvedOperation):
        store.get_checkpoint(key, request)
    store.recover_commit(key, request, pending, hashlib.sha256(body).hexdigest())
    assert store.get(key, request)['value'] == value and pending.read_bytes() == body
    with pytest.raises(FileExistsError):
        store.recover_commit(key, request, pending, hashlib.sha256(body).hexdigest())


@pytest.mark.parametrize('checkpoint_kind', ['fit', 'episode'])
def test_checkpoint_intent_without_value_reconstructs_only_deterministic_commit(
        tmp_path, setup, monkeypatch, checkpoint_kind):
    import scripts.study_continuation as continuation
    store, raw, backend = setup
    original = continuation.atomic_new
    interrupted = []
    def stop(path, value):
        if path.name == 'complete.json' and not interrupted:
            intent = json.loads((path.parent/'intent.json').read_bytes())
            if intent['key'][0] == checkpoint_kind:
                interrupted.append(path.parent)
                raise Stop()
        return original(path, value)
    monkeypatch.setattr(continuation, 'atomic_new', stop)
    with pytest.raises(Stop):
        run(tmp_path, store, backend, 'parent')
    assert len(interrupted) == 1
    path = interrupted[0]
    assert not list(path.glob('complete.json*'))
    old_intent = (path/'intent.json').read_bytes()
    count_before = len(raw.calls)
    monkeypatch.setattr(continuation, 'atomic_new', original)
    report = run(tmp_path, store, backend, 'child')
    assert report.complete and len(raw.calls) == 2
    assert (path/'intent.json').read_bytes() == old_intent
    assert (path/'complete.json').is_file()
    assert store.physical_accounting()['unknown_cost_requests'] == 0
    assert count_before == (1 if checkpoint_kind == 'episode' else 0)
    with pytest.raises(ValueError, match='only fit/episode'):
        store.get_checkpoint(['provider', 'one'], {})


def test_original_judge_replays_program_without_reexecution_or_early_grade(tmp_path, setup, monkeypatch):
    from fin_skills.tasks import suite
    from fin_skills.tasks import judge
    from fin_skills.rsi.methods import NoImprove
    store, raw, backend = setup
    executions, grades = [], []
    def execute(*args):
        executions.append(args)
        return dict(executed=False, error=dict(type='Fixture', message='fixed failure'))
    monkeypatch.setattr(judge, '_run_worker', execute)
    original_grade = judge.Judge.grade_attempt
    def grade(self, index):
        grades.append(index)
        return original_grade(self, index)
    monkeypatch.setattr(judge.Judge, 'grade_attempt', grade)
    class Interrupted(NoImprove):
        interrupt = True
        def _solve(self, *args):
            super()._solve(*args)
            if self.interrupt:
                raise Stop()
    cls = checkpointed_method(Interrupted, store)
    task = suite().heldout[0]
    first = cls()
    first.fit(backend, (), budget=FitBudget(), seeds=(11,), episode_budget=Budget())
    with pytest.raises(Stop):
        first.solve(backend, task, seed=101, budget=Budget())
    assert len(raw.calls) == len(executions) == 1 and grades == []
    assert judge._run_worker is execute  # Scoped hooks are restored after BaseException.
    Interrupted.interrupt = False
    second = cls()
    second.fit(backend, (), budget=FitBudget(), seeds=(11,), episode_budget=Budget())
    result = second.solve(backend, task, seed=101, budget=Budget())
    assert len(raw.calls) == len(executions) == 1
    assert result.usage['calls'] == result.usage['attempts'] == 1
    assert grades == [0] and result._judge.grade_attempt(0) is False


def test_original_external_entrypoint_and_context_guard_reuse_completed_cells(tmp_path, setup, monkeypatch):
    from benchmarks.agent_study import external_study
    from benchmarks.agent_study.external_packets import ScalarPolicy, packets
    from scripts.study_context import ContextGuardedBackend, guard_accounting
    from scripts.study_continuation import run_checkpointed
    from scripts.seven_study import partial_accounting
    store, raw, replay = setup
    public, scoring = packets(dict(task='CodeFinQA', question='How many?', context='3 items',
        context_type='text', options=[], answer='3'), revision='a'*40, split='test', row_index=0,
        source_cluster='fixture', policy=ScalarPolicy('Use items.', 0, 0, 'fixture'))
    tasks = [external_study.ScalarTask(public, scoring)]
    executed = []
    def scalar(code, **kwargs):
        executed.append(code)
        return dict(executed=True, value=3)
    monkeypatch.setattr(external_study, 'evaluate_scalar', scalar)
    original = external_study.benchmark
    def invoke(name):
        backend = ContextGuardedBackend(replay, dict(native=dict(effective_tokens=32768)),
                                        count_tokens=lambda _: 8)
        return run_checkpointed(external_study.run, backend, store, tasks,
            out=tmp_path/(name+'.jsonl'), model_id='fixture', seeds=(101,),
            execution_mode='trusted-fixtures')
    first, second = invoke('first'), invoke('second')
    assert all(row['valid'] for row in second.rows)
    assert len(raw.calls) == len(executed) == 2
    assert first.rows == second.rows
    assert external_study.benchmark is original
    trace = tmp_path/'second.jsonl.events.jsonl'
    logical = partial_accounting(trace)
    guard = guard_accounting(trace, effective_tokens=32768)
    assert logical['started_requests'] == guard['checked_requests'] == 2
    assert logical['unknown_cost_requests'] == guard['missing_checks'] == 0
    assert store.physical_accounting()['physical_requests'] == 2


def test_completed_prefix_and_fresh_tail_keep_unique_context_indices(tmp_path, setup, monkeypatch):
    from benchmarks.agent_study import external_study
    from benchmarks.agent_study.external_packets import ScalarPolicy, packets
    from fin_skills.rsi.run_record import RunRecord
    from scripts.study_context import ContextGuardedBackend, guard_accounting
    from scripts.study_continuation import run_checkpointed
    from scripts.seven_study import partial_accounting
    store, raw, replay = setup
    tasks = []
    for index in range(2):
        public, scoring = packets(dict(task='CodeFinQA', question=f'How many {index}?',
            context='3 items', context_type='text', options=[], answer='3'),
            revision='a'*40, split='test', row_index=index, source_cluster='fixture',
            policy=ScalarPolicy('Use items.', 0, 0, 'fixture'))
        tasks.append(external_study.ScalarTask(public, scoring))
    executions = []
    def scalar(code, **kwargs):
        executions.append(code)
        return dict(executed=True, value=3)
    monkeypatch.setattr(external_study, 'evaluate_scalar', scalar)
    original_append = RunRecord.append
    stop_at = [1]
    def append(record, row):
        original_append(record, row)
        if record.rows == stop_at[0]:
            raise Stop()
    monkeypatch.setattr(RunRecord, 'append', append)
    def invoke(name):
        guard = ContextGuardedBackend(replay, dict(native=dict(effective_tokens=32768)),
                                      count_tokens=lambda _: 8)
        return run_checkpointed(external_study.run, guard, store, tasks,
            out=tmp_path/(name+'.jsonl'), model_id='fixture', seeds=(101,),
            execution_mode='trusted-fixtures')
    for name, stop in [('parent', 1), ('child', 3)]:
        stop_at[0] = stop
        with pytest.raises(Stop):
            invoke(name)
        trace = tmp_path/(name+'.jsonl.events.jsonl')
        assert guard_accounting(trace, effective_tokens=32768)['checked_requests'] == stop
    parents = {p: p.read_bytes() for pattern in ('parent.jsonl*', 'child.jsonl*')
               for p in tmp_path.glob(pattern) if p.is_file()}
    stop_at[0] = None
    report = invoke('final')
    assert report.complete and len(report.rows) == len(raw.calls) == len(executions) == 4
    trace = tmp_path/'final.jsonl.events.jsonl'
    guard = guard_accounting(trace, effective_tokens=32768)
    logical = partial_accounting(trace)
    assert guard['checked_requests'] == logical['started_requests'] == 4
    assert guard['missing_checks'] == logical['unknown_cost_requests'] == 0
    checks = [json.loads(line) for line in trace.read_text().splitlines()
              if json.loads(line)['kind'] == 'context_checked']
    assert [c['request_index'] for c in checks] == list(range(4))
    assert sum(c.get('physical_replay', False) for c in checks) == 3
    assert all('continuation_original_event_sha256' in c for c in checks[:3])
    assert store.physical_accounting()['physical_requests'] == 4
    assert all(p.read_bytes() == body for p, body in parents.items())


def test_original_external_checkpoint_gap_reuses_provider_and_program(tmp_path, setup, monkeypatch):
    import scripts.study_continuation as continuation
    from benchmarks.agent_study import external_study
    from benchmarks.agent_study.external_packets import ScalarPolicy, packets
    store, raw, replay = setup
    public, scoring = packets(dict(task='CodeFinQA', question='How many?', context='3 items',
        context_type='text', options=[], answer='3'), revision='a'*40, split='test', row_index=0,
        source_cluster='fixture', policy=ScalarPolicy('Use items.', 0, 0, 'fixture'))
    tasks = [external_study.ScalarTask(public, scoring)]
    executions = []
    def scalar(code, **kwargs):
        executions.append(code)
        return dict(executed=True, value=3)
    monkeypatch.setattr(external_study, 'evaluate_scalar', scalar)
    original = continuation.atomic_new
    interrupted = []
    def stop(path, value):
        if path.name == 'complete.json' and not interrupted:
            intent = json.loads((path.parent/'intent.json').read_bytes())
            if intent['key'][0] == 'episode':
                interrupted.append(path.parent)
                raise Stop()
        return original(path, value)
    def invoke(name):
        return continuation.run_checkpointed(external_study.run, replay, store, tasks,
            out=tmp_path/(name+'.jsonl'), model_id='fixture', seeds=(101,),
            execution_mode='trusted-fixtures')
    monkeypatch.setattr(continuation, 'atomic_new', stop)
    with pytest.raises(Stop):
        invoke('parent')
    assert len(raw.calls) == len(executions) == 1
    assert not list(interrupted[0].glob('complete.json*'))
    monkeypatch.setattr(continuation, 'atomic_new', original)
    report = invoke('child')
    assert report.complete and len(raw.calls) == len(executions) == len(report.rows) == 2
    assert store.physical_accounting()['physical_requests'] == 2
