"""Durable operation replay for a separately released study continuation.

This is an execution component, not permission to resume an allocation. An owner must
verify the previous writer has stopped, transfer the exclusive claim, bind the parent
receipts and cumulative allocation budget, and qualify this adapter before deployment.
An incomplete provider response is deliberately not retried.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import uuid

CURRENT_CALL = ContextVar('continuation_meter_call', default=None)


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


class UnresolvedOperation(RuntimeError):
    """A previous physical operation started but has no durable complete response."""


def atomic_new(path, value):
    """Publish one complete immutable record; no separately committed row index."""
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.pending')
    data = canonical(value)
    with temporary.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    # link never replaces a concurrent owner's record. A crash before linking leaves
    # the complete pending artifact for explicit recovery, not a half published JSON.
    os.link(temporary, path)
    temporary.unlink()
    if os.name == 'posix':
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


class OperationStore:
    """One frozen namespace; callers retain the external Slurm/claim ownership guard."""
    def __init__(self, root, binding, *, create=False):
        self.root = Path(root)
        self.binding = deepcopy(binding)
        if create:
            self.root.mkdir(parents=True, exist_ok=False)
            atomic_new(self.root/'binding.json', binding)
        if digest(json.loads((self.root/'binding.json').read_bytes())) != digest(binding):
            raise ValueError('continuation binding changed')

    def path(self, key):
        return self.root/digest(key)

    def get(self, key, request):
        path = self.path(key)
        if not path.exists():
            return None
        intent = json.loads((path/'intent.json').read_bytes())
        if digest(intent) != digest(dict(key=key, request=request, binding_sha256=digest(self.binding))):
            raise ValueError('continuation operation identity changed')
        complete = path/'complete.json'
        if not complete.exists():
            raise UnresolvedOperation('durable intent without complete response: ' + str(path))
        receipt = json.loads(complete.read_bytes())
        if receipt['intent_sha256'] != digest(intent) or receipt['value_sha256'] != digest(receipt['value']):
            raise ValueError('continuation receipt identity changed')
        return deepcopy(receipt)

    def execute(self, key, request, operation):
        receipt = self.get(key, request)
        if receipt is not None:
            return receipt['value'], True
        path = self.path(key)
        # An exclusive directory arbitrates writers before any provider request.
        path.mkdir()
        intent = dict(key=key, request=request, binding_sha256=digest(self.binding))
        atomic_new(path/'intent.json', intent)
        value = operation()
        atomic_new(path/'complete.json', dict(intent_sha256=digest(intent),
                   value_sha256=digest(value), value=value))
        return deepcopy(value), False

    def completed(self, key, request, value):
        """Commit a deterministic fit/episode checkpoint after its operations finish."""
        existing = self.get_checkpoint(key, request)
        path = self.path(key)
        if existing is None and path.exists():
            # The checkpoint intent was committed, but writing its value never
            # started. Reconstructing control flow used the separate exact provider
            # and program receipts; this commit itself has no external side effect.
            intent = json.loads((path/'intent.json').read_bytes())
            atomic_new(path/'complete.json', dict(intent_sha256=digest(intent),
                       value_sha256=digest(value), value=value))
            return deepcopy(value), False
        result, replayed = self.execute(key, request, lambda: value)
        if digest(result) != digest(value):
            raise ValueError('committed fit/episode must not be replaced')
        return result, replayed

    def get_checkpoint(self, key, request):
        """Only deterministic fit/episode commits may reconstruct an intent-only gap.

        Provider and program operations remain fail-closed. If complete bytes were
        already written, explicit recover_commit must preserve and promote them.
        The caller must still hold exclusive allocation ownership.
        """
        if not key or key[0] not in ('fit', 'episode'):
            raise ValueError('only fit/episode checkpoints are reconstructible')
        try:
            return self.get(key, request)
        except UnresolvedOperation:
            # get checked the exact request, binding and intent before raising.
            if any(self.path(key).glob('complete.json.*.pending')):
                raise
            return None

    def recover_commit(self, key, request, pending, expected_sha256):
        """Promote exact fsynced bytes after the caller has established sole ownership.

        This closes a crash between writing a complete record and its atomic publication.
        It cannot recover a generation that never produced a complete response.
        """
        path, pending = self.path(key), Path(pending)
        if (pending.resolve().parent != path.resolve()
                or not pending.name.startswith('complete.json.')
                or not pending.name.endswith('.pending')):
            raise ValueError('pending record is outside the exact operation')
        data = pending.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected_sha256:
            raise ValueError('pending record bytes changed')
        intent = json.loads((path/'intent.json').read_bytes())
        expected = dict(key=key, request=request, binding_sha256=digest(self.binding))
        receipt = json.loads(data)
        if (digest(intent) != digest(expected) or receipt['intent_sha256'] != digest(intent)
                or receipt['value_sha256'] != digest(receipt['value'])):
            raise ValueError('pending record identity changed')
        os.link(pending, path/'complete.json')
        # Keep the pending artifact as original recovery evidence.
        return dict(operation=digest(key), recovered_sha256=expected_sha256,
                    original_pending=str(pending))

    def physical_accounting(self):
        requests = known = tokens = unknown = reserved_tokens = 0
        for path in self.root.iterdir():
            if not path.is_dir():
                continue
            intent_path = path/'intent.json'
            if not intent_path.exists():
                raise UnresolvedOperation('incomplete operation intent: ' + str(path))
            intent = json.loads(intent_path.read_bytes())
            if intent['key'][0] != 'provider':
                continue
            requests += 1
            complete = path/'complete.json'
            if not complete.exists():
                unknown += 1
                reserved_tokens += intent['request']['max_new_tokens']
                continue
            receipt = self.get(intent['key'], intent['request'])
            response = receipt['value']['response']
            usage = response.get('usage', {}) if isinstance(response, dict) else {}
            count = usage.get('completion_tokens')
            if type(count) is not int or count < 0:
                unknown += 1
                reserved_tokens += intent['request']['max_new_tokens']
            else:
                known += 1
                tokens += count
        return dict(physical_requests=requests, known_responses=known,
                    known_generated_tokens=tokens, unknown_cost_requests=unknown,
                    unknown_reserved_output_tokens=reserved_tokens)


class ReplayBackend:
    """Replay complete provider receipts with exact requests and original logical usage."""
    def __init__(self, backend, store):
        self.backend, self.store = backend, store
        self.stream, self.ordinal = None, 0

    @property
    def chat(self):
        return self.backend.chat

    def describe(self):
        return self.backend.describe()

    def __call__(self, messages):
        raise RuntimeError('explicit bounded calls are required')

    @contextmanager
    def phase(self, identity):
        previous = self.stream, self.ordinal
        call_token = CURRENT_CALL.set(None)
        self.stream, self.ordinal = deepcopy(identity), 0
        try:
            yield
        finally:
            self.stream, self.ordinal = previous
            CURRENT_CALL.reset(call_token)

    def generate_bounded(self, messages, *, max_new_tokens, seed):
        from fin_skills.rsi.meter import event
        if self.stream is None:
            raise ValueError('provider call has no frozen fit/episode stream')
        key = ['provider', self.stream, self.ordinal]
        request = dict(messages=messages, max_new_tokens=max_new_tokens, seed=seed,
                       backend={k: v for k, v in self.describe().items() if k != 'hardware'},
                       meter_call=CURRENT_CALL.get())
        def actual():
            response = self.backend.generate_bounded(messages, max_new_tokens=max_new_tokens, seed=seed)
            # Do not translate exceptions, refusals or invalid usage into a success.
            # Complete provider responses, including refusals, are replayed verbatim.
            return dict(response=response)
        value, replayed = self.store.execute(key, request, actual)
        event('continuation_provider_reference', operation=digest(key),
              disposition='replay' if replayed else 'new', binding_sha256=digest(self.store.binding),
              response_sha256=digest(value['response']))
        self.ordinal += 1
        return value['response']


@contextmanager
def captured_events():
    from fin_skills.rsi.meter import EVENT_SINK
    sink, events = EVENT_SINK.get(), []
    def capture(entry):
        events.append(deepcopy(entry))
        if entry.get('kind') == 'call_started' and entry.get('backend_mode') == 'bounded':
            CURRENT_CALL.set({k: entry[k] for k in ('role', 'call_index', 'seed', 'max_new_tokens')})
        if sink is not None:
            sink(entry)
    token = EVENT_SINK.set(capture)
    try:
        yield events
    finally:
        EVENT_SINK.reset(token)


@contextmanager
def execution_replay(store, stream):
    """Reconstruct the original judge from exact stored worker results, without grading early.

    The qualified judge still charges attempts, builds Visible, selects an attempt and
    grades after solve returns. Only its isolated program execution boundary is cached.
    Hooks are scoped to this serial owned worker and restored even on interruption.
    """
    from fin_skills.tasks import judge
    from benchmarks.agent_study import external_study
    worker, scalar = judge._run_worker, external_study.evaluate_scalar
    ordinal = 0
    def invoke(kind, request, operation):
        nonlocal ordinal
        key = ['program', stream, ordinal]
        result, _ = store.execute(key, dict(kind=kind, **request), operation)
        ordinal += 1
        return result
    def cached_worker(code, request, timeout):
        return invoke('positions', dict(code=code, request=request, timeout=timeout),
                      lambda: worker(code, request, timeout))
    def cached_scalar(code, **kwargs):
        return invoke('scalar', dict(code=code, kwargs=kwargs), lambda: scalar(code, **kwargs))
    judge._run_worker, external_study.evaluate_scalar = cached_worker, cached_scalar
    try:
        yield
    finally:
        judge._run_worker, external_study.evaluate_scalar = worker, scalar


def replay_events(events, checkpoint, generate):
    # A child has a fresh guard and its own logical request sequence. Reusing the
    # parent's indices collides with the first newly generated tail request.
    # The guard module is imported as study_context (coordinator layout) or as
    # scripts.study_context (repository layout); either may already be loaded, so match the
    # class by name and module rather than by one module's class object.
    def is_guard(obj):
        cls = type(obj)
        return (cls.__name__ == 'ContextGuardedBackend'
                and cls.__module__ in ('study_context', 'scripts.study_context'))
    current, guard, seen = generate, None, set()
    while not isinstance(current, ReplayBackend):
        if id(current) in seen or not hasattr(current, 'backend'):
            raise TypeError('checkpoint replay requires the bound backend chain')
        seen.add(id(current))
        if is_guard(current):
            if guard is not None:
                raise ValueError('multiple context guards in checkpoint replay')
            guard = current
        current = current.backend
    from fin_skills.rsi.meter import EVENT_SINK
    sink = EVENT_SINK.get()
    for entry in events:
        replayed = dict(entry, continuation_checkpoint=checkpoint, physical_replay=True,
                        continuation_original_event_sha256=digest(entry))
        if entry.get('kind') == 'context_checked':
            if (guard is None or entry['effective_context_tokens'] !=
                    guard.profile['native']['effective_tokens']):
                raise ValueError('checkpoint context guard differs from the current contract')
            replayed.update(request_index=guard.requests,
                            continuation_original_request_index=entry['request_index'])
            guard.requests += 1
        if sink is not None:
            sink(replayed)


def restore_fit(method, value):
    state = value['state']
    if method.name == 'playbook':
        if set(state) != {'playbook'}:
            raise ValueError('invalid playbook checkpoint')
        method.addendum = state['playbook']
    elif method.name == 'evolve':
        if set(state) != {'addendum'}:
            raise ValueError('invalid evolve checkpoint')
        method.addendum = state['addendum']
    elif method.learns:
        raise ValueError('learned method has no qualified state restore')
    method.fit_usage = deepcopy(value['usage'])
    method.fitted = True
    if method.state() != state:
        raise ValueError('restored method state differs from frozen fit')


class CommittedJudge:
    def __init__(self, grades):
        self.grades = grades

    def grade_attempt(self, index):
        return self.grades[str(index)]


def replay_owner(backend):
    """A context guard may remain outermost, including on replayed provider calls."""
    seen = set()
    while not isinstance(backend, ReplayBackend):
        if id(backend) in seen or not hasattr(backend, 'backend'):
            raise TypeError('checkpointed methods require the durable replay backend')
        seen.add(id(backend))
        backend = backend.backend
    return backend


def checkpointed_method(cls, store):
    """Wrap fit/solve without changing a method's prompts, decisions or grading.

    An unfinished fit/episode reconstructs its control flow by strict provider replay.
    Completed episodes and fits skip the original implementation entirely. The replay
    trace is logical accounting; OperationStore is the distinct physical provider ledger.
    """
    class Checkpointed(cls):
        def fit(self, generate, dev_tasks, *, budget, seeds, episode_budget):
            owner = replay_owner(generate)
            key = ['fit', self.name, self.library]
            request = dict(tasks=[t.manifest() for t in dev_tasks], budget=vars(budget),
                           seeds=list(seeds), episode_budget=episode_budget.as_dict())
            existing = store.get_checkpoint(key, request)
            if existing is not None:
                restore_fit(self, existing['value'])
                replay_events(existing['value']['events'], digest(key), generate)
                return self
            with owner.phase(key), captured_events() as events, execution_replay(store, key):
                super().fit(generate, dev_tasks, budget=budget, seeds=seeds,
                            episode_budget=episode_budget)
            value = dict(state=self.state(), usage=deepcopy(self.fit_usage), events=events)
            store.completed(key, request, value)
            return self

        def solve(self, generate, task, *, seed, budget):
            from fin_skills.rsi.methods import Episode, Step
            owner = replay_owner(generate)
            key = ['episode', self.name, self.library, task.id, seed]
            request = dict(task=task.manifest(), budget=budget.as_dict(), state=self.state())
            existing = store.get_checkpoint(key, request)
            if existing is not None:
                value = existing['value']
                data = deepcopy(value['episode'])
                data['steps'] = [Step(**s) for s in data['steps']]
                episode = Episode(**data)
                episode._judge = CommittedJudge(value['grades'])
                replay_events(value['events'], digest(key), generate)
                return episode
            with owner.phase(key), captured_events() as events, execution_replay(store, key):
                episode = super().solve(generate, task, seed=seed, budget=budget)
            # Hidden grading occurs only after solve has returned; it cannot affect
            # method selection or stopping. Only the original selected attempts count.
            indices = {s.attempt_index for s in episode.steps[:1] + episode.steps[-1:]}
            grades = {str(i): episode._judge.grade_attempt(i) for i in indices}
            store.completed(key, request, dict(episode=asdict(episode), grades=grades, events=events))
            return episode
    return Checkpointed


def run_checkpointed(original_run, generate, store, *args, **kwargs):
    """Use the original P/E entry point with explicit continuation provenance.

    No qualified source file is changed. A separately qualified allocation coordinator
    must bind this adapter, the store and its physical ledger in addition to the original
    complete report; the original report alone describes logical algorithm accounting.
    """
    from fin_skills.rsi import study
    from benchmarks.agent_study import external_study
    modules = {study.run: study, external_study.run: external_study}
    if original_run not in modules:
        raise ValueError('only the original declared P/E entry points are supported')
    if replay_owner(generate).store is not store:
        raise ValueError('backend and method checkpoint stores differ')
    module = modules[original_run]
    original_benchmark = module.benchmark
    def wrapped(methods, backend, *positional, **options):
        if backend is not generate:
            raise ValueError('study changed the bound backend')
        manifest = deepcopy(options.get('study_manifest') or {})
        manifest['continuation'] = dict(binding_sha256=digest(store.binding),
            adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            accounting='report is logical; durable operation store is physical',
            arbitrary_inflight_resume=False)
        options['study_manifest'] = manifest
        return original_benchmark([checkpointed_method(cls, store) for cls in methods],
                                  backend, *positional, **options)
    module.benchmark = wrapped
    try:
        return original_run(generate, *args, **kwargs)
    finally:
        module.benchmark = original_benchmark
