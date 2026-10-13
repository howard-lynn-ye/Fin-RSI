"""Slurm-side native CLI transport. Models return text; Beacon executes study tools.

All exchanges remain on Beacon. The desktop relay transports messages in memory,
never chooses a portfolio, and may not substitute a model or use native tools.
"""
from functools import lru_cache
import json
import re
import os
from pathlib import Path
import time

from benchmarks.agent_study.personal_chat import fingerprint, write_new
from benchmarks.agent_study.model_matrix import matrix
from benchmarks.agent_study import trading_study as old

REVISION = 'native-cli-20261013-v1-unpinned-provider'
QUEUE = Path('/beacon-projects/radfm/wy891/fin-multisource-models-20261005/native-queue-20261013-v1')
SUPPORTED = ('claude_subscription', 'antigravity_subscription', 'codex_subscription')


def policy(model):
    row, = [r for r in matrix() if r['model'] == model and r['transport'] in SUPPORTED]
    from benchmarks.agent_study.native_relay import WRAPPER, source_digest
    return dict(transport=row['transport'], model=model, effort=row['effort'], revision=REVISION,
        prompt_wrapper=WRAPPER, relay_sha256=source_digest(),
        final_output={'transport': 'text'}, queue=str(QUEUE),
        reference_tokenizer=list(old.MODELS[0][1:]),
        output_budget='1024 Qwen2.5-Coder-7B reference tokens; oversize text is rejected, never trimmed',
        reasoning_budget='provider CLI effort; hidden reasoning, system instructions and host global rules not controlled',
        repeat_kind='independent stateless invocations; repeat index is not a sampling seed',
        model_identity=('configured CLI model only; response events do not independently attest server model'
                        if row['transport'] == 'codex_subscription' else
                        'client init model must match request; not independent server model attestation'
                        if row['transport'] == 'antigravity_subscription' else
                        'native CLI assistant response model must exactly match requested model'),
        tool_policy='no native tool use; any observed native tool event invalidates the response')


@lru_cache(maxsize=1)
def reference_tokenizer(model, revision):
    from benchmarks.agent_study.transformers_chat import load_tokenizer
    return load_tokenizer(model, revision, local_files_only=True)


def episode_index(scope, protocol):
    match = re.fullmatch(r'(access|preflight|cycle)/(raw|library)(?:/(0|[1-9][0-9]*))?', scope or '')
    if not match or ((match[1] == 'access') != (match[3] is None)):
        raise ValueError('invalid native execution scope')
    index = int(match[3] or 0)
    if index >= protocol['decisions_per_arm']:
        raise ValueError('native decision index exceeds frozen schedule')
    return protocol['seed']*1000 + index


def validate_response(response, request):
    spec = request['native_policy']
    if (response.get('request_sha256') != request['request_sha256'] or
            response.get('requested_model') != spec['model'] or
            response.get('transport') != spec['transport'] or
            response.get('requested_effort') != spec['effort'] or
            response.get('success') is not True or response.get('exit_code') != 0 or
            response.get('native_tool_events') != [] or
            response.get('relay_sha256') != spec['relay_sha256'] or
            not isinstance(response.get('text'), str) or not response['text'].strip()):
        raise RuntimeError('native transport/identity/tool failure; preserve exchange, no default decision')
    models = response.get('reported_models')
    if spec['transport'] != 'codex_subscription':
        if not models or any(m != spec['model'] for m in models):
            raise RuntimeError('native response model differs from requested model')
    expected_identity = {'codex_subscription': 'configured_request_only',
                         'antigravity_subscription': 'client_init_model',
                         'claude_subscription': 'response_model'}[spec['transport']]
    if response.get('identity_level') != expected_identity:
        raise RuntimeError('native identity claim exceeds the available response evidence')


class NativeChat:
    def __init__(self, root, protocol, tokenizer, *, timeout=1800):
        if not os.environ.get('SLURM_JOB_ID'):
            raise RuntimeError('native study orchestration requires Slurm')
        self.root, self.protocol, self.tokenizer = Path(root), protocol, tokenizer
        self.spec = policy(protocol['model'][1])
        if protocol.get('native_policy') != self.spec:
            raise ValueError('native policy differs from frozen protocol')
        self.queue = Path(self.spec['queue'])
        self.calls, self.seed, self.timeout, self.request_scope = 0, 0, timeout, None
        self.identity = dict(model=protocol['model'][1], revision=protocol['model'][2],
                             max_tokens=protocol['max_tokens'])

    def __call__(self, messages):
        if self.seed != episode_index(self.request_scope, self.protocol) or not 0 <= self.calls < old.MAX_TURNS:
            raise RuntimeError('native request repeat, decision, or turn differs from frozen schedule')
        payload = dict(protocol_sha256=old.sha(self.root/'protocol.json'), request_scope=self.request_scope,
            native_policy=self.spec, messages=messages, call_index=self.calls,
            episode_index=self.seed, max_response_tokens=self.protocol['max_tokens'])
        digest = fingerprint(payload)
        request = dict(payload, request_sha256=digest)
        folder = self.queue/digest
        request_path, response_path = folder/'request.json', folder/'response.json'
        if request_path.exists():
            if old.load(request_path) != request:
                raise RuntimeError('native queued request changed; no resubmission')
        else:
            write_new(request_path, request)
        print(json.dumps(dict(native_request=digest, model=self.spec['model'])), flush=True)
        deadline = time.monotonic() + self.timeout
        while not response_path.exists():
            if time.monotonic() >= deadline:
                raise RuntimeError('native response absent; preserve request, no silent retry or HOLD')
            time.sleep(1)
        response = old.load(response_path)
        archive = self.root/'native-exchanges'/digest
        archive.mkdir(parents=True, exist_ok=True)
        for path in (request_path, response_path):
            target = archive/path.name
            if not target.exists():
                os.link(path, target)  # Same radfm filesystem; queue removal retains the archive.
            if old.sha(target) != old.sha(path):
                raise RuntimeError('archived native exchange differs from queued original')
        validate_response(response, request)
        receipt = dict(request_sha256=digest, response_sha256=old.sha(response_path))
        accepted = self.root/'native-receipts'/f'{digest}.json'
        if accepted.exists():
            if old.load(accepted) != receipt:
                raise RuntimeError('previously accepted native response changed')
        else:
            write_new(accepted, receipt)
        count = len(self.tokenizer.encode(response['text'], add_special_tokens=False))
        self.calls += 1
        return dict(choices=[dict(message=dict(content=response['text']),
                                 finish_reason='length' if count > self.protocol['max_tokens'] else 'stop')],
            usage=dict(reference_completion_tokens=count, provider_usage=response.get('usage')),
            backend_metadata=dict(identity=self.identity, native_exchange=receipt,
                                  native_identity_level=response.get('identity_level')))


def validate_record(root, record, protocol, scope):
    """Scoring rechecks the actual response and full input binding, not just a model label."""
    root = Path(root)
    spec = policy(protocol['model'][1])
    if protocol.get('native_policy') != spec:
        raise ValueError('native policy changed')
    for index, turn in enumerate(record['turns']):
        item = (turn.get('backend_metadata') or {}).get('native_exchange') or {}
        digest = item.get('request_sha256', '')
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('native exchange receipt absent')
        folder = root/'native-exchanges'/digest
        request, response = old.load(folder/'request.json'), old.load(folder/'response.json')
        payload = {k: v for k, v in request.items() if k != 'request_sha256'}
        if (fingerprint(payload) != digest or request['request_sha256'] != digest or
                request['protocol_sha256'] != old.sha(root/'protocol.json') or
                request['native_policy'] != spec or request['request_scope'] != scope or request['call_index'] != index or
                request['episode_index'] != episode_index(scope, protocol) or
                request['max_response_tokens'] != protocol['max_tokens'] or
                fingerprint(request['messages']) != turn['input_messages_sha256'] or
                old.sha(folder/'response.json') != item.get('response_sha256') or
                old.load(root/'native-receipts'/f'{digest}.json') != item or
                response['text'] != turn['response']):
            raise ValueError('native exchange differs from executed model turn')
        validate_response(response, request)
        tokenizer = reference_tokenizer(*spec['reference_tokenizer'])
        count = len(tokenizer.encode(response['text'], add_special_tokens=False))
        if (turn.get('usage', {}).get('reference_completion_tokens') != count or
                turn['finish_reason'] != ('length' if count > protocol['max_tokens'] else 'stop')):
            raise ValueError('native reference token count or length status differs from original text')


def create_backend(root, protocol):
    tokenizer = reference_tokenizer(*protocol['native_policy']['reference_tokenizer'])
    return NativeChat(root, protocol, tokenizer)
