"""Wait for a conversation-authored response; never generate a trading decision.

All request/response files stay in the Slurm run directory. An operator submits one
literal response bound to the displayed request hash. Missing responses stop the run;
there is no model invocation, strategy, default allocation or automatic submission.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False).encode()).hexdigest()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.pending-', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Atomic publish, exclusively; never replace an existing file.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def valid_response(value, request_sha256):
    return (value.get('request_sha256') == request_sha256 and
            value.get('authored_by') == 'assistant_in_current_conversation' and
            value.get('decision_generator') is None and
            isinstance(value.get('text'), str) and bool(value['text'].strip()) and
            isinstance(value.get('rationale'), str) and len(value['rationale'].strip()) >= 20)


def validate_exchange(root, arm, decision, turn_index, item, turn):
    relative = f'personal-exchange/{arm}/{decision:02d}/{turn_index:02d}/request.json'
    if item['request'] != relative:
        raise ValueError('personal receipt belongs to another decision/arm/turn')
    request_path = root / relative
    request, response, accepted = [json.loads(request_path.with_name(name).read_text(encoding='utf-8'))
                                   for name in ('request.json', 'response.json', 'accepted.json')]
    payload = {k: v for k, v in request.items() if k != 'request_sha256'}
    if ((request['arm'], request['decision'], request['turn']) != (arm, decision, turn_index) or
            fingerprint(payload) != item['request_sha256'] or
            request['request_sha256'] != item['request_sha256'] or
            fingerprint(request['messages']) != turn['input_messages_sha256'] or
            not valid_response(response, item['request_sha256']) or
            fingerprint(response) != item['response_sha256'] or accepted != item or
            response['text'] != turn['response']):
        raise ValueError('personal exchange binding changed')
    return relative


def validate_exchange_inventory(root, referenced):
    expected = set(referenced)
    for filename in ('request.json', 'response.json', 'accepted.json'):
        observed = {p.with_name('request.json').relative_to(root).as_posix()
                    for p in (root / 'personal-exchange').rglob(filename)}
        if observed != expected:
            raise ValueError('unreferenced or missing personal exchanges; no silent orphaning')


def submit_response(request_path, text, rationale):
    """Store explicitly supplied text verbatim; parsing/execution belongs to the runtime."""
    request_path = Path(request_path)
    request = json.loads(request_path.read_text(encoding='utf-8'))
    if not isinstance(text, str) or not text.strip():
        raise ValueError('an explicitly authored response is required')
    if not isinstance(rationale, str) or len(rationale.strip()) < 20:
        raise ValueError('record the personal decision rationale')
    response = dict(request_sha256=request['request_sha256'], text=text, rationale=rationale,
                    authored_by='assistant_in_current_conversation', decision_generator=None)
    target = request_path.with_name('response.json')
    write_new(target, response)
    return dict(path=str(target), response_sha256=fingerprint(response))


class PersonalChat:
    def __init__(self, root, tokenizer, *, max_tokens=1024, max_turns=8, timeout=900):
        self.root, self.tokenizer = Path(root), tokenizer
        self.max_tokens, self.timeout = max_tokens, timeout
        self.max_turns = max_turns
        self.calls, self.seed, self.arm, self.decision_index = 0, 0, None, None
        self.receipts = []

    def __call__(self, messages):
        if self.arm not in ('raw', 'library') or type(self.decision_index) is not int:
            raise RuntimeError('explicit decision context is required')
        if not 0 <= self.calls < self.max_turns:
            raise RuntimeError('personal response exceeds the same eight-call limit')
        payload = dict(arm=self.arm, decision=self.decision_index, turn=self.calls,
                       messages=messages, max_response_tokens=self.max_tokens,
                       token_measure='Qwen2.5-Coder-7B reference tokenizer; internal reasoning unmeasured')
        receipt = dict(payload, request_sha256=fingerprint(payload))
        folder = self.root / 'personal-exchange' / self.arm / f'{self.decision_index:02d}' / f'{self.calls:02d}'
        request, response = folder / 'request.json', folder / 'response.json'
        accepted = folder / 'accepted.json'
        if request.exists():
            if json.loads(request.read_text(encoding='utf-8')) != receipt:
                raise RuntimeError('resumed request differs; preserve the old run')
        else:
            write_new(request, receipt)
        print(json.dumps(dict(waiting_for_personal_response=str(request),
                              request_sha256=receipt['request_sha256'])), flush=True)
        deadline = time.monotonic() + self.timeout
        while not response.exists():
            if accepted.exists():
                raise RuntimeError('previously accepted response is missing; do not replace it')
            if time.monotonic() >= deadline:
                raise RuntimeError('personal response absent; stop without a default decision')
            time.sleep(.5)
        value = json.loads(response.read_text(encoding='utf-8'))
        if not valid_response(value, receipt['request_sha256']):
            raise RuntimeError('personal response identity/content does not match the request')
        item = dict(request=str(request.relative_to(self.root).as_posix()),
            request_sha256=receipt['request_sha256'], response_sha256=fingerprint(value))
        if accepted.exists():
            if json.loads(accepted.read_text(encoding='utf-8')) != item:
                raise RuntimeError('previously accepted response changed; do not reinterpret a past turn')
        else:
            write_new(accepted, item)
        count = len(self.tokenizer.encode(value['text'], add_special_tokens=False))
        self.calls += 1
        self.receipts.append(item)
        return dict(choices=[dict(message=dict(content=value['text']),
                                 finish_reason='length' if count > self.max_tokens else 'stop')],
                    usage=dict(reference_completion_tokens=count,
                               assistant_internal_reasoning_tokens=None))
