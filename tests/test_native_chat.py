"""Native provider provenance, stage separation, and no silent model/tool fallback."""
import json
from types import SimpleNamespace

import pytest

from benchmarks.agent_study import native_chat as native, native_relay as relay
from benchmarks.agent_study.personal_chat import fingerprint, write_new


def response(request, **updates):
    spec = request['native_policy']
    return dict(dict(request_sha256=request['request_sha256'], requested_model=spec['model'],
        requested_effort=spec['effort'], transport=spec['transport'], success=True,
        exit_code=0, native_tool_events=[], text='{"tool":"hold","arguments":{}}',
        reported_models=[spec['model']], identity_level='response_model',
        relay_sha256=spec['relay_sha256']), **updates)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    model = 'claude-opus-5-5'
    # Resolve the exact declared id, not an assumed provider spelling.
    model = next(r['model'] for r in native.matrix() if r['id'] == 'opus55')
    spec = native.policy(model)
    spec['queue'] = str(tmp_path/'queue')
    monkeypatch.setattr(native, 'policy', lambda _: spec)
    monkeypatch.setenv('SLURM_JOB_ID', 'test-only')
    protocol = dict(model=['opus', model, native.REVISION], max_tokens=1024,
                    native_policy=spec, batch='matrix_native', seed=11, decisions_per_arm=44)
    write_new(tmp_path/'protocol.json', protocol)
    tokenizer = SimpleNamespace(encode=lambda text, **kw: list(text))
    monkeypatch.setattr(native, 'reference_tokenizer', lambda *args: tokenizer)
    backend = native.NativeChat(tmp_path, protocol, tokenizer, timeout=.01)
    backend.seed = 11000
    return tmp_path, protocol, backend


def respond_immediately(monkeypatch, **updates):
    def publish(path, value):
        write_new(path, value)
        if path.name == 'request.json':
            write_new(path.with_name('response.json'), response(value, **updates))
    monkeypatch.setattr(native, 'write_new', publish)


def test_scopes_keep_identical_preflight_and_cycle_messages_independent(fixture, monkeypatch):
    root, protocol, backend = fixture
    respond_immediately(monkeypatch)
    messages = [{'role': 'user', 'content': 'same visible market'}]
    receipts = []
    for scope in ('access/raw', 'preflight/raw/0', 'cycle/raw/0'):
        backend.calls, backend.request_scope = 0, scope
        result = backend(messages)
        receipts.append(result['backend_metadata']['native_exchange']['request_sha256'])
        record = dict(turns=[dict(backend_metadata=result['backend_metadata'],
            input_messages_sha256=fingerprint(messages), response=result['choices'][0]['message']['content'],
            usage=result['usage'], finish_reason=result['choices'][0]['finish_reason'])])
        native.validate_record(root, record, protocol, scope)
        with pytest.raises(ValueError, match='differs'):
            native.validate_record(root, record, protocol, 'cycle/library/0')
    assert len(set(receipts)) == 3


def test_missing_response_does_not_become_a_hold(fixture):
    _, _, backend = fixture
    backend.request_scope = 'cycle/raw/0'
    with pytest.raises(RuntimeError, match='absent'):
        backend([{'role': 'user', 'content': 'request'}])


@pytest.mark.parametrize('updates', [
    {'reported_models': ['substituted-model']}, {'native_tool_events': [{'tool': 'search_web'}]},
    {'exit_code': 1}, {'success': False}, {'request_sha256': 'changed'},
    {'requested_effort': 'changed'}, {'relay_sha256': 'changed'},
])
def test_transport_substitution_or_tool_use_is_not_a_decision(fixture, monkeypatch, updates):
    _, _, backend = fixture
    respond_immediately(monkeypatch, **updates)
    backend.request_scope = 'cycle/raw/0'
    with pytest.raises(RuntimeError):
        backend([{'role': 'user', 'content': 'request'}])


def test_oversize_text_is_preserved_and_marked_length(fixture, monkeypatch):
    _, _, backend = fixture
    text = 'x'*1025
    respond_immediately(monkeypatch, text=text)
    backend.request_scope = 'cycle/raw/0'
    choice = backend([{'role': 'user', 'content': 'request'}])['choices'][0]
    assert choice['message']['content'] == text and choice['finish_reason'] == 'length'


def test_codex_events_do_not_claim_server_model_attestation():
    events = [dict(type='item.completed', item=dict(type='agent_message', text='{"tool":"hold","arguments":{}}')),
              dict(type='turn.completed', usage={'output_tokens': 12})]
    result = relay.parse_events('codex_subscription', '\n'.join(map(json.dumps, events)), 0)
    assert result['success'] and result['reported_models'] == []
    assert result['identity_level'] == 'configured_request_only'
    events.insert(0, dict(type='item.completed', item=dict(type='command_execution', command='read file')))
    assert not relay.parse_events('codex_subscription', '\n'.join(map(json.dumps, events)), 0)['success']


def test_gemini_native_tool_events_invalidate_otherwise_successful_response():
    events = [dict(event='init', init={'model': 'gemini-3.8-flash-high'}),
              dict(event='step_update', step_update={'step_type': 'tool_call'}),
              dict(event='result', result={'status': 'SUCCESS', 'response': '{}'})]
    assert not relay.parse_events('antigravity_subscription', '\n'.join(map(json.dumps, events)), 0)['success']


def test_source_digest_preserves_identity_across_checkout_line_endings(tmp_path):
    path = tmp_path/'relay.py'
    path.write_bytes(b'first\nsecond\n')
    expected = relay.source_digest(path)
    path.write_bytes(b'first\r\nsecond\r\n')
    assert relay.source_digest(path) == expected
    path.write_bytes(b'changed\nsecond\n')
    assert relay.source_digest(path) != expected


def test_archived_receipt_survives_queue_removal(fixture, monkeypatch):
    root, protocol, backend = fixture
    respond_immediately(monkeypatch)
    backend.request_scope = 'cycle/raw/0'
    messages = [{'role': 'user', 'content': 'request'}]
    result = backend(messages)
    digest = result['backend_metadata']['native_exchange']['request_sha256']
    from pathlib import Path
    queue = Path(protocol['native_policy']['queue'])/digest
    (queue/'request.json').unlink()
    (queue/'response.json').unlink()
    native.validate_record(root, dict(turns=[dict(backend_metadata=result['backend_metadata'],
        input_messages_sha256=fingerprint(messages), response=result['choices'][0]['message']['content'],
            usage=result['usage'], finish_reason=result['choices'][0]['finish_reason'])]),
        protocol, 'cycle/raw/0')

def test_hashing_is_identical_on_both_sides_and_rejects_nan():
    value = {'messages': [{'content': '收益率 Ω "quoted"\ntext'}], 'seed': 11}
    assert relay.fingerprint(value) == fingerprint(value)
    for function in (relay.fingerprint, fingerprint):
        with pytest.raises(ValueError):
            function({'invalid': float('nan')})


def test_repeat_mismatch_is_rejected_before_publishing(fixture):
    root, _, backend = fixture
    backend.request_scope, backend.seed = 'cycle/raw/0', 23000
    with pytest.raises(RuntimeError, match='repeat'):
        backend([{'role': 'user', 'content': 'request'}])
    assert not (root/'queue').exists()


def test_gemini_init_echo_is_not_server_attestation():
    events = [dict(event='init', init={'model': 'gemini-3.8-flash-high'}),
              dict(event='result', result={'status': 'SUCCESS', 'response': '{}'})]
    value = relay.parse_events('antigravity_subscription', '\n'.join(map(json.dumps, events)), 0)
    assert value['success'] and value['identity_level'] == 'client_init_model'


def test_oversize_native_response_cannot_execute_a_trade(fixture, monkeypatch):
    from benchmarks.agent_study import trading_runtime_v8 as runtime
    from test_trading_terminal_v8 import Controller
    _, _, backend = fixture
    respond_immediately(monkeypatch, text='{"tool":"hold","arguments":{}}'+' '*1100)
    backend.request_scope = 'cycle/raw/0'
    record = runtime.decide(backend, Controller('raw'), 'fixture', final_output={'transport': 'text'})
    runtime.validate_decision(record)
    assert record['decision_action'] == 'failed' and record['target'] is None
    assert all(turn['finish_reason'] == 'length' for turn in record['turns'])

def test_observed_codex_startup_notice_is_not_a_model_tool_call():
    notice = dict(type='item.completed', item=dict(type='error',
        message='Code Mode is unavailable because code-mode host is disabled. Code mode will fail closed.'))
    start = dict(type='turn.started')
    answer = dict(type='item.completed', item=dict(type='agent_message', text='{}'))
    done = dict(type='turn.completed')
    parse = lambda rows: relay.parse_events('codex_subscription', '\n'.join(map(json.dumps, rows)), 0)
    result = parse([notice, start, answer, done])
    assert result['success'] and result['diagnostic_events'] == [notice]
    assert result['native_tool_events'] == []
    assert not parse([start, notice, answer, done])['success']
    unknown = dict(type='item.completed', item=dict(type='error', message='unknown provider error'))
    assert not parse([unknown, start, answer, done])['success']