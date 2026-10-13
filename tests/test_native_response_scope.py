"""The native CLI must receive the response-only role above the simulated task."""
import json
import os
from types import SimpleNamespace

from benchmarks.agent_study import native_relay as relay


def test_codex_invocation_declares_role_and_disables_remaining_tool_surfaces(monkeypatch):
    # Stub only this module's platform guard; never mutate os.name globally or
    # actually launch a native Windows client from the Linux test allocation.
    monkeypatch.setattr(relay, 'os', SimpleNamespace(name='nt', environ=os.environ.copy()))
    captured = {}
    def run(command, **kwargs):
        captured.update(command=command, **kwargs)
        events = [{'type': 'turn.started'},
                  {'type': 'item.completed', 'item': {'type': 'agent_message',
                    'text': '{"tool":"hold","arguments":{}}'}},
                  {'type': 'turn.completed', 'usage': {}}]
        return SimpleNamespace(stdout='\n'.join(json.dumps(e) for e in events), stderr='', returncode=0)
    monkeypatch.setattr(relay.subprocess, 'run', run)
    monkeypatch.setattr(relay.subprocess, 'CREATE_NO_WINDOW', 0, raising=False)
    request = dict(request_sha256='a' * 64, messages=[dict(role='user', content='fixture')],
        native_policy=dict(transport='codex_subscription', model='gpt-6-luna', effort='medium',
                           prompt_wrapper=relay.WRAPPER))
    result = relay.invoke(request)
    assert result['success'] and result['requested_model'] == 'gpt-6-luna'
    command = captured['command']
    settings = [command[i+1] for i, value in enumerate(command[:-1]) if value == '-c']
    declared, = [s.split('=', 1)[1] for s in settings if s.startswith('developer_instructions=')]
    assert json.loads(declared) == relay.WRAPPER
    for feature in ('multi_agent', 'multi_agent_v2', 'sleep_tool', 'goals', 'tool_suggest'):
        assert f'features.{feature}=false' in settings
    assert '--ignore-user-config' in command and '--ephemeral' in command


def test_wait_tool_event_still_invalidates_response_despite_a_valid_final_json():
    events = [{'type': 'turn.started'},
              {'type': 'item.completed', 'item': {'type': 'collab_tool_call', 'tool': 'wait'}},
              {'type': 'item.completed', 'item': {'type': 'agent_message',
                'text': '{"tool":"hold","arguments":{}}'}},
              {'type': 'turn.completed'}]
    result = relay.parse_events('codex_subscription', '\n'.join(json.dumps(e) for e in events), 0)
    assert not result['success'] and result['native_tool_events']
