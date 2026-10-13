"""A transport must preserve one supplied prompt without client-added turns."""
import json
import os
from types import SimpleNamespace

import pytest
from benchmarks.agent_study import native_relay as relay


def events(*steps, status='SUCCESS', num_turns=1):
    return [dict(event='init', init=dict(model='gemini-3.8-flash-high',
                agent=relay.GEMINI_AGENT_NAME, tools=[])), *steps,
            dict(event='result', result=dict(status=status, num_turns=num_turns,
                 response='{"tool":"read_snapshot","arguments":{}}'))]


def parse(rows):
    return relay.parse_events('antigravity_subscription', '\n'.join(map(json.dumps, rows)), 0)


def test_provider_function_format_error_is_not_reported_as_native_tool_execution():
    error = dict(event='step_update', step_update=dict(step_type='error_message',
                 step_index=1, message='improperly formatted function call'))
    value = parse(events(error, status='ERROR'))
    assert not value['success'] and value['native_tool_events'] == []
    assert value['provider_error_events'][0] == error


@pytest.mark.parametrize('step_index', [2, 4, None])
def test_client_injected_or_unidentified_user_input_cannot_be_an_accepted_answer(step_index):
    followup = dict(event='step_update', step_update=dict(step_type='user_input', step_index=step_index))
    value = parse(events(followup))
    assert not value['success'] and value['provider_error_events'] == [followup]


def test_multiple_internal_turns_are_rejected_even_if_no_followup_event_is_emitted():
    assert not parse(events(num_turns=4))['success']


def test_repeated_updates_of_the_initial_prompt_do_not_invent_additional_inputs():
    initial = dict(event='step_update', step_update=dict(step_type='user_input', step_index=0))
    assert parse(events(initial, initial))['success']


def test_native_tool_event_is_still_rejected_after_transport_change():
    tool = dict(event='step_update', step_update=dict(step_type='tool_call', step_index=1))
    value = parse(events(tool))
    assert not value['success'] and value['native_tool_events'] == [tool]


def test_gemini_sends_complete_long_prompt_through_text_stdin_not_argv(monkeypatch):
    monkeypatch.setattr(relay, 'os', SimpleNamespace(name='nt', environ=os.environ.copy()))
    captured = {}
    def run(command, **kwargs):
        captured.update(command=command, **kwargs)
        return SimpleNamespace(stdout='\n'.join(map(json.dumps, events())), stderr='', returncode=0)
    monkeypatch.setattr(relay.subprocess, 'run', run)
    monkeypatch.setattr(relay.subprocess, 'CREATE_NO_WINDOW', 0, raising=False)
    messages = [dict(role='user', content='nonfinancial-fixture-' * 3000)]
    request = dict(request_sha256='a'*64, messages=messages,
        native_policy=dict(transport='antigravity_subscription', model='gemini-3.8-flash-high',
                           effort='high', prompt_wrapper=relay.WRAPPER))
    value = relay.invoke(request)
    assert value['success']
    command = captured['command']
    assert command[command.index('--input-format')+1] == 'text'
    assert command[command.index('--output-format')+1] == 'stream-json'
    assert captured['input'] == relay.WRAPPER + json.dumps(messages, ensure_ascii=False)
    assert len(captured['input']) > 32767 and all(len(arg) < 1000 for arg in command)
    assert '--json-schema' not in command  # echoing a schema did not enforce it
