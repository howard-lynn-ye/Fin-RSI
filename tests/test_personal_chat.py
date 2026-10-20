import json
from types import SimpleNamespace

import pytest

from benchmarks.agent_study.personal_chat import PersonalChat, submit_response
from benchmarks.agent_study.personal_chat import fingerprint, validate_exchange, validate_exchange_inventory


class Tokenizer:
    def encode(self, text, **kwargs):
        return text.split()


def pending(tmp_path, messages=None):
    chat = PersonalChat(tmp_path, Tokenizer(), timeout=0)
    chat.arm, chat.decision_index = 'raw', 0
    messages = messages or [{'role': 'user', 'content': 'synthetic decision'}]
    with pytest.raises(RuntimeError, match='absent'):
        chat(messages)
    request = tmp_path / 'personal-exchange/raw/00/00/request.json'
    assert request.exists() and not request.with_name('response.json').exists()
    assert chat.calls == 0 and chat.receipts == []
    return chat, request, messages


def test_no_decision_exists_until_operator_supplies_literal_response(tmp_path):
    chat, request, messages = pending(tmp_path)
    literal = '{"tool":"submit","arguments":{"weights":{"SPY":0.4}}}'
    submit_response(request, literal, 'Personal synthetic fixture: explicitly choose forty percent.')
    result = chat(messages)
    assert result['choices'][0]['message']['content'] == literal
    assert result['choices'][0]['finish_reason'] == 'stop'
    assert chat.calls == 1 and len(chat.receipts) == 1
    with pytest.raises(FileExistsError):
        submit_response(request, literal, 'No overwrite of the first personally submitted action.')


def test_changed_request_or_response_binding_refuses_reuse(tmp_path):
    chat, request, messages = pending(tmp_path)
    submit_response(request, 'explicit response', 'Enough explanation for the recorded personal choice.')
    with pytest.raises(RuntimeError, match='resumed request differs'):
        chat([{'role': 'user', 'content': 'a different information set'}])
    path = request.with_name('response.json')
    value = json.loads(path.read_text())
    value['request_sha256'] = '0' * 64
    path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match='does not match'):
        chat(messages)


def test_response_budget_and_eight_turn_limit_are_enforced(tmp_path):
    chat, request, messages = pending(tmp_path)
    submit_response(request, 'word ' * 1025, 'Deliberately oversized response must never be executed.')
    assert chat(messages)['choices'][0]['finish_reason'] == 'length'
    chat.calls = 8
    with pytest.raises(RuntimeError, match='eight-call limit'):
        chat(messages)


def test_same_decision_runtime_executes_only_authored_submission(tmp_path):
    from benchmarks.agent_study import trading_runtime_v6 as runtime
    chat = PersonalChat(tmp_path, Tokenizer(), timeout=0)
    chat.arm, chat.decision_index = 'raw', 0
    controller = SimpleNamespace(arm='raw', calls=[], menu_seed=11)
    with pytest.raises(RuntimeError, match='absent'):
        runtime.decide(chat, controller, 'Synthetic same-runtime task')
    request = tmp_path / 'personal-exchange/raw/00/00/request.json'
    submit_response(request, '{"tool":"submit","arguments":{"weights":{"SPY":0.4}}}',
                    'I explicitly choose this synthetic allocation for an accounting fixture.')
    record = runtime.decide(chat, controller, 'Synthetic same-runtime task')
    assert record['submitted'] and record['target']['SPY'] == .4
    assert len(record['turns']) == len(chat.receipts) == 1
    item = chat.receipts[0]
    relative = validate_exchange(tmp_path, 'raw', 0, 0, item, record['turns'][0])
    validate_exchange_inventory(tmp_path, [relative])
    with pytest.raises(ValueError, match='another decision'):
        validate_exchange(tmp_path, 'library', 0, 0, item, record['turns'][0])
    with pytest.raises(ValueError, match='binding changed'):
        validate_exchange(tmp_path, 'raw', 0, 0, item,
                          dict(record['turns'][0], input_messages_sha256='bad'))
    with pytest.raises(ValueError, match='orphaning'):
        validate_exchange_inventory(tmp_path, [])


def test_accepted_response_cannot_change_during_resume(tmp_path):
    chat, request, messages = pending(tmp_path)
    submit_response(request, 'first explicit response', 'This is the first recorded personal response.')
    chat(messages)
    original = chat.receipts[0]
    chat.calls, chat.receipts = 0, []
    chat(messages)
    assert chat.receipts == [original]
    path = request.with_name('response.json')
    value = json.loads(path.read_text())
    value['text'] = 'a revised response after seeing feedback'
    path.write_text(json.dumps(value))
    chat.calls = 0
    with pytest.raises(RuntimeError, match='previously accepted response changed'):
        chat(messages)
    path.unlink()
    with pytest.raises(RuntimeError, match='previously accepted response is missing'):
        chat(messages)


def test_shared_runtime_does_not_execute_oversized_personal_submit():
    from benchmarks.agent_study import trading_runtime_v6 as runtime
    def oversized(messages):
        return dict(choices=[dict(message=dict(content=
            '{"tool":"submit","arguments":{"weights":{"SPY":1}}}'), finish_reason='length')])
    record = runtime.decide(oversized, SimpleNamespace(arm='raw', calls=[], menu_seed=0), 'fixture')
    assert not record['submitted'] and record['target'] is None
    assert len(record['turns']) == 8
    assert all(t['parse'] == 'generation-truncated' and t['tool'] is None for t in record['turns'])


def test_personal_case_cannot_be_mislabeled_as_a_sampling_seed_or_model_mean(tmp_path):
    pytest.importorskip('bs4')
    from benchmarks.agent_study.multisource_model_study import model_spec, aggregate
    assert model_spec('assistant', 11, 'personal')[1] == 'assistant_in_current_conversation'
    with pytest.raises(ValueError, match='one run'):
        model_spec('assistant', 23, 'personal')
    with pytest.raises(ValueError, match='separately'):
        aggregate(tmp_path, 'personal')


def test_personal_operator_cannot_see_the_next_date_before_locking_both_arms():
    pytest.importorskip('bs4')
    from benchmarks.agent_study.multisource_model_study import decision_groups
    inputs = dict(arms=['library', 'raw'], picks=[5, 15])
    def expanded(personal):
        return [(arm, i, pick) for arm, rows in decision_groups(inputs, personal) for i, pick in rows]
    assert expanded(True) == [('raw', 0, 5), ('library', 0, 5), ('raw', 1, 15), ('library', 1, 15)]
    assert expanded(False) == [('library', 0, 5), ('library', 1, 15), ('raw', 0, 5), ('raw', 1, 15)]
