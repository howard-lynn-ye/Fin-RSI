from types import SimpleNamespace

import pytest

from benchmarks.agent_study.transformers_chat import (
    context_limit, decode_response, tokenize_chat, validate_reasoning_tokens,
)


def test_chat_tokens_preserve_single_bos_and_eos():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return '<bos>user hello<eos>assistant'

        def __call__(self, text, add_special_tokens=True, **kwargs):
            tokens = [s for s in text.replace('<', ' <').replace('>', '> ').split()]
            return {'input_ids': (['<bos>'] if add_special_tokens else []) + tokens}

    result = tokenize_chat(Tokenizer(), [{'role': 'user', 'content': 'hello'}])
    assert result['input_ids'] == ['<bos>', 'user', 'hello', '<eos>', 'assistant']


@pytest.mark.parametrize('model,tokenizer,wanted', [(4096, 32768, 4096),
    (131072, 65536, 32768), (32768, 8192, 8192), (None, 10**30, 32768)])
def test_context_uses_actual_smallest_supported_limit(model, tokenizer, wanted):
    assert context_limit(SimpleNamespace(max_position_embeddings=model),
                         SimpleNamespace(model_max_length=tokenizer)) == wanted


@pytest.mark.parametrize('implicit', [False, True])
def test_reasoning_special_tokens_survive_until_final_action_parsing(implicit):
    from benchmarks.agent_study.trading_runtime_v6 import extract_call
    draft = '{"tool":"submit","arguments":{"SPY":1}}'
    final = '{"tool":"submit","arguments":{"IEF":1}}'

    class Tokenizer:
        all_special_tokens = ['<think>', '</think>', '<eos>']

        def decode(self, tokens, *, skip_special_tokens):
            assert skip_special_tokens is False
            return ('' if implicit else '<think>') + draft + '</think>' + final + '<eos>'

    prompt = 'assistant\n<think>\n' if implicit else 'assistant\n'
    text = decode_response(Tokenizer(), [], prompt)
    call, status = extract_call(text)
    assert call['arguments'] == {'IEF': 1}
    assert '<eos>' not in text and '<think>' in text


@pytest.mark.parametrize('text,status', [
    ('<think>{"tool":"submit","arguments":{"SPY":1}}', 'reasoning-incomplete'),
    ('</think>{"tool":"submit","arguments":{"SPY":1}}', 'reasoning-malformed'),
    ('<think><think>nested</think></think>{"tool":"submit","arguments":{"SPY":1}}',
     'reasoning-malformed'),
])
def test_incomplete_or_malformed_reasoning_cannot_execute_a_trade(text, status):
    from benchmarks.agent_study.trading_runtime_v6 import extract_call
    assert extract_call(text) == (None, status)


def test_unqualified_reasoning_delimiters_stop_before_inference():
    validate_reasoning_tokens(SimpleNamespace(all_special_tokens=['<think>', '</think>', '<eos>']))
    with pytest.raises(ValueError, match='unqualified reasoning'):
        validate_reasoning_tokens(SimpleNamespace(all_special_tokens=['[THINK]', '[/THINK]']))


@pytest.mark.parametrize('fenced', [False, True])
def test_multiple_json_actions_cannot_silently_execute_only_the_first(fenced):
    from benchmarks.agent_study.trading_runtime_v6 import extract_call
    calls = ['{"tool":"list_algorithms","arguments":{}}',
             '{"tool":"submit","arguments":{"SPY":1}}']
    if fenced:
        calls = ['```json\n' + call + '\n```' for call in calls]
    assert extract_call('\n'.join(calls)) == (None, 'multiple-actions')


def test_two_python_blocks_are_rejected_before_lenient_fallback():
    from benchmarks.agent_study.trading_runtime_v6 import extract_call
    response = '"tool":"run_python"\n```python\nprint(1)\n```\n```python\nsubmit({"SPY":1})\n```'
    assert extract_call(response) == (None, 'multiple-actions')


def test_json_literal_inside_one_python_action_is_not_an_extra_action():
    from benchmarks.agent_study.trading_runtime_v6 import extract_call
    code = 'example = {"tool":"read_market","arguments":{}}\nprint(example)'
    call, status = extract_call('```python\n' + code + '\n```')
    assert call == {'tool': 'run_python', 'arguments': {'code': code}}
    assert status == 'python-fence'


def test_nested_json_code_and_reasoning_examples_do_not_count_as_extra_actions():
    import json
    from benchmarks.agent_study.trading_runtime_v6 import extract_call
    action = dict(tool='run_python', arguments=dict(
        code='print({"tool": "example", "arguments": {}})'))
    text = '<think>{"tool":"read_market"}</think>' + json.dumps(action)
    assert extract_call(text) == (action, 'json')


@pytest.mark.parametrize('malformed', [
    '{"tool":"submit","arguments":{"weights":weights}}',
    '{"tool":"submit","arguments":{"weights":',
])
@pytest.mark.parametrize('malformed_first', [False, True])
def test_malformed_extra_action_cannot_be_hidden_by_a_valid_read(malformed, malformed_first):
    from benchmarks.agent_study.trading_runtime_v6 import extract_call
    valid = '{"tool":"read_evidence","arguments":{}}'
    parts = [malformed, valid] if malformed_first else [valid, malformed]
    assert extract_call('\n'.join(parts)) == (None, 'multiple-actions')


def test_single_legacy_python_envelope_and_quoted_json_keep_their_existing_meaning():
    from benchmarks.agent_study.trading_runtime_v6 import extract_call
    text = '{"tool":"run_python","arguments":{"code":"""print(1)"""}}'
    assert extract_call(text) == ({'tool': 'run_python', 'arguments': {'code': 'print(1)'}}, 'lenient')


def test_decoding_stops_before_a_fabricated_next_turn():
    class Tokenizer:
        all_special_tokens = ['<eos>']

        def decode(self, tokens, **kwargs):
            assert tokens == [1, 2]
            return 'final answer'
    assert decode_response(Tokenizer(), [1, 2, 99, 3, 4], 'assistant', (99,)) == 'final answer'


def test_model_matrix_rejects_alias_counting_and_unpinned_revisions(tmp_path):
    import json
    from benchmarks.agent_study.model_matrix import matrix
    rows = [dict(id=str(i), model=f'model/{i}', transport='hf', revision='a'*40)
            for i in range(20)]
    path = tmp_path / 'models.json'
    path.write_text(json.dumps({'models': rows}))
    assert len(matrix(path)) == 20
    rows[-1]['model'] = rows[0]['model']
    path.write_text(json.dumps({'models': rows}))
    with pytest.raises(ValueError, match='duplicate model'):
        matrix(path)
    rows[-1]['model'] = 'model/last'
    rows[-1]['revision'] = 'main'
    path.write_text(json.dumps({'models': rows}))
    with pytest.raises(ValueError, match='full commit hashes'):
        matrix(path)


def test_matrix_cannot_silently_replace_a_declared_revision_or_model():
    from benchmarks.agent_study.model_matrix import hf_models
    legacy = [('7b', 'Org/Old', 'a' * 40)]
    row = dict(id='alias', model='Org/Old', transport='hf', revision='b' * 40)
    with pytest.raises(ValueError, match='revision differs'):
        hf_models([row], legacy)
    row.update(id='7b', model='Other/New', revision='a' * 40)
    with pytest.raises(ValueError, match='collides'):
        hf_models([row], legacy)
