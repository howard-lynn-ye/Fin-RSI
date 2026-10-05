from types import SimpleNamespace

import pytest

from benchmarks.agent_study.transformers_chat import context_limit, tokenize_chat


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


def test_model_matrix_rejects_alias_counting_and_unpinned_revisions(tmp_path):
    pytest.importorskip('bs4')
    import json
    from benchmarks.agent_study.qualify_model_matrix import matrix
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
