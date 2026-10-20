"""TransformersChat generate kwargs: optional sampling override and max_time (no GPU needed).

torch / transformers are replaced by stubs, so this checks only the call the class makes.
"""
import sys
import types
from contextlib import nullcontext

import numpy as np
import pytest

from benchmarks.agent_study import transformers_chat as tc


class _Inputs(dict):
    def __init__(self, n):
        super().__init__(input_ids=np.zeros((1, n), dtype=int))
        self.input_ids = self['input_ids']

    def to(self, device):
        return self


class _Tokenizer:
    eos_token_id = 2
    all_special_tokens = ['<|endoftext|>']

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return 'prompt'

    def __call__(self, prompt, add_special_tokens=False, **kwargs):
        return _Inputs(5)

    def decode(self, ids, skip_special_tokens=False):
        return 'x' * len(ids)


class _Model:
    device = 'cpu'

    def __init__(self):
        self.kwargs = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return np.zeros((1, 5 + 3), dtype=int) + 7


@pytest.fixture
def chat(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', types.SimpleNamespace(inference_mode=nullcontext))
    monkeypatch.setitem(sys.modules, 'transformers', types.SimpleNamespace(set_seed=lambda s: None))
    c = object.__new__(tc.TransformersChat)
    c.tokenizer, c.model = _Tokenizer(), _Model()
    c.max_tokens, c.seed, c.calls, c.context_limit = 16, 11, 0, 4096
    c.stop_token_ids, c.max_time, c.sampling = (2,), None, {}
    return c


def generate_kwargs(c):
    c([{'role': 'user', 'content': 'hi'}])
    return {k: v for k, v in c.model.kwargs.items() if k != 'input_ids'}


def test_defaults_unchanged(chat):
    assert generate_kwargs(chat) == dict(max_new_tokens=16, do_sample=True, temperature=.1,
                                         top_p=1.0, pad_token_id=2)


def test_sampling_override_merges(chat):
    chat.sampling = dict(temperature=0.6, top_p=0.95, top_k=20)
    assert generate_kwargs(chat) == dict(max_new_tokens=16, do_sample=True, temperature=0.6,
                                         top_p=0.95, top_k=20, pad_token_id=2)


def test_max_time_only_when_set(chat):
    assert 'max_time' not in generate_kwargs(chat)
    chat.max_time = 30
    assert generate_kwargs(chat)['max_time'] == 30.0
