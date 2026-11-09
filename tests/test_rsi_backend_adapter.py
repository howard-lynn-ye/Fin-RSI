"""Backend wiring only: no torch import, model loading, Slurm job or inference."""
from types import SimpleNamespace

import pytest

from benchmarks.agent_study import rsi_backend


class FakeChat:
    def __init__(self):
        self.identity = dict(model="fixture", revision="a" * 40)
        self.max_tokens, self.seed, self.calls = 2048, 999, 12
        self.context_limit = 32768
        self.model = SimpleNamespace(dtype="torch.bfloat16", config=SimpleNamespace(
            _commit_hash="a" * 40), generation_config=SimpleNamespace(to_dict=lambda: {"top_k": 50}))
        self.tokenizer = SimpleNamespace(chat_template="fixture template")
        self.hardware = {"fixture": True}
        self.received = []
    def __call__(self, messages):
        self.received.append((self.max_tokens, self.seed + self.calls))
        self.calls += 1
        return {"choices": [{"message": {"content": "fixture"}}],
                "usage": {"completion_tokens": 1}}


def test_adapter_forwards_cap_and_exact_seed_and_restores_state(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "synthetic-no-job-submitted")
    chat = FakeChat()
    adapter = rsi_backend.BoundedTransformersChat(chat)
    adapter.generate_bounded([], max_new_tokens=7, seed=11)
    adapter.generate_bounded([], max_new_tokens=12000, seed=23)
    assert chat.received == [(7, 11), (2048, 23)]
    assert (chat.max_tokens, chat.seed, chat.calls) == (2048, 999, 12)


def test_adapter_refuses_generation_without_allocation(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    chat = FakeChat()
    with pytest.raises(RuntimeError, match="Slurm"):
        rsi_backend.BoundedTransformersChat(chat).generate_bounded([], max_new_tokens=7, seed=11)
    assert chat.received == []


def test_descriptor_tracks_model_template_dependencies_and_implementation(monkeypatch):
    monkeypatch.setattr(rsi_backend, "version", lambda _: "synthetic-test-version")
    chat = FakeChat()
    adapter = rsi_backend.BoundedTransformersChat(chat)
    before = adapter.describe()
    chat.tokenizer.chat_template += " changed"
    assert adapter.describe()["chat_template_sha256"] != before["chat_template_sha256"]
    assert before["implementation_sha256"] and before["dependency_versions"]
    chat.model.config._commit_hash = "b" * 40
    with pytest.raises(ValueError, match="revision differs"):
        adapter.describe()
