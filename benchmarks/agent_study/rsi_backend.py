"""Bounded adapter over an already loaded, qualified TransformersChat instance.

This module never loads or downloads weights. Actual generation requires a Slurm allocation;
the caller must first perform the project's complete runtime/resource qualification.
"""
import hashlib
from importlib.metadata import version
import os
from pathlib import Path


class BoundedTransformersChat:
    def __init__(self, chat):
        self.chat = chat
        self.output_cap = chat.max_tokens

    def __call__(self, messages):
        raise RuntimeError("use generate_bounded with an explicit remaining cap and seed")

    def describe(self):
        chat = self.chat
        identity = chat.identity
        revision = identity["revision"]
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError("a complete model revision is required")
        resolved = getattr(chat.model.config, "_commit_hash", None)
        if resolved is not None and resolved != revision:
            raise ValueError("loaded model revision differs from the requested revision")
        if str(chat.model.dtype) != "torch.bfloat16" or getattr(chat.model, "is_quantized", False):
            raise ValueError("the declared backend requires complete BF16 weights")
        if chat.max_tokens != self.output_cap:
            raise ValueError("backend output configuration changed outside a bounded call")
        implementation = Path(__file__).with_name("transformers_chat.py")
        return dict(model_id=identity["model"], model_revision=revision,
                    tokenizer_revision=revision, precision=str(chat.model.dtype),
                    context_tokens=chat.context_limit, max_new_tokens=self.output_cap,
                    temperature=0.1, top_p=1.0,
                    generation_defaults=chat.model.generation_config.to_dict(),
                    chat_template_sha256=hashlib.sha256(
                        str(chat.tokenizer.chat_template).encode()).hexdigest(),
                    implementation_sha256=hashlib.sha256(implementation.read_bytes()).hexdigest(),
                    dependency_versions={p: version(p) for p in ("torch", "transformers", "accelerate")},
                    hardware=chat.hardware,
                    revision_provenance="pinned loader argument; resolved model config checked when present")

    def generate_bounded(self, messages, *, max_new_tokens, seed):
        if not os.environ.get("SLURM_JOB_ID"):
            raise RuntimeError("actual backend generation requires a Slurm allocation")
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("remaining output allowance must be a positive integer")
        chat = self.chat
        saved = chat.max_tokens, chat.seed, chat.calls
        try:
            chat.max_tokens = min(max_new_tokens, self.output_cap)
            # TransformersChat uses seed + calls: make this exactly the Meter's stream seed.
            chat.seed, chat.calls = int(seed), 0
            return chat(messages)
        finally:
            chat.max_tokens, chat.seed, chat.calls = saved
