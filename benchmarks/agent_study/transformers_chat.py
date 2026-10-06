"""Local open-weight chat callable; requires a Slurm GPU allocation on Beacon."""
import re
import time


def load_tokenizer(model, revision, **kwargs):
    from transformers import AutoTokenizer
    # Match these pinned exports to the publisher's tekken.json: Small needs
    # correction; Nemo already matches. Do not apply the fix to other families.
    if model in ('mistralai/Mistral-Nemo-Instruct-2407',
                 'mistralai/Mistral-Small-24B-Instruct-2501'):
        kwargs['fix_mistral_regex'] = True
    return AutoTokenizer.from_pretrained(model, revision=revision, **kwargs)


def tokenize_chat(tokenizer, messages, **kwargs):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Chat templates already supply BOS/EOS/control tokens. Do not add them twice.
    return tokenizer(prompt, add_special_tokens=False, **kwargs)


def context_limit(config, tokenizer):
    limits = (32768, getattr(config, 'max_position_embeddings', 32768),
              tokenizer.model_max_length)
    return min(n for n in limits if isinstance(n, int) and n > 0)


def validate_reasoning_tokens(tokenizer):
    unsupported = [token for token in tokenizer.all_special_tokens
                   if re.search('think|thought|reason', token, re.IGNORECASE)
                   and token not in ('<think>', '</think>')]
    if unsupported:
        raise ValueError(f'unqualified reasoning delimiters: {unsupported}')


def decode_response(tokenizer, tokens, prompt, stop_token_ids=()):
    # Some tokenizers classify reasoning delimiters as special tokens. Dropping
    # them can turn a proposed call inside reasoning into an executable action.
    ids = list(tokens)
    for i, token in enumerate(ids):
        if int(token) in stop_token_ids:
            ids = ids[:i]
            break
    text = tokenizer.decode(ids, skip_special_tokens=False)
    for token in sorted(tokenizer.all_special_tokens, key=len, reverse=True):
        if token not in ('<think>', '</think>'):
            text = text.replace(token, '')
    if prompt.rstrip().endswith('<think>'):
        text = '<think>' + text
    return text


class TransformersChat:
    def __init__(self, model, revision, *, max_tokens=2048, seed=0):
        import torch
        from transformers import AutoModelForCausalLM
        if not torch.cuda.is_available():
            raise RuntimeError("GPU allocation required; no CPU inference fallback")
        self.tokenizer = load_tokenizer(model, revision)
        validate_reasoning_tokens(self.tokenizer)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, revision=revision, torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation="sdpa")
        self.model.eval()
        self.max_tokens, self.seed, self.calls = max_tokens, seed, 0
        self.context_limit = context_limit(self.model.config, self.tokenizer)
        stops = self.model.generation_config.eos_token_id
        self.stop_token_ids = tuple(stops if isinstance(stops, (list, tuple)) else
                                    [self.tokenizer.eos_token_id if stops is None else stops])

    def __call__(self, messages):
        import torch
        from transformers import set_seed
        set_seed(self.seed + self.calls)
        self.calls += 1
        inputs = tokenize_chat(self.tokenizer, messages, return_tensors="pt").to(self.model.device)
        count = inputs.input_ids.shape[-1]
        if count + self.max_tokens > self.context_limit:
            raise ValueError("context budget exceeded; no silent truncation")
        started = time.monotonic()
        with torch.inference_mode():
            result = self.model.generate(**inputs, max_new_tokens=self.max_tokens,
                                         do_sample=True, temperature=.1, top_p=1.0,
                                         pad_token_id=self.tokenizer.eos_token_id)
        tokens = result[0, count:]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        stopped = any(int(token) in self.stop_token_ids for token in tokens)
        return {"choices": [{"message": {"content": decode_response(
                    self.tokenizer, tokens, prompt, self.stop_token_ids)},
                              "finish_reason": "length" if len(tokens) == self.max_tokens and not stopped else "stop"}],
                "backend_metadata": {"stop_token_ids": self.stop_token_ids},
                "usage": {"prompt_tokens": count, "completion_tokens": len(tokens),
                          "total_tokens": count + len(tokens)},
                "generation_seconds": time.monotonic() - started}
