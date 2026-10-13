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


class InvalidSamplingScores(RuntimeError):
    def __init__(self, diagnostic):
        super().__init__('nonfinite or empty sampling distribution; generation stopped')
        self.diagnostic = diagnostic


class FiniteScores:
    """Fail before multinomial poisons the CUDA context; never replace logits."""
    def __init__(self, prompt_tokens):
        self.prompt_tokens = prompt_tokens

    def __call__(self, input_ids, scores):
        import torch
        finite = torch.isfinite(scores)
        if (torch.isnan(scores).any() or torch.isposinf(scores).any()
                or not finite.any(dim=-1).all()):
            raise InvalidSamplingScores(dict(
                nan_count=int(torch.isnan(scores).sum()),
                positive_inf_count=int(torch.isposinf(scores).sum()),
                finite_per_row=finite.sum(dim=-1).tolist(),
                generated_token_ids=input_ids[:, self.prompt_tokens:].tolist()))
        return scores

class TransformersChat:
    def __init__(self, model, revision, *, max_tokens=2048, seed=0):
        import torch
        from transformers import AutoModelForCausalLM
        if not torch.cuda.is_available():
            raise RuntimeError("GPU allocation required; no CPU inference fallback")
        self.identity = dict(model=model, revision=revision, max_tokens=max_tokens)
        self.tokenizer = load_tokenizer(model, revision)
        validate_reasoning_tokens(self.tokenizer)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, revision=revision, torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation="sdpa")
        self.model.eval()
        self.hardware = dict(gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                             device_map={k: str(v) for k, v in self.model.hf_device_map.items()})
        self.max_tokens, self.seed, self.calls = max_tokens, seed, 0
        self.context_limit = context_limit(self.model.config, self.tokenizer)
        stops = self.model.generation_config.eos_token_id
        self.stop_token_ids = tuple(stops if isinstance(stops, (list, tuple)) else
                                    [self.tokenizer.eos_token_id if stops is None else stops])

    def prepare_final_decision(self):
        from benchmarks.agent_study.trading_schema_v8 import output_contract, prepare_decoder
        if not hasattr(self, '_decision_tokenizer_data'):
            self._decision_tokenizer_data = prepare_decoder(self.tokenizer, self.stop_token_ids)
        return output_contract(True)

    def final_decision(self, messages):
        self.prepare_final_decision()
        return self._generate(messages, structured=True)

    def __call__(self, messages):
        return self._generate(messages, structured=False)

    def _generate(self, messages, *, structured):
        import torch
        from transformers import set_seed
        set_seed(self.seed + self.calls)
        self.calls += 1
        inputs = tokenize_chat(self.tokenizer, messages, return_tensors="pt").to(self.model.device)
        count = inputs.input_ids.shape[-1]
        if count + self.max_tokens > self.context_limit:
            raise ValueError("context budget exceeded; no silent truncation")
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        options, metadata = {}, {'stop_token_ids': self.stop_token_ids, 'identity': dict(self.identity),
                                 'hardware': self.hardware}
        if structured:
            from benchmarks.agent_study.trading_schema_v8 import output_contract, prefix_filter
            options['prefix_allowed_tokens_fn'] = prefix_filter(self._decision_tokenizer_data, prompt)
            metadata['final_output'] = output_contract(True)
        started = time.monotonic()
        try:
            with torch.inference_mode():
                result = self.model.generate(**inputs, max_new_tokens=self.max_tokens,
                                             do_sample=True, temperature=.1, top_p=1.0,
                                             logits_processor=[FiniteScores(count)],
                                             pad_token_id=self.tokenizer.eos_token_id, **options)
        except InvalidSamplingScores as exc:
            from benchmarks.agent_study.trading_schema_v8 import DecoderFailure
            failure = DecoderFailure(dict(error=str(exc), sampling_diagnostic=exc.diagnostic,
                backend_metadata=metadata), [], metadata.get('final_output'), messages)
            failure.args = ('infrastructure failure: invalid sampling scores; no decision executed',)
            raise failure from exc
        tokens = result[0, count:]
        stopped = any(int(token) in self.stop_token_ids for token in tokens)
        return {"choices": [{"message": {"content": decode_response(
                    self.tokenizer, tokens, prompt, self.stop_token_ids)},
                              "finish_reason": "length" if len(tokens) == self.max_tokens and not stopped else "stop"}],
                "backend_metadata": metadata,
                "usage": {"prompt_tokens": count, "completion_tokens": len(tokens),
                          "total_tokens": count + len(tokens)},
                "generation_seconds": time.monotonic() - started}
