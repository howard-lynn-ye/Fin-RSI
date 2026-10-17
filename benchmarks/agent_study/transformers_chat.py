"""Local open-weight chat callable; requires a Slurm GPU allocation on Beacon."""
import re
import time
from contextlib import contextmanager
from collections.abc import Mapping


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

def qwen32_dual_map(model, revision, capacities):
    """Explicit BF16 placement; the two cards do not form one memory pool."""
    if (model, revision) != ('Qwen/Qwen2.5-Coder-32B-Instruct',
                            '381fc969f78efac66bc87ff7ddeadb7e73c218a7'):
        raise ValueError('dual layout is qualified only for the pinned Qwen coder 32B')
    if len(capacities) != 2 or min(capacities) < 44 * 1024**3:
        raise ValueError('dual layout requires exactly two allocated GPUs of at least 44 GiB')
    layout = {'model.embed_tokens': 0, 'model.rotary_emb': 0,
              'model.norm': 1, 'lm_head': 0}
    layout.update({f'model.layers.{i}': int(i >= 32) for i in range(64)})
    # Keeping logits on the input device avoids the previously reproduced
    # cross-device copy+cast inside generation. No parameter is quantized/offloaded.
    return layout


def host_staged_sender(torch, fallback, honor_type):
    """Preserve dtype and values through blocking host copies, never peer DMA."""
    def send(value, device, non_blocking=False, skip_keys=None):
        if torch.is_tensor(value):
            target = torch.device(f'cuda:{device}' if isinstance(device, int) else device)
            if (value.device.type == target.type == 'cuda'
                    and value.device.index != target.index):
                return value.to('cpu', non_blocking=False).to(target, non_blocking=False)
            return fallback(value, device, non_blocking=non_blocking, skip_keys=skip_keys)
        if isinstance(value, (tuple, list)):
            return honor_type(value, (send(v, device, non_blocking, skip_keys) for v in value))
        if isinstance(value, Mapping):
            skip = {skip_keys} if isinstance(skip_keys, str) else set(skip_keys or ())
            return type(value)({k: v if k in skip else send(v, device, non_blocking, skip_keys)
                                for k, v in value.items()})
        return fallback(value, device, non_blocking=non_blocking, skip_keys=skip_keys)
    return send


@contextmanager
def transfer_scope(receipt, torch):
    """Scope the transfer hook to one serial benchmark generation; restore on error."""
    if not receipt or receipt.get('transfer_route') != 'cpu_staging':
        yield
        return
    import accelerate.hooks as hooks
    from accelerate.utils.operations import honor_type
    original = hooks.send_to_device
    hooks.send_to_device = host_staged_sender(torch, original, honor_type)
    try:
        yield
    finally:
        hooks.send_to_device = original


def allocated_layout(model, revision, torch):
    import os
    mode = os.environ.get('FIN_STUDY_GPU_LAYOUT', 'single')
    if mode == 'single':
        return {'': 0}, None
    if mode not in ('dual_qwen32_bf16', 'dual_qwen32_bf16_host'):
        raise ValueError('unknown explicit GPU layout: ' + mode)
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('multi-GPU qualification requires a Slurm allocation')
    sizes = [torch.cuda.get_device_properties(i).total_memory
             for i in range(torch.cuda.device_count())]
    layout = qwen32_dual_map(model, revision, sizes)
    checks = []
    # Check actual BF16 activation, float32 and index-copy paths in both
    # directions, including noncontiguous tensors, before loading any weights.
    for src, dst in ((0, 1), (1, 0)):
        for dtype in (torch.bfloat16, torch.float32, torch.int64):
            for shape in ((128, 256), (1, 152064), (257, 5120)):
                reference = torch.arange(shape[0] * shape[1], dtype=torch.int64).reshape(shape)
                reference = ((reference % 251) - 125).to(dtype=dtype)
                for transpose in (False, True):
                    expected = reference.t() if transpose else reference
                    actual = expected.to(device=f'cuda:{src}')
                    if mode == 'dual_qwen32_bf16_host':
                        actual = actual.to('cpu', non_blocking=False)
                    actual = actual.to(device=f'cuda:{dst}', copy=True, non_blocking=False)
                    torch.cuda.synchronize(dst)
                    if not torch.equal(actual.cpu(), expected):
                        raise RuntimeError(
                            f'GPU transfer corrupted: {src}->{dst}, {dtype}, {shape}, transpose={transpose}')
                    checks.append(dict(src=src, dst=dst, dtype=str(dtype), shape=shape,
                                       transpose=transpose, exact=True))
    return layout, dict(mode=mode, memory_bytes=sizes, transfer_checks=checks,
                        transfer_route='cpu_staging' if mode.endswith('_host') else 'peer')


def qualify_dual_context(backend, torch, receipt):
    """Prove placement and full configured context fit before investment decisions."""
    if receipt is None:
        return
    import json
    if backend.model.config.num_hidden_layers != 64:
        raise RuntimeError('unexpected model architecture for explicit layout')
    if any(p.device.type != 'cuda' or p.dtype != torch.bfloat16
           for p in backend.model.parameters()):
        raise RuntimeError('weights must remain entirely on GPU in BF16')
    if set(str(v) for v in backend.model.hf_device_map.values()) != {'0', '1'}:
        raise RuntimeError('both allocated GPUs must contain model layers')
    for i in range(2):
        torch.cuda.reset_peak_memory_stats(i)
    count = backend.context_limit - 1
    token = backend.tokenizer.encode('market', add_special_tokens=False)[-1]
    ids = torch.full((1, count), token, dtype=torch.long, device='cuda:0')
    with torch.inference_mode(), transfer_scope(receipt, torch):
        result = backend.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
            max_new_tokens=1, do_sample=False, use_cache=True,
            logits_processor=[FiniteScores(count)],
            pad_token_id=backend.tokenizer.eos_token_id)
    if result.shape[-1] != count + 1:
        raise RuntimeError('full-context generation did not complete')
    receipt.update(full_context_tokens=count + 1,
                   peak_allocated_bytes=[torch.cuda.max_memory_allocated(i) for i in range(2)],
                   passed=True)
    del result, ids
    torch.cuda.empty_cache()
    print(json.dumps({'gpu_layout_qualification': receipt}), flush=True)



class TransformersChat:
    def __init__(self, model, revision, *, max_tokens=2048, seed=0):
        import torch
        from transformers import AutoModelForCausalLM
        if not torch.cuda.is_available():
            raise RuntimeError("GPU allocation required; no CPU inference fallback")
        self.identity = dict(model=model, revision=revision, max_tokens=max_tokens)
        self.tokenizer = load_tokenizer(model, revision)
        validate_reasoning_tokens(self.tokenizer)
        # Default placement stays single-GPU. Explicit dual placement is guarded;
        # weights, BF16 precision and generation budgets remain unchanged.
        layout, layout_receipt = allocated_layout(model, revision, torch)
        self.model = AutoModelForCausalLM.from_pretrained(
            model, revision=revision, torch_dtype=torch.bfloat16, device_map=layout,
            attn_implementation="sdpa")
        self.model.eval()
        self.hardware = dict(gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                             device_map={k: str(v) for k, v in self.model.hf_device_map.items()})
        self.max_tokens, self.seed, self.calls = max_tokens, seed, 0
        self.context_limit = context_limit(self.model.config, self.tokenizer)
        stops = self.model.generation_config.eos_token_id
        self.stop_token_ids = tuple(stops if isinstance(stops, (list, tuple)) else
                                    [self.tokenizer.eos_token_id if stops is None else stops])
        qualify_dual_context(self, torch, layout_receipt)
        if layout_receipt is not None:
            self.hardware['placement_qualification'] = layout_receipt

    def prepare_final_decision(self):
        from benchmarks.agent_study.trading_schema_v8 import output_contract, prepare_decoder
        if not hasattr(self, '_decision_tokenizer_data'):
            self._decision_tokenizer_data = prepare_decoder(self.tokenizer, self.stop_token_ids)
        return output_contract(True)

    def final_decision(self, messages):
        self.prepare_final_decision()
        return self._generate(messages, structured=True)

    def prepare_research_call(self, contract):
        from benchmarks.agent_study.trading_research_v9 import check_contract
        from benchmarks.agent_study.trading_schema_v8 import TRANSPORT
        check_contract(contract)
        if contract['transport'] != TRANSPORT:
            raise ValueError('structured research contract required')
        self.prepare_final_decision()
        return contract

    def research_call(self, messages, contract):
        self.prepare_research_call(contract)
        return self._generate(messages, structured=False, research_output=contract)

    def __call__(self, messages):
        return self._generate(messages, structured=False)

    def _generate(self, messages, *, structured, research_output=None):
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
        elif research_output is not None:
            from benchmarks.agent_study.trading_research_v9 import prefix_filter
            options['prefix_allowed_tokens_fn'] = prefix_filter(
                self._decision_tokenizer_data, prompt, research_output)
            metadata['research_output'] = research_output
        started = time.monotonic()
        try:
            with torch.inference_mode(), transfer_scope(
                    self.hardware.get('placement_qualification'), torch):
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
