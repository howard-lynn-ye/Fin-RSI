"""Native-context admission and exact per-request checks without changing model inputs."""
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tokenizer_identity(tokenizer):
    return dict(class_name=type(tokenizer).__name__, model_max_length=tokenizer.model_max_length,
        chat_template_sha256=digest(tokenizer.chat_template), vocabulary_sha256=digest(tokenizer.get_vocab()),
        special_tokens_sha256=digest(tokenizer.special_tokens_map))


def native_identity(config, tokenizer):
    from benchmarks.agent_study.transformers_chat import context_limit
    native = getattr(config, 'max_position_embeddings', None)
    if type(native) is not int or native < 1:
        raise ValueError('explicit positive native model context required')
    return dict(native_model_tokens=native, tokenizer_tokens=tokenizer.model_max_length,
        effective_tokens=context_limit(config, tokenizer), rope_scaling=getattr(config,'rope_scaling',None),
        tokenizer=tokenizer_identity(tokenizer))


def expected_context(config, profile=None):
    """The original exact32k contract remains the default unless a bound profile is supplied."""
    if profile is None:
        return config['models']['common_generation']['context_tokens']
    native = profile['native']
    limits = [32768, native['native_model_tokens'], native['tokenizer_tokens']]
    effective = min(n for n in limits if type(n) is int and n > 0)
    if effective != native['effective_tokens']:
        raise ValueError('native context profile has an inconsistent effective limit')
    return effective


def validate_profile(ref, *, model_id, model, protocol_sha, enrollment_sha, amendment_sha,
                     source_sha, planned_cells, helper_hashes):
    if not isinstance(ref, dict) or set(ref) != {'path','sha256'} or sha(ref['path']) != ref['sha256']:
        raise ValueError('matching explicit context-profile path/hash required')
    path = Path(ref['path'])
    profile = json.loads(path.read_text(encoding='utf-8'))
    position_cells = sum(c['track']=='P' for c in planned_cells)
    external_cells = sum(c['track']=='E' for c in planned_cells)
    expected = dict(status='initial_inputs_checked', model_id=model_id, model=model['model'],
        revision=model['revision'], protocol_sha256=protocol_sha,
        enrollment_terminal_sha256=enrollment_sha, amendment_sha256=amendment_sha,
        source_manifest_sha256=source_sha, helper_hashes=helper_hashes,
        initial_cell_counts=dict(P=position_cells,E=external_cells))
    if any(profile.get(k) != value for k,value in expected.items()) or profile.get('overflow_initial_cells') != 0:
        raise ValueError('context profile differs from complete fixed inputs or declared identity')
    evidence = path.parent / profile['initial_rows_file']
    if evidence.parent.resolve() != path.parent.resolve() or sha(evidence) != profile['initial_rows_sha256']:
        raise ValueError('context input evidence changed')
    rows = [json.loads(line) for line in evidence.read_text(encoding='utf-8').splitlines()]
    keys = [(r['track'],r['method'],r['library'],r['task_id'],r['seed']) for r in rows]
    if len(set(keys)) != len(keys) or len(rows) != position_cells+external_cells:
        raise ValueError('context input evidence is duplicate or incomplete')
    key = lambda r: (r['track'],r['method'],r['library'],r['task_id'],r['seed'])
    planned = {key(c):c for c in planned_cells}
    if set(keys) != set(planned):
        raise ValueError('context evidence differs from fixed cell identities')
    for row in rows:
        cell = planned[key(row)]
        fields = ('task_sha256',) if row['track']=='P' else ('public_sha256',)
        if any(row.get(f) != cell[f] for f in fields):
            raise ValueError('context evidence differs from fixed input content')
        if row['track']=='P' and row.get('frozen_messages_sha256') != cell['messages_sha256']:
            raise ValueError('context evidence differs from frozen P messages')
    cap = expected_context({}, profile)
    if any(type(r['prompt_tokens']) is not int or r['prompt_tokens'] < 1
           or r['output_cap'] != 2048 or r['prompt_tokens']+2048 > cap for r in rows):
        raise ValueError('context evidence cannot fit the full unchanged output cap')
    return profile


class ContextIncompatibility(RuntimeError):
    """The complete unchanged request cannot fit; no model-quality verdict is available."""


class ContextGuardedBackend:
    def __init__(self, backend, profile, *, count_tokens=None, emit=None):
        self.backend, self.profile, self.requests = backend, profile, 0
        if count_tokens is None:
            from benchmarks.agent_study.transformers_chat import tokenize_chat
            self.count_tokens = lambda messages: len(tokenize_chat(backend.chat.tokenizer,messages)['input_ids'])
            if native_identity(backend.chat.model.config,backend.chat.tokenizer) != profile['native']:
                raise ValueError('loaded native/tokenizer configuration differs from context profile')
        else:
            self.count_tokens = count_tokens
        if emit is None:
            from fin_skills.rsi.meter import event
            emit = event
        self.emit = emit

    def __call__(self, messages):
        raise RuntimeError('bounded generation required')

    def describe(self):
        return self.backend.describe()

    def generate_bounded(self, messages, *, max_new_tokens, seed):
        # Match the unchanged bounded backend's cap exactly. Never shrink it for context.
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError('positive requested output cap required')
        output = min(max_new_tokens,self.backend.describe()['max_new_tokens'])
        count = self.count_tokens(messages)
        if type(count) is not int or count < 1:
            raise ValueError('exact positive tokenizer count required')
        cap = self.profile['native']['effective_tokens']
        fits = count+output <= cap
        self.emit('context_checked', backend_mode='context_guard', request_index=self.requests,
            messages_sha256=digest(messages), seed=seed, requested_max_new_tokens=max_new_tokens,
            prompt_tokens=count, output_cap=output, effective_context_tokens=cap,
            forwarded=fits, known_generated_tokens=None if fits else 0)
        self.requests += 1
        if not fits:
            raise ContextIncompatibility(f'complete prompt {count} + output {output} exceeds native ceiling {cap}; provider not called')
        return self.backend.generate_bounded(messages,max_new_tokens=max_new_tokens,seed=seed)


def guard_accounting(path, *, effective_tokens=None):
    """Prove non-forwarded zero generation separately; never rewrite the original Meter ledger."""
    path = Path(path)
    if not path.exists():
        return dict(available=False, checked_requests=None, proven_zero_generation_requests=None)
    active, checked, zero, missing, incomplete = None, 0, 0, 0, False
    indices = set()
    for raw in path.read_bytes().splitlines(keepends=True):
        if not raw.endswith(b'\n'):
            incomplete = True
            break
        event = json.loads(raw)
        if event.get('backend_mode') == 'bounded' and event['kind'] == 'call_started':
            if active is not None:
                raise ValueError('overlapping bounded requests in context ledger')
            active = dict(start=event,check=None,response=False)
        elif event['kind'] == 'context_checked':
            if active is None or active['check'] is not None:
                raise ValueError('context check without exactly one bounded request')
            start = active['start']
            if (event['messages_sha256'] != digest(start['messages']) or event['seed'] != start['seed']
                    or event['requested_max_new_tokens'] != start['max_new_tokens']):
                raise ValueError('context check is bound to a different complete request')
            if (type(event['forwarded']) is not bool or type(event['prompt_tokens']) is not int
                    or event['prompt_tokens'] < 1 or event['output_cap'] != min(2048,start['max_new_tokens'])
                    or (effective_tokens is not None and event['effective_context_tokens'] != effective_tokens)
                    or event['request_index'] in indices):
                raise ValueError('context check has invalid cap/count or duplicate request identity')
            indices.add(event['request_index'])
            active['check'] = event
            checked += 1
        elif event['kind'] == 'response_received' and active is not None:
            active['response'] = True
        elif event.get('backend_mode') == 'bounded' and event['kind'] in ('call_failed','call_completed'):
            if active is None:
                raise ValueError('context ledger outcome without request')
            check = active['check']
            if check is None:
                missing += 1
            elif not check['forwarded']:
                if (active['response'] or event['kind'] != 'call_failed'
                        or event['error'] != 'ContextIncompatibility'
                        or check['known_generated_tokens'] != 0
                        or check['prompt_tokens']+check['output_cap'] <= check['effective_context_tokens']):
                    raise ValueError('claimed zero generation lacks non-forwarding proof')
                zero += 1
            elif check['prompt_tokens']+check['output_cap'] > check['effective_context_tokens']:
                raise ValueError('forwarded request violates the context contract')
            active = None
    return dict(available=True,checked_requests=checked,proven_zero_generation_requests=zero,
        missing_checks=missing,unfinished_request=active is not None,incomplete_final_event=incomplete,
        interpretation='Zero claims require a matching guard rejection and failed caller outcome; original Meter unknown costs are retained separately')
