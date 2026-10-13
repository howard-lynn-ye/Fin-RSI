"""CPU-only actual-tokenizer diagnosis; no inference and no trade selection."""
import json
from pathlib import Path
import sys
import torch
from transformers import GenerationConfig

B = Path('/beacon-projects/radfm/wy891/fin-multisource-models-20261005')
sys.path.insert(0, str(B/'algorithm-discovery-20261013-candidate20/source'))
from benchmarks.agent_study import trading_schema_v8 as schema
from benchmarks.agent_study.transformers_chat import load_tokenizer
from benchmarks.agent_study.trading_submission_v8 import extract_decision

record = json.loads((B/'v9-access-matrix-20261013-r8/pairs/mistral12b-37/infrastructure-failures/preflight-library-43.json').read_text())
text = record['response']['choices'][0]['message']['content']
parser = schema.parser_factory()
for i, c in enumerate(text):
    if c not in parser.get_allowed_characters():
        print(json.dumps(dict(first_disallowed_character=i, codepoint=ord(c))), flush=True)
        break
    parser = parser.add_character(c)
model = record['response']['backend_metadata']['identity']
cache = '/beacon-projects/radfm/wy891/fin-skills-campaign-model-transfer-20260923-v1/model-cache/hub'
tokenizer = load_tokenizer(model['model'], model['revision'], cache_dir=cache, local_files_only=True)
data = schema.prepare_decoder(tokenizer, [tokenizer.eos_token_id])
prompt = tokenizer.apply_chat_template(record['request_messages'], tokenize=False, add_generation_prompt=True)
fn = schema.prefix_filter(data, prompt)
sequence = tokenizer.encode(prompt, add_special_tokens=False)
tokens = tokenizer.encode(text, add_special_tokens=False)
blocked = None
for i, token in enumerate(tokens):
    if token not in fn(0, torch.tensor(sequence)):
        blocked = dict(index=i, token=token, text=tokenizer.decode([token]))
        break
    sequence.append(token)
print(json.dumps(dict(reencoded_filter_blocked=blocked, token_count=len(tokens),
    reconstructed_exact=tokenizer.decode(tokens, skip_special_tokens=False)==text)), flush=True)
for arm in ('library',):
    record = json.loads((B/f'v9-access-matrix-20261013-r8/pairs/falcon3-10b-11/model-access/{arm}.json').read_text())
    for i, turn in enumerate(record['turns']):
        if turn.get('tool') != 'research_context':
            continue
        call, parse = extract_decision(turn['response'])
        query = (call or {}).get('arguments', {}).get('query')
        print(json.dumps(dict(falcon_turn=i, parse=parse,
            query_type=type(query).__name__, length=len(query) if isinstance(query, str) else None,
            query=query[:180] if isinstance(query, str) else query,
            response_prefix=turn['response'][:200] if call is None else None)), flush=True)
