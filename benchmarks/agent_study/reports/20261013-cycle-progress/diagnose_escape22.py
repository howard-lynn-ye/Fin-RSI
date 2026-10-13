"""CPU qualification with installed real tokenizers; no model inference."""
import hashlib
import json
from pathlib import Path
import sys
import torch

from benchmarks.agent_study import multisource_model_study as study
from benchmarks.agent_study import trading_schema_v8 as schema, trading_research_v9 as research
from benchmarks.agent_study.transformers_chat import load_tokenizer

B = Path('/beacon-projects/radfm/wy891/fin-multisource-models-20261005')
M = Path('/beacon-projects/radfm/wy891/fin-manual-multisource-20261005')
source = B/'control-filter-20261013-candidate21/source'
assert Path(study.__file__).resolve().is_relative_to(source)
bad_path = B/'v9-access-matrix-20261013-r8/pairs/mistral12b-37/infrastructure-failures/preflight-library-43.json'
observed = json.loads(bad_path.read_text())['response']['choices'][0]['message']['content']
caches = {
    'mistral12b': Path('/beacon-projects/radfm/wy891/fin-skills-campaign-model-transfer-20260923-v1/model-cache/hub'),
    'falcon3-10b': B/'matrix-weights-v2/cache/hf/hub',
    'qwen-general-7b': B/'matrix-weights-v2/cache/hf/hub',
    '32b': Path('/beacon-projects/radfm/wy891/fin-skills-audit-20260921/cache/hf/hub')}
family = 'mistral12b'
_, model, revision = next(row for row in study.MODELS if row[0] == family)
tokenizer = load_tokenizer(model, revision, cache_dir=str(caches[family]), local_files_only=True)
from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data
prompt = tokenizer.apply_chat_template([dict(role='user', content='Choose your own decision.')], tokenize=False, add_generation_prompt=True)
for hardened in (False, True):
    data = build_token_enforcer_tokenizer_data(tokenizer)
    if hardened:
        schema.harden_string_token_cache(data)
    for rationale in ('escaped\tline\nend\x00', 'escaped\tline\nend', 'normal text'):
        text = json.dumps(dict(tool='submit', arguments=dict(allocation=dict(SPY=3, GLD=2, CASH=1), rationale=rationale)))
        tokens = tokenizer.encode(text, add_special_tokens=False)
        fn = schema.prefix_filter(data, prompt)
        sequence = tokenizer.encode(prompt, add_special_tokens=False)
        blocked = None
        for i, token in enumerate(tokens):
            if token not in fn(0, torch.tensor(sequence)):
                blocked = dict(position=i, token=token, decoded=tokenizer.decode([token]), prior=tokenizer.decode(tokens[:i])[-50:])
                break
            sequence.append(token)
        print(json.dumps(dict(hardened=hardened, rationale=rationale, blocked=blocked)), flush=True)