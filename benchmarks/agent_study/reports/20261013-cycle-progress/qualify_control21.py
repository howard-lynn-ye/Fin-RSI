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
rows = []
for family, cache in caches.items():
    _, model, revision = next(row for row in study.MODELS if row[0] == family)
    tokenizer = load_tokenizer(model, revision, cache_dir=str(cache), local_files_only=True)
    data = schema.prepare_decoder(tokenizer, [tokenizer.eos_token_id])
    prompt = tokenizer.apply_chat_template([dict(role='user', content='Choose your own decision.')],
        tokenize=False, add_generation_prompt=True)
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    cases = []
    for stage in ('final', 'research'):
        texts = [
            (json.dumps(dict(tool='hold', arguments={})), True),
            (json.dumps(dict(tool='submit', arguments=dict(
                allocation=dict(SPY=3, GLD=2, CASH=1), rationale='escaped\tline\nend\x00'))), True),
            (observed, False),
            ('{"tool":"submit","arguments":{"allocation":{"CASH":1},"rationale":"a\t"}}', False),
            ('{"tool":"submit","arguments":{"allocation":{"CASH":1},"rationale":"a\x00"}}', False)]
        if stage == 'research':
            texts += [(json.dumps(dict(tool='research_context', arguments=dict(query='news\tpolicy'))), True),
                      ('{"tool":"research_context","arguments":{"query":"news\tpolicy"}}', False)]
        for text, wanted in texts:
            fn = (schema.prefix_filter(data, prompt) if stage == 'final' else
                  research.prefix_filter(data, prompt, research.output_contract('library')))
            sequence, accepted, blocked_token_index = list(prefix), True, None
            for i, token in enumerate(tokenizer.encode(text, add_special_tokens=False)):
                if token not in fn(0, torch.tensor(sequence)):
                    accepted, blocked_token_index = False, i
                    break
                sequence.append(token)
            if accepted:
                accepted = tokenizer.eos_token_id in fn(0, torch.tensor(sequence))
            assert accepted == wanted, (family, stage, wanted, blocked_token_index)
            cases.append(dict(stage=stage, text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                expected_accepted=wanted, accepted=accepted, blocked_token_index=blocked_token_index))
    row = dict(family=family, model=model, revision=revision, cases=cases, passed=True)
    rows.append(row)
    print(json.dumps(dict(family=family, cases=len(cases), passed=True)), flush=True)
result = dict(passed=True, source=str(source), final_output=schema.output_contract(True),
    research_output=research.output_contract('library'), observed_failure_sha256=study.old.sha(bad_path),
    models=rows, scope='CPU-only token filtering; no financial decision or model call. '
    'Actual inference and returns remain separate evidence.')
study.old.write(Path(sys.argv[1]), result)
