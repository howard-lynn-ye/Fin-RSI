"""CPU-only token filter qualification with the three actual pinned tokenizers."""
import json
from pathlib import Path
import os
import sys

import torch
from transformers import GenerationConfig
from benchmarks.agent_study import multisource_model_study as study, trading_schema_v8 as schema
from benchmarks.agent_study.transformers_chat import load_tokenizer, validate_reasoning_tokens

B=Path('/beacon-projects/radfm/wy891/fin-multisource-models-20261005')
caches={'7b': Path('/beacon-projects/radfm/wy891/fin-skills-audit-20260921/cache/hf/hub'),
        'granite-8b': B/'matrix-weights-v1/cache/hf/hub',
        'qwen3-4b': B/'matrix-weights-v2/cache/hf/hub'}
rows=[]
for family in ('7b','granite-8b','qwen3-4b'):
    _, model, revision=next(row for row in study.MODELS if row[0]==family)
    cache=caches[family]
    tokenizer=load_tokenizer(model,revision,cache_dir=str(cache),local_files_only=True)
    validate_reasoning_tokens(tokenizer)
    generation=GenerationConfig.from_pretrained(model,revision=revision,cache_dir=str(cache),local_files_only=True)
    stops=generation.eos_token_id
    stops=stops if isinstance(stops,list) else [tokenizer.eos_token_id if stops is None else stops]
    data=schema.prepare_decoder(tokenizer,stops)
    prompt=tokenizer.apply_chat_template([{'role':'user','content':'Choose your own final decision.'}],
        tokenize=False,add_generation_prompt=True)
    prefix=tokenizer.encode(prompt,add_special_tokens=False)
    good=[{'tool':'hold','arguments':{}},
          {'tool':'submit','arguments':{'allocation':{'CASH':1}}},
          {'arguments':{'allocation':{'SPY':3,'GLD':8,'CASH':2}},'tool':'submit'},
          {'tool':'submit','arguments':{'allocation':{**{t:1 for t in study.md.TICKERS},'CASH':0}}}]
    for answer in good:
        text=json.dumps(answer,separators=(',',':'))
        fn=schema.prefix_filter(data,prompt)
        sequence=list(prefix)
        for token in tokenizer.encode(text,add_special_tokens=False):
            assert token in fn(0,torch.tensor(sequence)), (family,'valid text blocked',text)
            sequence.append(token)
        assert tokenizer.eos_token_id in fn(0,torch.tensor(sequence)), (family,'EOS blocked')
    bad=['{"tool":"submit","arguments":{"allocation":{"SPY":0.3},"CASH":0}}',
         '{"tool":"submit","arguments":{"allocation":{"SPY":0.35},"CASH":0}}',
         '{"tool":"hold","arguments":{}} extra text',
         '{"tool":"hold","arguments":{}} {"tool":"hold","arguments":{}}']
    for text in bad:
        fn=schema.prefix_filter(data,prompt)
        sequence=list(prefix)
        rejected=False
        for token in tokenizer.encode(text,add_special_tokens=False):
            if token not in fn(0,torch.tensor(sequence)):
                rejected=True
                break
            sequence.append(token)
        assert rejected, (family,'malformed response survived',text)
    rows.append(dict(model=model,revision=revision,accepted=len(good),rejected=len(bad),
                     eos_token_id=tokenizer.eos_token_id,stop_token_ids=stops))
    print(json.dumps(rows[-1]),flush=True)
study.old.write(Path(sys.argv[1]),dict(passed=True,final_output=schema.output_contract(True),models=rows,
    scope='Actual pinned tokenizer prefix filtering and EOS, CPU only; no model inference or returns.'))
