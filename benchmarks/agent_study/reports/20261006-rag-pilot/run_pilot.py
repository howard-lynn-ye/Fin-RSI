"""One predeclared paired development date; no scoring or return-based selection."""
import json
from pathlib import Path
import sys

import torch
from benchmarks.agent_study import multisource_model_study as study
from benchmarks.agent_study import trading_runtime_v9 as runtime
from benchmarks.agent_study.qualify_model_matrix import prompts
from benchmarks.agent_study.transformers_chat import TransformersChat, tokenize_chat

root = Path(sys.argv[1])
p = study.verify(root)
assert p['interface'] == 'v9'
runtime.require_qualification(root)
out = root / 'gpu-pilot'
out.mkdir(exist_ok=False)
study.old.write(out / 'started.json', {
    'protocol_sha256': study.old.sha(root / 'protocol.json'),
    'script_sha256': study.old.sha(Path(__file__)),
    'scope': 'one first-date paired usability probe, not Return Rate or a completed model',
    'counts_toward_twenty_models': False,
})
backend = TransformersChat(p['model'][1], p['model'][2], max_tokens=p['max_tokens'], seed=p['seed'])
assert not any(str(v) in ('cpu', 'disk') for v in backend.model.hf_device_map.values())
counts = []
for arm, messages in prompts(root):
    rendered = backend.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    assert all(m['content'] in rendered for m in messages)
    ids = tokenize_chat(backend.tokenizer, messages)['input_ids']
    assert ids == backend.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    assert len(ids) + p['max_tokens'] <= backend.context_limit
    counts.append(len(ids))
inputs = study.old.load(root / 'inputs.json')
packets = study.old.load(root / 'evidence-packets.json')
records = study.old.load(root / 'evidence.json')
day = inputs['dates'][0]
summary = {}
for arm in ('raw', 'library'):
    visible = out / arm
    study.md.truncate(root / 'data', visible, day)
    runtime.write_evidence_snapshot(records, packets[day], visible)
    controller = runtime.Controller(out, visible, arm, p['seed'] * 1000)
    backend.seed, backend.calls = p['seed'] * 1000, 0
    holdings = dict.fromkeys(study.md.TICKERS, 0.)
    record = runtime.decide(backend, controller,
        runtime.task(day, holdings, p['deadline']) + study.evidence_text(packets[day]))
    runtime.validate_decision(record)
    study.old.write(out / f'{arm}.json', record)
    summary[arm] = dict(action=record['decision_action'], completed=record['decision_completed'],
        turns=len(record['turns']), failed_turns=sum(not t['turn_ok'] for t in record['turns']),
        tool_calls=[t['tool'] for t in record['turns']],
        record_sha256=study.old.sha(out / f'{arm}.json'))
    print(json.dumps({'family': p['model'][0], 'arm': arm, **summary[arm]}), flush=True)
result = dict(model=p['model'], seed=p['seed'], date=day, probe=summary,
              max_initial_tokens=max(counts), gpu=torch.cuda.get_device_name(0),
              protocol_sha256=study.old.sha(root / 'protocol.json'),
              script_sha256=study.old.sha(Path(__file__)),
              scope='No scores computed; retain failed decisions. Not a complete return experiment.',
              counts_toward_twenty_models=False)
study.old.write(out / 'completed.json', result)
