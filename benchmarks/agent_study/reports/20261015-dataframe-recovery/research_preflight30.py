"""Run only real-model interface qualification, never a return experiment."""
import json
from pathlib import Path
import sys

from benchmarks.agent_study import multisource_model_study as study
from benchmarks.agent_study import trading_preflight_v8 as preflight
from benchmarks.agent_study.qualify_model_matrix import prompts
from benchmarks.agent_study.transformers_chat import TransformersChat, context_limit, load_tokenizer, tokenize_chat
from transformers import AutoConfig

root = Path(sys.argv[1])
p = study.verify(root)
assert p['interaction_revision'] == '20261013-nested-receipts-v9'
name, revision = p['model'][1:]
tokenizer = load_tokenizer(name, revision=revision, local_files_only=True)
config = AutoConfig.from_pretrained(name, revision=revision, local_files_only=True)
limit = context_limit(config, tokenizer)
counts = {'raw': [], 'library': []}
for arm, messages in prompts(root):
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    assert all(m['content'] in rendered for m in messages), 'template dropped content'
    tokens = tokenize_chat(tokenizer, messages)['input_ids']
    assert tokens == tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    assert len(tokens) + p['max_tokens'] <= limit, 'context does not fit'
    counts[arm].append(len(tokens))
study.old.write(root / 'template-preflight.json', dict(protocol_sha256=study.old.sha(root / 'protocol.json'),
    context_limit=limit, max_initial_tokens={a:max(c) for a,c in counts.items()}))
backend = TransformersChat(name, revision, max_tokens=p['max_tokens'])
assert not any(str(v) in ('cpu','disk') for v in backend.model.hf_device_map.values()), 'Insufficient GPU memory; no silent CPU offload'
failure = None
try:
    preflight.run(root, backend)
except ValueError as exc:
    if not (root / 'decision-preflight.json').exists():
        public = dict(model=p['model'], passed=False, failure=str(exc),
            access=study.old.load(root/'model-access-preflight.json') if (root/'model-access-preflight.json').exists() else None)
        study.old.write(root/'public-preflight.json', public)
        print(json.dumps(public), flush=True)
        raise
    failure = str(exc)
report = study.old.load(root / 'decision-preflight.json')
details = []
for row in report['cases']:
    record = study.old.load(root / row['record'])
    last = record['turns'][-1]
    details.append(dict(row, final_parse=last['parse'], final_finish_reason=last['finish_reason'],
        accepted_target=record['target'], final_error=last['result'].get('error'),
        tool_calls=record['tool_calls']))
public = dict(model=p['model'], seed=p['seed'], interaction_revision=p['interaction_revision'],
    protocol_sha256=study.old.sha(root / 'protocol.json'), final_output=p['final_output'], decoder_check=report['decoder_check'], passed=report['passed'], failure=failure,
    access=study.old.load(root / 'model-access-preflight.json'),
    cases=details, interpretation=report['interpretation'], scope='Three calendar positions, both arms; independent all-cash scenarios. '
                        'Interface test only, no Return Rate or full-period completion.')
study.old.write(root / 'public-preflight.json', public)
print(json.dumps(public), flush=True)
if failure:
    raise SystemExit(2)

review = preflight.campaign_readiness(report['cases'],
    require_library_use=p['model'][0] in ('qwen-general-7b', '32b'))
study.old.write(root/'research-readiness.json', review)
print(json.dumps(review), flush=True)
if not review['ready']:
    raise SystemExit(3)