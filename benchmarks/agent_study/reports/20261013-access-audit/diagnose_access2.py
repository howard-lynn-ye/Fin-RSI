"""Read original completed records without changing their inputs or scores."""
from collections import Counter
import hashlib
import json
import re
from pathlib import Path

from benchmarks.agent_study import multisource_model_study as study
from benchmarks.agent_study import trading_runtime_v9 as runtime
from benchmarks.agent_study.trading_submission_v8 import extract_decision
from benchmarks.agent_study.transformers_chat import load_tokenizer, tokenize_chat

B = Path('/beacon-projects/radfm/wy891/fin-multisource-models-20261005')
batch = B / 'v9-context-model-preflight-20261013-r6'
results = []
for family in ('7b', 'granite-8b', 'qwen3-4b'):
    root = batch / (family + '-11')
    p = study.verify(root)
    study.validate_receipt(root)
    tokenizer = load_tokenizer(*p['model'][1:], local_files_only=True) if family == '7b' else None
    packets = study.old.load(root / 'evidence-packets.json')
    arms = {}
    for arm in ('raw', 'library'):
        tools, responses, early, counts, errors = Counter(), Counter(), 0, [], []
        for path in sorted((root / 'decisions' / arm).glob('*.json')):
            record = study.old.load(path)
            runtime.validate_decision(record, expected_final_output=p['final_output'])
            expected = [dict(role='system', content=runtime.system_prompt(arm)),
                        dict(role='user', content=runtime.task(record['date'], record['holdings_before'],
                            p['deadline']) + study.evidence_text(packets[record['date']]))]
            actual = record['initial_request']
            assert actual[0] == expected[0], 'system prompt differs'
            pattern = r'(Current holdings as fractions of equity: )(\{[^\n]+?\})(\. Cost:)'
            got, wanted = re.search(pattern, actual[1]['content']), re.search(pattern, expected[1]['content'])
            assert got and wanted, 'holdings display absent'
            got_values, wanted_values = json.loads(got[2]), json.loads(wanted[2])
            assert got_values.keys() == wanted_values.keys()
            assert max(abs(got_values[k] - wanted_values[k]) for k in got_values) < 1e-12
            # Saved JSON sorts dictionary keys. That changes their display order
            # when reconstructing the original task; preserve the actual display
            # after comparing its values, then verify every other character.
            expected[1]['content'] = expected[1]['content'].replace(wanted[2], got[2], 1)
            assert actual == expected, 'initial prompt differs beyond serialized holdings order'
            assert record['turns'][0]['input_messages_sha256'] == hashlib.sha256(runtime.encoded(expected).encode()).hexdigest()
            early += len(record['turns']) == 1 and record['decision_completed']
            if tokenizer is not None:
                rendered = tokenizer.apply_chat_template(expected, tokenize=False, add_generation_prompt=True)
                assert all(m['content'] in rendered for m in expected), 'template omitted message'
                ids = tokenize_chat(tokenizer, expected)['input_ids']
                assert ids == tokenizer.apply_chat_template(expected, tokenize=True, add_generation_prompt=True)
                assert record['turns'][0]['usage']['prompt_tokens'] == len(ids), 'actual token receipt differs'
                counts.append(len(ids))
            for t in record['turns']:
                responses[t.get('tool') or t['parse']] += 1
                if not t['result'].get('ok'):
                    call, _ = extract_decision(t['response'])
                    args = (call or {}).get('arguments') or {}
                    compact = {k: str(v)[:160] for k, v in args.items()
                               if k in ('name', 'algorithm_id', 'category', 'id', 'query', 'start', 'end')}
                    errors.append(dict(date=record['date'], tool=t.get('tool'), arguments=compact,
                        error=str(t['result'].get('error', ''))[:400], record_sha256=study.old.sha(path)))
            tools.update(c['tool'] for c in record['tool_calls'])
        arms[arm] = dict(records=len(list((root/'decisions'/arm).glob('*.json'))),
            early_first_response_decisions=early, emitted_tools=responses, controller_calls=tools,
            initial_prompts_exactly_match=True,
            actual_template_and_token_receipts_verified=tokenizer is not None,
            initial_token_range=[min(counts), max(counts)] if counts else None,
            research_entry_in_system=('research_context' in runtime.system_prompt(arm)), errors=errors)
    results.append(dict(model=p['model'], protocol_sha256=study.old.sha(root/'protocol.json'), arms=arms))
out = B / 'v9-structured-cycle-public-20261013' / 'access-diagnosis.json'
study.old.write(out, dict(results=results,
    scope='Original complete traces and exact initial prompts; no regeneration or outcome-based selection. '
          'A zero-call trace proves non-attempt, not inability or a causal explanation of the model behavior.'))
print(out.read_text(), flush=True)
