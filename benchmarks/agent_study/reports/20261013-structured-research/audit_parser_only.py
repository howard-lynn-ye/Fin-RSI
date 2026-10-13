"""Classify saved text only; never execute or rescore an old decision."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from benchmarks.agent_study.trading_submission_v8 import extract_decision
assert os.environ.get('SLURM_JOB_ID')
B = Path('/beacon-projects/radfm/wy891/fin-multisource-models-20261005')
root = B/'v9-access-matrix-20261013-r8/pairs/qwen-general-7b-11'
rows = []
for arm in ('raw', 'library'):
    counts, accepted = Counter(), []
    for path in sorted((root/'decisions'/arm).glob('*.json')):
        record = json.loads(path.read_text())
        for turn in record['turns']:
            if turn['phase'] != 'research' or turn['result'].get('ok'):
                continue
            call, status = extract_decision(turn['response'])
            eligible = bool(turn['finish_reason'] != 'length' and call and status == 'python-label')
            counts['original_failed_research_responses'] += 1
            counts['now_syntax_eligible_python'] += eligible
            if eligible:
                accepted.append(dict(date=record['date'], turn_index=turn['turn_index'],
                    original_record_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    code_sha256=hashlib.sha256(call['arguments']['code'].encode()).hexdigest()))
    rows.append(dict(arm=arm, counts=counts, eligible_requests=accepted))
output = B/'python-label-diagnosis12.json'
with output.open('x') as stream:
    json.dump(dict(scope='Parser-only compatibility audit. Zero code executions, new model calls or rescored trades.',
        original_root=str(root), rows=rows,
        limitation='Syntax eligibility does not prove semantic validity, successful execution, library use or higher return.'), stream, indent=2)
print(json.dumps(dict(output=str(output), sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
    counts=[dict(arm=r['arm'], **r['counts']) for r in rows])), flush=True)
