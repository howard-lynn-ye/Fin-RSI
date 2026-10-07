import collections
import json
import os
from pathlib import Path

B = Path('/beacon-projects/radfm/wy891/fin-multisource-models-20261005')
verified = json.loads((B/'verified-status-1961154.json').read_text())
groups = {}
for row in verified['rows']:
    source = Path(row['source'])
    root = source.parent / ('batch' if source.parent.name == 'v7-models-v1' else 'pairs') / row['pair']
    item = groups.setdefault(row['family'], {})
    for arm in ('raw','library'):
        out = item.setdefault(arm, dict(decisions=0, submitted=0, turns=0,
            parse=collections.Counter(), finish=collections.Counter(),
            tool=collections.Counter(), successful_tool=collections.Counter(),
            completion_tokens=0, reasoning_marked_responses=0))
        for file in sorted((root/'decisions'/arm).glob('*.json')):
            record = json.loads(file.read_text())
            out['decisions'] += 1
            out['submitted'] += bool(record['submitted'])
            for turn in record['turns']:
                out['turns'] += 1
                out['parse'][str(turn.get('parse'))] += 1
                out['finish'][str(turn.get('finish_reason'))] += 1
                out['tool'][str(turn.get('tool'))] += 1
                if turn.get('result',{}).get('ok'):
                    out['successful_tool'][str(turn.get('tool'))] += 1
                out['completion_tokens'] += (turn.get('usage') or {}).get('completion_tokens',0)
                text = turn.get('response') or ''
                out['reasoning_marked_responses'] += ('<think>' in text or '</think>' in text)
report = dict(based_on='verified-status-1961154.json', groups=groups,
    scope='Counts from recorded responses. Budget and syntax failures are not a causal attribution of return gaps.')
dest = B/f'response-diagnostics-{os.environ["SLURM_JOB_ID"]}.json'
with dest.open('x') as f:
    json.dump(report,f,indent=2)
print(json.dumps(report,indent=2),flush=True)
