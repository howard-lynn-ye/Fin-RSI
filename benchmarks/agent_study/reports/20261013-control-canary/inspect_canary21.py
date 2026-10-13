"""Read-only qualification progress on Slurm, excluding prompts/source prose."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

B = Path('/beacon-projects/radfm/wy891/fin-multisource-models-20261005')
S = B/'control-filter-20261013-candidate21/source'
sys.path.insert(0, str(S))
from benchmarks.agent_study import multisource_model_study as study
from benchmarks.agent_study import trading_preflight_v8 as preflight
from benchmarks.agent_study import trading_schema_v8 as schema

rows = []
for family in ('mistral12b', 'falcon3-10b'):
    root = B/'control-canary-20261013-r21/pairs'/f'{family}-11'
    if not (root/'protocol.json').is_file():
        rows.append(dict(family=family, state='not_started'))
        continue
    p = study.verify(root)
    row = dict(family=family, model=p['model'], state='in_progress',
        protocol_sha256=study.old.sha(root/'protocol.json'),
        interaction_revision=p['interaction_revision'],
        access=None, cases=[], infrastructure_failures=[], errors=[])
    access = root/'model-access-preflight.json'
    if access.exists():
        row['access'] = study.old.load(access)
    for path in sorted((root/'interface-preflight').glob('*.json')):
        r = study.old.load(path)
        try:
            study.terminal_runtime(p['interface']).validate_decision(r,
                expected_final_output=p['final_output'],
                expected_research_output=p['research_output'][r['arm']])
            schema.check_backend(r, p)
        except Exception as exc:
            row['errors'].append(dict(record=path.name, error=f'{type(exc).__name__}: {exc}'))
            continue
        row['cases'].append(dict(record=path.name, sha256=study.old.sha(path),
            arm=r['arm'], date=r['date'], decision_index=r['decision_index'],
            observations=preflight.observations(r),
            tools=[dict(tool=t.get('tool'), parse=t['parse'], ok=t['result'].get('ok'),
                        error=t['result'].get('error') if not t['result'].get('ok') else None)
                   for t in r['turns']]))
    for path in sorted((root/'infrastructure-failures').glob('*.json')):
        r = study.old.load(path)
        row['infrastructure_failures'].append(dict(record=path.name,
            sha256=study.old.sha(path), status=r.get('status')))
    if (root/'decision-preflight.json').exists():
        try:
            preflight.require(root)
            row['state'] = 'passed'
        except Exception as exc:
            row['state'] = 'failed'
            row['errors'].append(dict(error=f'{type(exc).__name__}: {exc}'))
    if row['access'] and not row['access'].get('passed'):
        row['state'] = 'failed_access'
    if row['infrastructure_failures']:
        row['state'] = 'infrastructure_failure'
    rows.append(row)
out = dict(observed_utc=datetime.now(timezone.utc).isoformat(), models=rows,
    scope='Three independent calendar positions per arm; interface qualification only. '
          'No cumulative Return Rate or complete-cycle result. Access probe is instructed; '
          'use during investment cases remains autonomous.')
if len(sys.argv) > 1:
    study.old.write(Path(sys.argv[1]), out)
print(json.dumps(out), flush=True)
