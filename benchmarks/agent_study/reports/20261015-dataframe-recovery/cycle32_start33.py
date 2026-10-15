"""Verify existing 32B qualification without making any new interface-model calls."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from benchmarks.agent_study import multisource_model_study as study
from benchmarks.agent_study import trading_preflight_v8 as preflight

B = Path('/beacon-projects/radfm/wy891/fin-multisource-models-20261005')
M = Path('/beacon-projects/radfm/wy891/fin-manual-multisource-20261005')
S = B/'library-contract-audit-20261013-candidate29/source'
R = B/'research-canary-20261014-r30/pairs/32b-11'
O = B/'full32-cycle-20261015-r33'
assert Path(study.__file__).resolve().is_relative_to(S)
verification = study.old.load(M/'fin-context-source-verification30.json')
assert all(study.old.sha(S/rel) == sha for rel, sha in verification['sha256'].items())
p = study.verify(R)
assert p['model'][0] == '32b' and p['seed'] == 11
assert p['interaction_revision'] == '20261013-nested-receipts-v9'
assert p['batch'] == 'matrix_hf'
preflight.require_campaign_ready(R, require_library_use=True)
assert not (R/'inference-started.json').exists()
assert not list((R/'infrastructure-failures').glob('*.json'))
out = dict(job=os.environ['SLURM_JOB_ID'],started_utc=datetime.now(timezone.utc).isoformat(),
    model=p['model'],seed=p['seed'],source=str(S),root=str(R),protocol_sha256=study.old.sha(R/'protocol.json'),
    merged_commit='f36c3f63e317e6990b6692f6771bdb333b72b181',
    preflight_sha256=study.old.sha(R/'decision-preflight.json'),
    mode='Full raw/library cycle using unchanged passed source29; no new interface-model calls.',
    window=p['window'],decisions_per_arm=p['decisions_per_arm'],initial_capital=p['initial_capital'],
    primary_metric='100 * (ending capital / initial capital - 1), after modeled costs',
    paid_calls=False, seven_b_run=False, rsi_enabled=False)
with (O/'started.json').open('x') as f:
    json.dump(out,f,indent=2)
print(json.dumps(out),flush=True)
