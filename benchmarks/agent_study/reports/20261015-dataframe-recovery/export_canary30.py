"""Verify original pilot records and export decisions without third-party source prose."""
from datetime import datetime, timezone
import json
from pathlib import Path
from benchmarks.agent_study import multisource_model_study as study
from benchmarks.agent_study import trading_preflight_v8 as preflight

B = Path('/beacon-projects/radfm/wy891/fin-multisource-models-20261005')
R = B/'research-canary-20261014-r30'
models = []
for family in ('qwen-general-7b', '32b'):
    root = R/'pairs'/f'{family}-11'
    protocol = study.verify(root)
    report = preflight.require(root)
    review = preflight.campaign_readiness(report['cases'], require_library_use=True)
    records = []
    for row in report['cases']:
        path = root/row['record']
        original = study.old.load(path)
        turns = []
        for t in original['turns']:
            kept = {k:t[k] for k in ('phase','tool','parse','response','finish_reason','backend_metadata') if k in t}
            kept['result'] = {k:t['result'][k] for k in ('ok','error','submission','truncated','contract','hint',
                'recovery_call','nested_tool_summary','nested_tool_note','conversion','cash_weight',
                'allocation_input','weights_input','budget_rounding_applied') if k in t['result']}
            kept['result_sha256'] = __import__('hashlib').sha256(
                json.dumps(t['result'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
            turns.append(kept)
        records.append(dict(original_record=row['record'], original_sha256=study.old.sha(path),
            date=original['date'], arm=original['arm'], decision_index=original['decision_index'],
            decision_action=original['decision_action'], decision_completed=original['decision_completed'],
            target=original['target'], tool_calls=original['tool_calls'], turns=turns))
    models.append(dict(family=family,model=protocol['model'],protocol_sha256=study.old.sha(root/'protocol.json'),
        interaction_revision=protocol['interaction_revision'],seed=protocol['seed'],readiness=review,
        report=report,access=study.old.load(root/'model-access-preflight.json'),records=records))
out=dict(observed_utc=datetime.now(timezone.utc).isoformat(), job=2180196, models=models,
    scope='All 12 independent pilot decisions, not a continuous account or Return Rate.',
    disclosure='Exports preserve model-generated requests, decisions, errors and execution receipts. '
        'Source prose, tool stdout, prompts and feedback text are omitted from publication; original '
        'hash-bound records remain on Beacon. No scoring labels or market datasets are exported.')
with (R/'public-audit31.json').open('x') as f:
    json.dump(out,f,indent=2)
print(json.dumps(dict(models=[dict(family=m['family'],ready=m['readiness']['ready'],
    reasons=m['readiness']['reasons'],decisions=len(m['records'])) for m in models])))
