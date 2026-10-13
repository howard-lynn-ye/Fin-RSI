"""Verify immutable preflight records and publish only compact outcome evidence."""
import json
from pathlib import Path
import sys

import pandas as pd
from benchmarks.agent_study import multisource_model_study as study
from benchmarks.agent_study import trading_preflight_v8 as gate

batch = Path(sys.argv[1])
models = []
for family in ('7b', 'granite-8b', 'qwen3-4b'):
    root = batch / (family + '-11')
    if not (root / 'public-preflight.json').exists():
        models.append(dict(family=family, status='infrastructure_or_incomplete', passed=False))
        continue
    p = study.verify(root)
    runtime = study.terminal_runtime(p['interface'])
    runtime.require_qualification(root)
    report = study.old.load(root / 'decision-preflight.json')
    public = study.old.load(root / 'public-preflight.json')
    inputs = study.old.load(root / 'inputs.json')
    dates = pd.read_csv(root / 'data' / study.md.HIDDEN, index_col=0).index
    assert [(r['arm'], r['decision_index']) for r in report['cases']] == gate.cases(inputs['picks'])
    assert report['protocol_sha256'] == public['protocol_sha256'] == study.old.sha(root / 'protocol.json')
    assert report['interpretation'] == public['interpretation'] == gate.interpretation(report['cases'])
    assert public['passed'] == report['passed'] == all(r['decision_completed'] for r in report['cases'])
    for row, delivered in zip(report['cases'], public['cases'], strict=True):
        path = root / 'interface-preflight' / f"{row['arm']}-{row['decision_index']:02d}.json"
        record = study.old.load(path)
        runtime.validate_decision(record)
        assert row['record'] == path.relative_to(root).as_posix()
        assert row['sha256'] == study.old.sha(path)
        assert record['protocol_sha256'] == report['protocol_sha256']
        assert record['arm'] == row['arm']
        assert record['decision_index'] == row['decision_index']
        assert record['date'] == row['date'] == str(dates[inputs['picks'][row['decision_index']]])
        assert all(row[k] == v for k, v in gate.observations(record).items())
        assert all(delivered[k] == v for k, v in row.items())
        assert delivered['accepted_target'] == record['target']
        assert delivered['tool_calls'] == record['tool_calls']
        errors = [dict(phase=t['phase'], parse=t['parse'], tool=t['tool'],
                       error=t['result'].get('error'), finish=t['finish_reason'])
                  for t in record['turns'] if not t['turn_ok']]
        print(json.dumps(dict(family=family, arm=row['arm'], index=row['decision_index'],
                              errors=errors)), flush=True)
        if not record['decision_completed']:
            print('PRIVATE FAILED RESPONSE', family, row['arm'], row['decision_index'],
                  record['turns'][-1]['response'], flush=True)
    if report['passed']:
        gate.require(root)
    arms = {}
    for arm in ('raw', 'library'):
        rows = [r for r in report['cases'] if r['arm'] == arm]
        arms[arm] = dict(cases=len(rows), completed=sum(r['decision_completed'] for r in rows),
            research_attempts=sum(r['research_attempts'] for r in rows),
            research_successes=sum(r['research_successes'] for r in rows),
            research_errors=sum(r['research_errors'] for r in rows),
            library_execution_cases=sum(r['library_execution_observed'] for r in rows),
            parse_failures=sum(r['parse_failures'] for r in rows),
            truncated_responses=sum(r['truncated_responses'] for r in rows))
    models.append(dict(family=family, model=p['model'], passed=report['passed'],
        status='verified_original_records', arms=arms,
        public_report_sha256=study.old.sha(root / 'public-preflight.json'),
        protocol_sha256=study.old.sha(root / 'protocol.json'),
        interpretation=report['interpretation']))
out = dict(interaction_revision='20261013-single-action-envelope', models=models,
    original_records_verified=all(m['status'] == 'verified_original_records' for m in models),
    all_decision_gates_passed=all(m['passed'] for m in models),
    return_rate_available=False, historical_availability_verified=False,
    scope='Fixed three-date independent all-cash scenarios, both arms, seed11. '
          'No full-cycle Return Rate, causal utility or full-capability claim.')
study.old.write(batch / 'verified-summary.json', out)
print('PUBLIC SUMMARY', json.dumps(out), flush=True)
