from pathlib import Path
from collections import Counter
import json,sys
from datetime import datetime,timezone
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'source'))
from benchmarks.agent_study import trading_study_v5 as study
from benchmarks.agent_study import trading_study as old
plan=old.load(root/'batch-plan.json')
assert study.sources()==plan['source_sha256']
prepared=old.load(root/'prepared.json')
assert prepared['passed']
coverage=[]
for item in plan['pairs']:
    pair=root/'batch'/f"{item['family']}-{item['seed']}"
    protocol=study.verify(pair)
    expected=next(x for x in prepared['pairs'] if x['pair']==pair.name)
    assert old.sha(pair/'protocol.json')==expected['protocol_sha256']
    if not (pair/'completed.json').exists():
        study.score(pair)
    else:
        assert old.load(pair/'completed.json')['scores_sha256']==old.sha(pair/'scores.json')
    receipt=old.load(pair/'inference-receipt.json')
    assert receipt['protocol_sha256']==old.sha(pair/'protocol.json')
    for name,digest in receipt['decisions'].items():
        assert old.sha(pair/name)==digest
    for arm in ('raw','library'):
        records=[old.load(p) for p in sorted((pair/'decisions'/arm).glob('*.json'))]
        assert len(records)==protocol['decisions_per_arm']
        calls=[c for r in records for c in r['tool_calls']]
        ok_algorithm=lambda r:any(c['tool']=='run_algorithm' and c['ok'] for c in r['tool_calls'])
        bad=Counter(str(t['result'].get('error','unspecified execution/delivery failure'))[:180] for r in records for t in r['turns'] if not t.get('turn_ok',False))
        coverage.append(dict(family=item['family'],seed=item['seed'],arm=arm,decisions=len(records),
            submitted=sum(bool(r['submitted']) for r in records),
            successful_algorithm_calls=sum(c['tool']=='run_algorithm' and c['ok'] for c in calls),
            failed_algorithm_calls=sum(c['tool']=='run_algorithm' and not c['ok'] for c in calls),
            decisions_with_successful_algorithm_call=sum(ok_algorithm(r) for r in records),
            algorithm_used_and_submitted=sum(ok_algorithm(r) and bool(r['submitted']) for r in records),
            successful_skill_reads=sum(c['tool']=='read_skill' and c['ok'] for c in calls),
            successful_guard_calls=sum(c['tool']=='run_guard' and c['ok'] for c in calls),
            generation_cutoff_turns=sum(t.get('finish_reason')=='length' for r in records for t in r['turns']),
            direct_package_reference_turns=sum('fin_skills' in t.get('response','') for r in records for t in r['turns']),
            direct_package_reference_note='Text references only; not proof of executed package calls or valid calculations.',
            failure_messages=dict(bad)))
aggregate=study.aggregate(root/'batch')
assert aggregate['status']=='complete'
assert sum(c['decisions'] for c in coverage)==prepared['total_planned_decisions']
result=dict(status='complete',finished_utc=datetime.now(timezone.utc).isoformat(),
    aggregate_sha256=old.sha(root/'batch'/'aggregate.json'),batch_plan_sha256=old.sha(root/'batch-plan.json'),
    return_rate_pct=aggregate['return_rate_pct'],coverage=coverage,
    smoke_combined_passed=plan['smoke_combined_passed'],
    interpretation=plan['interpretation'],limits=aggregate['interpretation'])
old.write(root/'analysis.json',result)
lines=['# Trading interface v5: complete development return benchmark','',
       'Three seeds per model on the same previously inspected market window. All planned decisions and failures are retained.',
       'The earlier smoke library-use gate failed. These returns measure the published bundle as actually used, including non-use; they do not establish standalone algorithm benefit.','',
       '| Model | Raw mean return (%) | Library mean return (%) | Library - raw (pp) |',
       '|---|---:|---:|---:|']
for family,values in aggregate['return_rate_pct'].items():
    lines.append(f"| {family} | {values['raw']:.6f} | {values['library']:.6f} | {values['difference_pp']:+.6f} |")
lines+=['','| Model | Seed | Raw return (%) | Library return (%) | Difference (pp) |','|---|---:|---:|---:|---:|']
for row in aggregate['seed_results']:
    lines.append(f"| {row['family']} | {row['seed']} | {row['paths']['raw']['return_rate_pct']:.6f} | {row['paths']['library']['return_rate_pct']:.6f} | {row['return_difference_pp']:+.6f} |")
lines+=['','| Model | Seed | Arm | Submitted / decisions | Decisions with successful algorithm call | Successful algorithm calls |','|---|---:|---|---:|---:|---:|']
for c in coverage:
    lines.append(f"| {c['family']} | {c['seed']} | {c['arm']} | {c['submitted']}/{c['decisions']} | {c['decisions_with_successful_algorithm_call']} | {c['successful_algorithm_calls']} |")
lines+=['','Full risk metrics, failure counts, baselines and token counts: batch/aggregate.json.',
        'Detailed library-use audit and failure messages: analysis.json.',
        'Each pair retains protocol, qualification, inference receipt, decisions, NAV and score hashes.',
        'No claim of significance, independent markets, unseen evaluation or live trading performance.']
with (root/'RESULTS.md').open('x') as f:f.write('\n'.join(lines)+'\n')
old.write(root/'completed.json',dict(analysis_sha256=old.sha(root/'analysis.json'),report_sha256=old.sha(root/'RESULTS.md'),aggregate_sha256=result['aggregate_sha256'],completed_utc=result['finished_utc']))
print(json.dumps(result),flush=True)
