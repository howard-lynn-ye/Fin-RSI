"""Post hoc diagnostics of the already completed v5; never edits its artifacts."""
import json,collections
from pathlib import Path
import numpy as np,pandas as pd
from benchmarks.agent_study import trading_study as old,trading_study_v5 as v5,trading_runtime_v5 as ledger
from finalize import exposure
ROOT=Path(__file__).resolve().parents[1]
PRIOR=Path('/beacon-projects/radfm/wy891/fin-skills-trading-return-v5-20261002');out=ROOT/'v5-posthoc';out.mkdir(exist_ok=False)
rows=[]
for family in ('7b','14b'):
    for seed in (11,23,37):
        pair=PRIOR/'batch'/f'{family}-{seed}';p=v5.verify(pair);receipt=old.load(pair/'inference-receipt.json')
        assert receipt['protocol_sha256']==old.sha(pair/'protocol.json')
        for name,digest in receipt['decisions'].items():assert old.sha(pair/name)==digest
        total=pd.read_csv(pair/'data/total_return_close.csv',index_col=0);picks=old.load(pair/'inputs.json')['decision_sessions']
        for arm in ('raw','library'):
            records=[old.load(pair/'decisions'/arm/f'{i:02d}.json') for i in range(len(picks))];targets=[r['target'] for r in records]
            nav,trades=ledger.ledger(total,picks,targets);metrics=ledger.metrics(nav)
            assert abs(100*metrics['cumulative_return']-old.load(pair/'scores.json')['paths'][arm]['return_rate_pct'])<1e-10
            calls=[c for r in records for c in r['tool_calls']];turns=[t for r in records for t in r['turns']];parse=collections.Counter(t.get('parse') for t in turns)
            errors=collections.Counter(str(t.get('result',{}).get('error'))[:200] for t in turns if t.get('result',{}).get('error'))
            cost={}
            try:
                for bps in (0,5,10,25):old.COST_BPS=bps;cost[str(bps)]=100*ledger.metrics(ledger.ledger(total,picks,targets)[0])['cumulative_return']
            finally:old.COST_BPS=5
            match=0
            for r in records:
                candidate=[c.get('weights') for c in r['tool_calls'] if c.get('ok') and c.get('tool')=='run_algorithm']
                candidate += [t['result'].get('result',{}).get('weights') for t in r['turns'] if t.get('tool')=='run_algorithm' and t.get('result',{}).get('ok')]
                if r['target'] and any(w and max(abs(r['target'].get(t,0)-w.get(t,0)) for t in old.TICKERS)<1e-5 for w in candidate):match+=1
            row=dict(family=family,seed=seed,arm=arm,return_rate_pct=100*metrics['cumulative_return'],metrics=metrics,exposure=exposure(total,picks,targets),turnover=sum(t['turnover'] for t in trades),cost_sensitivity_fixed_decisions_pct=cost,submitted=sum(r['submitted'] for r in records),decisions=len(records),successful_algorithm_calls=sum(c.get('tool')=='run_algorithm' and c.get('ok',False) for c in calls),decisions_with_algorithm=sum(any(c.get('tool')=='run_algorithm' and c.get('ok') for c in r['tool_calls']) for r in records),exact_tool_weight_adoption=match,successful_skill_reads=sum(c.get('tool')=='read_skill' and c.get('ok',False) for c in calls),parse_counts=dict(parse),frequent_errors=errors.most_common(12),completion_tokens=sum((t.get('usage') or {}).get('completion_tokens',0) for t in turns))
            rows.append(row)
groups=[]
for family in ('7b','14b'):
    for arm in ('raw','library'):
        r=[x for x in rows if x['family']==family and x['arm']==arm]
        groups.append(dict(family=family,arm=arm,mean_return_rate_pct=float(np.mean([x['return_rate_pct'] for x in r])),mean_sharpe=float(np.mean([x['metrics']['sharpe'] for x in r])),mean_max_drawdown=float(np.mean([x['metrics']['max_drawdown'] for x in r])),mean_risky_weight=float(np.mean([x['exposure']['mean_risky_weight'] for x in r])),mean_turnover=float(np.mean([x['turnover'] for x in r])),mean_asset_weights={t:float(np.mean([x['exposure']['mean_asset_weights'][t] for x in r])) for t in old.TICKERS},submitted=sum(x['submitted'] for x in r),decisions=sum(x['decisions'] for x in r),algorithm_decisions=sum(x['decisions_with_algorithm'] for x in r),exact_tool_weight_adoption=sum(x['exact_tool_weight_adoption'] for x in r),successful_skill_reads=sum(x['successful_skill_reads'] for x in r)))
old.write(out/'analysis.json',dict(label='Post hoc diagnostics, not new inference or causal mechanism identification',prior_completed_sha256=old.sha(PRIOR/'completed.json'),rows=rows,groups=groups))
lines=['# Completed v5: post hoc diagnostics','','These statistics describe existing frozen decisions. They cannot identify which component caused the return differences.','', '| Model | Arm | Mean return % | Mean Sharpe | Mean max DD % | Mean invested % | Mean turnover | Submitted | Algorithm decisions | Exact adoption |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for g in groups:lines.append(f'| {g["family"]} | {g["arm"]} | {g["mean_return_rate_pct"]:.3f} | {g["mean_sharpe"]:.3f} | {100*g["mean_max_drawdown"]:.3f} | {100*g["mean_risky_weight"]:.3f} | {g["mean_turnover"]:.3f} | {g["submitted"]}/{g["decisions"]} | {g["algorithm_decisions"]} | {g["exact_tool_weight_adoption"]} |')
lines += ['', 'All four groups have zero successful skill-document reads. Algorithm execution and exact adoption are distinct. Missing submissions retain earlier holdings, so a lower submission rate does not mechanically imply lower market exposure. Mean risk metrics average separate seed trajectories; they do not describe a pooled portfolio. Cost sensitivity reprices fixed decisions, not behavior under new costs.','', 'The factorial study is needed to separate document access, numerical-tool access and their interaction under the same adjusted data input. These post hoc observations alone do not establish that library algorithms caused either positive or negative effects.']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n');old.write(out/'completed.json',dict(analysis_sha256=old.sha(out/'analysis.json'),report_sha256=old.sha(out/'REPORT.md')))
print('\n'.join(lines),flush=True)
