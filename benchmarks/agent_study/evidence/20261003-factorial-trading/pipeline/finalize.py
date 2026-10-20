"""Complete-batch scoring, fixed baselines and descriptive dependence sensitivity."""
import json,tempfile
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,pandas as pd
from benchmarks.agent_study import factorial_study as fs,factorial_data as fd,trading_study as old,trading_runtime_v5 as ledger
from benchmarks.agent_study.factorial_tools import ARMS,Tools,validate
ROOT=Path(__file__).resolve().parents[1]

def exposure(total,picks,targets):
    h=dict.fromkeys(total.columns,0.);rs=total.pct_change().fillna(0);trade={p+1:t for p,t in zip(picks,targets)};rows=[]
    for k in range(picks[0]+1,len(total)):
        h,_=old.drift(h,rs.iloc[k])
        if k in trade and trade[k] is not None:h=dict(trade[k])
        rows.append(dict(date=total.index[k],risky=sum(h.values()),cash=1-sum(h.values()),maximum_asset=max(h.values()),**h))
    f=pd.DataFrame(rows)
    return dict(mean_risky_weight=float(f.risky.mean()),mean_cash_weight=float(f.cash.mean()),mean_largest_asset_weight=float(f.maximum_asset.mean()),mean_asset_weights={t:float(f[t].mean()) for t in total.columns})

def baselines(basket):
    d=ROOT/'datasets'/basket;t=old.load(d/'tickers.json');fd.configure(t);total=pd.read_csv(d/'total_return_close.csv',index_col=0);picks=old.schedule(list(total.index))
    methods=('equal_weight','inverse_volatility','min_variance','hrp');targets={m:[] for m in methods}
    for i,p in enumerate(picks):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as td:
            v=Path(td)/'visible';fd.visible(d,v,str(total.index[p]));tool=Tools(v,'tools',t)
            for m in methods:
                reply=tool.run_algorithm(m);w,err=validate(reply['result']['weights'],t);assert err is None,(m,err);targets[m].append(w)
    targets['equal_weight_hold']=[targets['equal_weight'][0]]+[None]*(len(picks)-1)
    targets['cash']=[None]*len(picks)
    if basket=='original':targets['60_40_rebalance']=[dict.fromkeys(t,0.)|{'SPY':.6,'IEF':.4} for _ in picks]
    out={}
    for name,w in targets.items():
        nav,trades=ledger.ledger(total,picks,w);met=ledger.metrics(nav)
        out[name]=dict(return_rate_pct=100*met['cumulative_return'],metrics=met,turnover=sum(x['turnover'] for x in trades),exposure=exposure(total,picks,w))
        old.write(ROOT/'baselines'/basket/f'{name}.json',dict(targets=w,nav=nav.to_dict(),trades=trades,**out[name]))
    return out

def block_sensitivity(basket,family,items):
    # Resample paired calendar blocks jointly across all conditions AND seeds.
    arrays=[]
    for item in sorted(items,key=lambda x:x['seed']):
        pair=ROOT/'batch'/f'{basket}-{family}-{item["seed"]}'
        arrays.append(np.column_stack([pd.Series(old.load(pair/'nav'/f'{a}.json')['nav']).pct_change().dropna().to_numpy() for a in ARMS]))
    r=np.stack(arrays,axis=1);n=len(r);out={};rng=np.random.default_rng(20261003)
    for block in (10,20,40):
        simulations=[]
        for _ in range(2000):
            starts=rng.integers(0,n,size=int(np.ceil(n/block)));ix=((starts[:,None]+np.arange(block))%n).reshape(-1)[:n]
            returns=np.prod(1+r[ix],axis=0)-1
            b,k,t,f=returns.mean(axis=0)*100
            simulations.append([f-b,(k-b+f-t)/2,(t-b+f-k)/2,f-k-t+b])
        intervals=np.quantile(simulations,[.025,.975],axis=0)
        out[str(block)]={name:dict(low=float(intervals[0,j]),high=float(intervals[1,j])) for j,name in enumerate(('full_minus_base','knowledge_main','tools_main','interaction'))}
    return dict(kind='paired circular time-block bootstrap; joint calendar resampling across seeds, 2000 draws, descriptive 95% intervals in percentage points',block_lengths=out,limitation='One historical market path. These intervals do not establish generalization across markets or independent random-seed replications.')

def main():
    plan=old.load(ROOT/'batch-plan.json');prepared=old.load(ROOT/'prepared.json');assert old.sha(ROOT/'batch-plan.json')==prepared['plan_sha256']
    missing=[x['root'] for x in plan if not (Path(x['root'])/'inference-receipt.json').exists()]
    if missing:
        old.write(ROOT/'incomplete.json',dict(missing_pairs=missing,observed_utc=datetime.now(timezone.utc).isoformat()));raise RuntimeError('Incomplete batch; no partial headline aggregation')
    results=[]
    for x in plan:
        pair=Path(x['root']);assert old.sha(pair/'protocol.json')==x['protocol_sha256'];r=fs.score(pair)
        p=fs.verify(pair);total=pd.read_csv(Path(p['data_dir'])/'total_return_close.csv',index_col=0);picks=old.load(pair/'inputs.json')['decision_sessions']
        for arm in ARMS:r['paths'][arm]['exposure']=exposure(total,picks,[old.load(pair/'decisions'/arm/f'{i:02d}.json')['target'] for i in range(len(picks))])
        results.append(r)
    fixed={b:baselines(b) for b in prepared['baskets']};groups=[]
    for basket in prepared['baskets']:
        for family in ('7b','14b'):
            items=[r for r in results if r['basket']==basket and r['family']==family];assert {x['seed'] for x in items}=={11,23,37}
            means={a:float(np.mean([x['paths'][a]['return_rate_pct'] for x in items])) for a in ARMS}
            effects={e:float(np.mean([x['effects_pp'][e] for x in items])) for e in items[0]['effects_pp']}
            groups.append(dict(basket=basket,family=family,mean_return_rate_pct=means,mean_effects_pp=effects,seed_effects=[dict(seed=x['seed'],**x['effects_pp']) for x in items],descriptive_block_sensitivity=block_sensitivity(basket,family,items)))
    report=dict(completed_utc=datetime.now(timezone.utc).isoformat(),decisions=sum(x['planned_decisions'] for x in plan),results=results,groups=groups,baselines=fixed,source_sha256=fs.sources(),qualification_sha256=old.sha(ROOT/'qualification.json'),plan_sha256=old.sha(ROOT/'batch-plan.json'),transfer_pending=prepared['transfer_pending'])
    old.write(ROOT/'analysis.json',report)
    lines=['# Factorial trading study results','',f'Complete verified decisions: {report["decisions"]}. Four cells share the same point-in-time adjusted price input.','', '| Basket | Model | Base % | Knowledge % | Tools % | Full % | K main pp | T main pp | Interaction pp |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for g in groups:
        a=g['mean_return_rate_pct'];e=g['mean_effects_pp'];lines.append(f'| {g["basket"]} | {g["family"]} | '+ ' | '.join(f'{a[x]:.3f}' for x in ARMS)+f' | {e["knowledge_main"]:.3f} | {e["tools_main"]:.3f} | {e["interaction"]:.3f} |')
    lines += ['','Means are over the three declared sampling seeds on one common historical market path. Original assets were exposed to development. Transfer assets were preselected before downloading outcomes, but share the calendar and economic exposures. Current skill documents are not historical point-in-time knowledge. This is not an independent new time period.','', 'All submissions, non-submissions, failed calls, natural adoption, fixed-decision cost sensitivity, exposure, risk metrics, per-seed effects, baseline paths and descriptive paired block intervals are in analysis.json. Numerical-tool access includes calling-contract documentation; K is the additional eight-skill corpus.','', 'Baselines use the same next-close timing, ten-session schedule and 5 bps traded-side cost. Supported portfolio methods use 252 historical returns and fixed library defaults. Equal-weight hold trades once; 60/40 means rebalanced SPY/IEF on the original basket. No returns were used to select model seeds or arms.']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    old.write(ROOT/'completed.json',dict(completed_utc=report['completed_utc'],decisions=report['decisions'],analysis_sha256=old.sha(ROOT/'analysis.json'),report_sha256=old.sha(ROOT/'RESULTS.md')))
    print('\n'.join(lines),flush=True)
if __name__=='__main__':main()
