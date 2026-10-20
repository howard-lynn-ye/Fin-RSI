import hashlib,json,sys,shutil,math,os
from pathlib import Path
import numpy as np,pandas as pd
R=Path('/beacon-projects/radfm/wy891/fin-skills-factorial-20261003')
P=Path(__file__).resolve().parent
S=P/'source'
sys.path.insert(0,str(R/'source'))
from benchmarks.agent_study import factorial_study as fs,trading_study as old,trading_runtime_v5 as ledger
from benchmarks.agent_study.factorial_tools import ARMS
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
load=lambda p:json.loads(p.read_text())
def equal(a,b):
    if isinstance(a,dict):
        assert set(a)<=set(b)
        for k,v in a.items():equal(v,b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    elif isinstance(a,(float,int)) and not isinstance(a,bool):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-9),(a,b)
    else:assert a==b,(a,b)
c=load(R/'completed.json');a=load(R/'analysis.json');plan=load(R/'batch-plan.json');prep=load(R/'prepared.json')
assert c['analysis_sha256']==sha(R/'analysis.json') and c['report_sha256']==sha(R/'RESULTS.md')
assert a['source_sha256']==fs.sources()==prep['sources']
assert prep['plan_sha256']==a['plan_sha256']==sha(R/'batch-plan.json')
assert a['qualification_sha256']==prep['qualification_sha256']==sha(R/'qualification.json')
for name,digest in load(R/'pipeline-receipt.json')['files'].items():assert sha(R/name)==digest,name
expected={(b,m,s) for b in ('original','transfer') for m in ('7b','14b') for s in (11,23,37)}
assert {(x['basket'],x['family'],x['seed']) for x in a['results']}==expected
count=0;errors=[]
for item in plan:
    pair=Path(item['root']);p=fs.verify(pair);rec=load(pair/'inference-receipt.json');scores=load(pair/'scores.json')
    assert item['protocol_sha256']==rec['protocol_sha256']==sha(pair/'protocol.json')
    assert load(pair/'completed.json')['scores_sha256']==sha(pair/'scores.json')
    result=next(x for x in a['results'] if (x['basket'],x['family'],x['seed'])==(p['basket'],p['model'][0],p['seed']))
    equal(scores,result)
    picks=load(pair/'inputs.json')['decision_sessions']
    names={f'decisions/{arm}/{i:02d}.json' for arm in ARMS for i in range(len(picks))}
    assert len(names)==176 and names==set(rec['decisions'])
    assert names=={f.relative_to(pair).as_posix() for f in (pair/'decisions').glob('*/*.json')}
    for name,digest in rec['decisions'].items():assert sha(pair/name)==digest;count+=1
    total=pd.read_csv(Path(p['data_dir'])/'total_return_close.csv',index_col=0)
    for arm in ARMS:
        records=[load(pair/'decisions'/arm/f'{i:02d}.json') for i in range(len(picks))]
        equal(fs.coverage(records),result['paths'][arm])
        saved=load(pair/'nav'/f'{arm}.json')
        nav,trades=ledger.ledger(total,picks,[r['target'] for r in records])
        equal(dict(nav=nav.to_dict(),trades=trades),saved)
        equal(ledger.metrics(nav),result['paths'][arm]['metrics'])
        # Independent dollar holdings implementation, including initial transaction fee.
        values=np.zeros(len(total.columns));cash=1.;prev=total.iloc[picks[0]].to_numpy(float)
        targets={k+1:r['target'] for k,r in zip(picks,records)}
        values_nav=[1.]
        for k in range(picks[0]+1,len(total)):
            price=total.iloc[k].to_numpy(float);values*=price/prev;prev=price
            wealth=float(values.sum()+cash);t=targets.get(k)
            if t is not None:
                w=np.array([t.get(str(col),0.) for col in total.columns]);turnover=np.abs(w-values/wealth).sum()
                wealth*=1-p['cost_bps']/10000*turnover
                values=wealth*w;cash=wealth*(1-w.sum())
            values_nav.append(float(values.sum()+cash))
        error=float(np.max(np.abs(np.array(values_nav)-nav.to_numpy())))
        assert error<1e-10,(pair,arm,error)
        errors.append(error)
for b,methods in a['baselines'].items():
    for name,result in methods.items():equal(result,load(R/'baselines'/b/f'{name}.json'))
assert count==c['decisions']==a['decisions']==2112
E=S/'benchmarks/agent_study/evidence/20261003-factorial-trading';E.mkdir(parents=True,exist_ok=False)
for name in ('PROTOCOL.md','ANALYSIS_PLAN.md','analysis.json','RESULTS.md','completed.json','prepared.json','qualification.json','scoring-qualification.json','baseline-qualification.json','pipeline-receipt.json','batch-plan.json','smoke-completed.json','corpus-audit.json','corpus-reviewed.json','BASELINE_RELOCATION.md'):
    shutil.copy2(R/name,E/name)
for pair in sorted((R/'batch').iterdir()):
    if not pair.is_dir():continue
    dest=E/'receipts'/pair.name;dest.mkdir(parents=True)
    for name in ('protocol.json','inputs.json','inference-receipt.json','completed.json','scores.json'):shutil.copy2(pair/name,dest/name)
for f in sorted((R/'scripts').iterdir()):
    if f.is_file():
        dest=E/'pipeline'/f.name;dest.parent.mkdir(exist_ok=True);shutil.copy2(f,dest)
for f in (R/'source/benchmarks/agent_study').glob('factorial_*.py'):shutil.copy2(f,S/'benchmarks/agent_study'/f.name)
report=dict(verified_decisions=count,verified_pairs=len(plan),verified_paths=len(errors),maximum_independent_nav_error=max(errors),source_receipts_passed=True,data_receipts_passed=True,pipeline_receipts_passed=True,decision_receipts_passed=True,analysis_sha256=sha(R/'analysis.json'),raw_artifacts='Retained on Beacon; not included in this release',slurm_job_id=os.environ['SLURM_JOB_ID'])
(E/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
manifest={f.relative_to(E).as_posix():sha(f) for f in sorted(E.rglob('*')) if f.is_file()}
(E/'SHA256SUMS.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
print('SUMMARY',json.dumps(a['groups']),flush=True)
print('BASELINES',json.dumps({b:{k:v['return_rate_pct'] for k,v in ms.items()} for b,ms in a['baselines'].items()}),flush=True)
print('ACCESS',json.dumps([dict(basket=g['basket'],family=g['family'],arms={arm:{k:sum(x['paths'][arm][k] for x in a['results'] if x['basket']==g['basket'] and x['family']==g['family']) for k in ('decisions','submitted','decisions_with_algorithm','successful_skill_reads','submitted_weights_matching_a_tool_output','failed_turns','total_turns','prompt_tokens','completion_tokens')} for arm in ARMS}) for g in a['groups']]),flush=True)
