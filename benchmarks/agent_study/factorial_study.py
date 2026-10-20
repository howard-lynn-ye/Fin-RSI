"""Frozen, resumable 2x2 paired trading experiment; all cells retained."""
import argparse,json,os,random,tempfile
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,pandas as pd
from benchmarks.agent_study import trading_study as old,trading_study_v5 as v5,trading_runtime_v5 as ledger
from benchmarks.agent_study import factorial_data as data,factorial_runtime as runtime
from benchmarks.agent_study.factorial_tools import ARMS,KNOWLEDGE,NUMERICAL,Tools
from benchmarks.agent_study.trading_tools_v5 import turn_counts
REPO=Path(__file__).resolve().parents[2]

def sources():
    result=v5.sources()
    for p in Path(__file__).parent.glob('factorial_*.py'):result[p.relative_to(REPO).as_posix()]=old.sha(p)
    return result

def freeze(root,basket,family,seed):
    root=Path(root);campaign=root.parent.parent;d=campaign/'datasets'/basket
    root.mkdir(parents=True,exist_ok=False)
    tickers=old.load(d/'tickers.json');data.configure(tickers)
    total=pd.read_csv(d/'total_return_close.csv',index_col=0);picks=old.schedule(list(total.index))
    assert str(total.index[-1])=='2026-09-25'
    arms=list(ARMS);random.Random(20261003+seed).shuffle(arms)
    old.write(root/'inputs.json',dict(arms=arms,decision_sessions=picks,decision_dates=[str(total.index[i]) for i in picks]))
    old.write(root/'protocol.json',dict(version='factorial-20261003-v1',created_utc=datetime.now(timezone.utc).isoformat(),basket=basket,model=next(m for m in old.MODELS if m[0]==family),seed=seed,tickers=tickers,data_dir=str(d),
        max_turns=old.MAX_TURNS,max_tokens=old.MAX_TOKENS,temperature=.1,cost_bps=old.COST_BPS,step=old.STEP,
        primary_endpoint='Cumulative net Return Rate (%); paired K and T main effects and KxT interaction in percentage points.',
        window=[str(total.index[picks[0]+1]),str(total.index[-1])],planned_decisions=len(picks)*len(arms),decisions_per_arm=len(picks),
        sources=sources(),input_sha256=old.sha(root/'inputs.json'),data_sha256={p.name:old.sha(p) for p in sorted(d.iterdir()) if p.is_file()},
        corpus_sha256=old.sha(campaign/'corpus.json'),protocol_text_sha256=old.sha(campaign/'PROTOCOL.md'),
        limits=['Development basket is previously exposed. Transfer basket shares the same calendar and broad economic exposures; not an independent market or future period.',
                'Current frozen skill documents are not claimed to be historically available. K is eight predeclared portfolio/research skills; T includes required calling contracts.',
                'Seeds vary sampling, not market paths. Do not select runs or tune this batch after observing returns.']))

def verify(root):
    root=Path(root);p=old.load(root/'protocol.json');campaign=root.parent.parent
    assert p['sources']==sources(),'source changed after freeze'
    assert p['input_sha256']==old.sha(root/'inputs.json')
    assert p['corpus_sha256']==old.sha(campaign/'corpus.json')
    assert p['protocol_text_sha256']==old.sha(campaign/'PROTOCOL.md')
    for name,digest in p['data_sha256'].items():assert old.sha(Path(p['data_dir'])/name)==digest
    data.configure(p['tickers']);return p

def run(root):
    root=Path(root);campaign=root.parent.parent;p=verify(root)
    qualification=old.load(campaign/'qualification.json');assert qualification['passed'] and qualification['sources']==sources()
    assert old.load(campaign/'smoke-completed.json')['transport_passed'],'development transport gate failed'
    assert not (root/'inference-receipt.json').exists(),'completed run exists'
    if (root/'inference-started.json').exists():assert old.load(root/'inference-started.json')['protocol_sha256']==old.sha(root/'protocol.json')
    else:old.write(root/'inference-started.json',dict(protocol_sha256=old.sha(root/'protocol.json'),job=os.environ.get('SLURM_JOB_ID')))
    from benchmarks.agent_study.transformers_chat import TransformersChat
    backend=TransformersChat(p['model'][1],p['model'][2],max_tokens=p['max_tokens'])
    inputs=old.load(root/'inputs.json');total=pd.read_csv(Path(p['data_dir'])/'total_return_close.csv',index_col=0)
    returns=total.pct_change().fillna(0);picks=inputs['decision_sessions'];corpus=old.load(campaign/'corpus.json');receipts={}
    (root/'tmp').mkdir(exist_ok=True)
    for arm in inputs['arms']:
        holdings=dict.fromkeys(p['tickers'],0.)
        for i,pick in enumerate(picks):
            date=str(total.index[pick]);file=root/'decisions'/arm/f'{i:02d}.json'
            if file.exists():
                record=old.load(file);assert record['date']==date and record['index']==pick
                assert max(abs(record['holdings_before'][t]-holdings[t]) for t in p['tickers'])<1e-12
            else:
                backend.seed,backend.calls=p['seed']*1000+i,0
                with tempfile.TemporaryDirectory(dir=root/'tmp') as td:
                    workspace=Path(td)/'visible';data.visible(p['data_dir'],workspace,date)
                    c=runtime.Controller(root,workspace,arm,p['tickers'],p['seed']*1000+i,corpus)
                    record=runtime.decide(backend,c,v5.task(date,pick,holdings))
                record.update(date=date,index=pick,holdings_before=dict(holdings),arm=arm)
                old.write(file,record)
                print(json.dumps(dict(basket=p['basket'],family=p['model'][0],seed=p['seed'],arm=arm,decision=i+1,planned=len(picks),submitted=record['submitted'])),flush=True)
            receipts[file.relative_to(root).as_posix()]=old.sha(file)
            end=picks[i+1]+1 if i+1<len(picks) else pick+1
            for k in range(pick+1,end):
                holdings,_=old.drift(holdings,returns.iloc[k])
                if k==pick+1 and record['target'] is not None:holdings=dict(record['target'])
    assert len(receipts)==p['planned_decisions']
    old.write(root/'inference-receipt.json',dict(protocol_sha256=old.sha(root/'protocol.json'),decisions=receipts))

def coverage(records):
    calls=[c for r in records for c in r['tool_calls']]
    adopted=0
    for r in records:
        target=r.get('target')
        if target and any(c.get('ok') and c.get('tool')=='run_algorithm' and c.get('weights') and max(abs(target.get(t,0)-c['weights'].get(t,0)) for t in target)<1e-5 for c in r['tool_calls']):adopted+=1
    return dict(submitted=sum(r['submitted'] for r in records),decisions=len(records),
        successful_algorithm_calls=sum(c.get('tool')=='run_algorithm' and c.get('ok',False) for c in calls),
        successful_skill_reads=sum(c.get('tool')=='read_skill' and c.get('ok',False) for c in calls),
        algorithm_discovery_calls=sum(c.get('tool') in ('list_algorithms','describe_algorithm') and c.get('ok',False) for c in calls),
        decisions_with_algorithm=sum(any(c.get('tool')=='run_algorithm' and c.get('ok',False) for c in r['tool_calls']) for r in records),
        submitted_weights_matching_a_tool_output=adopted,
        **turn_counts([t for r in records for t in r['turns']]),
        total_turns=sum(len(r['turns']) for r in records),
        prompt_tokens=sum((t.get('usage') or {}).get('prompt_tokens',0) for r in records for t in r['turns']),
        completion_tokens=sum((t.get('usage') or {}).get('completion_tokens',0) for r in records for t in r['turns']),
        generation_seconds=sum(t.get('generation_seconds',0) for r in records for t in r['turns']))

def score(root):
    root=Path(root);p=verify(root);inputs=old.load(root/'inputs.json');receipt=old.load(root/'inference-receipt.json');picks=inputs['decision_sessions']
    assert receipt['protocol_sha256']==old.sha(root/'protocol.json')
    expected={f'decisions/{a}/{i:02d}.json' for a in ARMS for i in range(len(picks))};assert expected==set(receipt['decisions'])
    assert expected=={f.relative_to(root).as_posix() for f in (root/'decisions').glob('*/*.json')}
    total=pd.read_csv(Path(p['data_dir'])/'total_return_close.csv',index_col=0);paths={}
    for arm in ARMS:
        records=[]
        for i in range(len(picks)):
            name=f'decisions/{arm}/{i:02d}.json';assert old.sha(root/name)==receipt['decisions'][name];records.append(old.load(root/name))
        targets=[r['target'] for r in records];nav,trades=ledger.ledger(total,picks,targets)
        metrics=ledger.metrics(nav);paths[arm]=dict(return_rate_pct=100*metrics['cumulative_return'],metrics=metrics,turnover=sum(t['turnover'] for t in trades),**coverage(records))
        fixed_costs={}
        for cost in (0,10,25):
            old.COST_BPS=cost
            fixed_costs[str(cost)]=100*ledger.metrics(ledger.ledger(total,picks,targets)[0])['cumulative_return']
        old.COST_BPS=p['cost_bps'];paths[arm]['fixed_decision_cost_sensitivity_pct']=fixed_costs
        old.write(root/'nav'/f'{arm}.json',dict(nav=nav.to_dict(),trades=trades))
    r={a:paths[a]['return_rate_pct'] for a in ARMS}
    effects=dict(full_minus_base=r['full']-r['base'],knowledge_main=(r['knowledge']-r['base']+r['full']-r['tools'])/2,tools_main=(r['tools']-r['base']+r['full']-r['knowledge'])/2,interaction=r['full']-r['knowledge']-r['tools']+r['base'])
    result=dict(basket=p['basket'],family=p['model'][0],seed=p['seed'],window=p['window'],paths=paths,effects_pp=effects,limits=p['limits'])
    old.write(root/'scores.json',result);old.write(root/'completed.json',dict(scores_sha256=old.sha(root/'scores.json')));return result

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=('run','score'));a.add_argument('root',type=Path);args=a.parse_args();globals()[args.mode](args.root.resolve())
