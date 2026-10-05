import json,tempfile
from pathlib import Path
import numpy as np,pandas as pd
from benchmarks.agent_study import factorial_data as fd,factorial_study as fs,factorial_runtime as rt
from benchmarks.agent_study import trading_study as old,market_data as md,trading_runtime_v5 as ledger
from benchmarks.agent_study.factorial_tools import ARMS,KNOWLEDGE,NUMERICAL,SKILLS,validate

def qualify(root,baskets,corpus):
    results={}
    def check(name,passed,**detail):
        results[name]=dict(passed=bool(passed),**detail)
        print('CHECK '+name+' '+str(bool(passed)),flush=True)
    for basket in baskets:
        d=root/'datasets'/basket;t=old.load(d/'tickers.json');fd.configure(t)
        total=pd.read_csv(d/md.HIDDEN,index_col=0)
        q=pd.read_csv(d/'quotes.csv');a=pd.read_csv(d/'corporate_actions.csv')
        adjusted=md.adjust(q,a)[t]
        delta=float((adjusted.pct_change()-total.pct_change()).abs().max().max())
        check(basket+'/adjustment_parity',delta<5e-6,max_daily_return_error=delta,tolerance=5e-6)
        picks=old.schedule(list(total.index));weights=dict.fromkeys(t,1/len(t))
        nav,trades=ledger.ledger(total,picks,[weights]*len(picks))
        # Independent vector recurrence, including first trading fee and next-close timing.
        h=np.zeros(len(t));value=1.;reference=[1.];r=total[t].pct_change().fillna(0).to_numpy();ps={p+1 for p in picks}
        for k in range(picks[0]+1,len(total)):
            g=1+np.dot(h,r[k]);value*=g;h=h*(1+r[k])/g
            if k in ps:
                w=np.full(len(t),1/len(t));value*=1-np.abs(w-h).sum()*old.COST_BPS/10000;h=w
            reference.append(value)
        check(basket+'/equal_weight_accounting',np.allclose(nav.values,reference,rtol=0,atol=1e-12),max_nav_error=float(np.max(np.abs(nav.values-reference))),first_fee=trades[0]['cost'])
        hashes=[]
        for arm in ARMS:
            with tempfile.TemporaryDirectory(dir=root/'tmp') as td:
                visible=Path(td)/'visible';fd.visible(d,visible,'2024-12-02');hashes.append(old.sha(visible/'prices.csv'))
                c=rt.Controller(root,visible,arm,t,123,corpus)
                probes={'visible':('import pandas as pd\np=pd.read_csv("prices.csv")\nassert p.date.max()=="2024-12-02"\nprint("OK")',True),
                        'future_ledger':(f'open({str(d/md.HIDDEN)!r}).read()',False),
                        'future_raw':(f'open({str(md.RAW)!r}).read()',False),
                        'network':('import socket; socket.socket()',False),
                        'write':('open("prices.csv","a").write("bad")',False),
                        'direct_import':('import fin_skills',False),
                        'skill_source':(f'open({str(fs.REPO/"fin_skills/_skills"/SKILLS[0]/"SKILL.md")!r}).read()',False),
                        'submit':(f'submit({weights!r})',True),
                        'bad_submit':(f'submit({{ {t[0]!r}: -1.0 }})',False),
                        'knowledge_python':(f'r=read_skill({SKILLS[0]!r});assert r["ok"];print(r["ok"])',arm in KNOWLEDGE),
                        'algorithm_python':('r=run_algorithm("equal_weight");assert r["ok"],r;submit(r["result"]["weights"])',arm in NUMERICAL)}
                for name,(code,expected) in probes.items():
                    reply=c.call('run_python',dict(code=code));passed=bool(reply.get('ok'))==expected
                    if name in ('future_ledger','future_raw','network','write','skill_source'):passed &= 'PermissionError' in (reply.get('output','')+str(reply.get('error','')))
                    if name in ('submit','algorithm_python') and expected:passed &= reply.get('submission') is not None
                    check(f'{basket}/{arm}/{name}',passed,reply=reply)
                for tool,args,expected in [('list_skills',{},arm in KNOWLEDGE),('read_skill',dict(name=SKILLS[0]),arm in KNOWLEDGE),('list_algorithms',{},arm in NUMERICAL),('describe_algorithm',dict(algorithm_id='hrp'),arm in NUMERICAL),('run_algorithm',dict(algorithm_id='hrp'),arm in NUMERICAL),('read_market',dict(end='2026-09-25'),False)]:
                    reply=c.call(tool,args);check(f'{basket}/{arm}/{tool}',bool(reply.get('ok'))==expected,reply=reply)
                if arm in NUMERICAL:
                    for method in ('inverse_volatility','min_variance'):
                        reply=c.call('run_algorithm',dict(algorithm_id=method));w=reply.get('result',{}).get('weights')
                        valid=validate(w,t)[0] if w else None
                        check(f'{basket}/{arm}/{method}',reply.get('ok') and valid is not None,reply=reply)
        check(basket+'/identical_visible_inputs',len(set(hashes))==1,sha256=hashes)
    return dict(passed=all(x['passed'] for x in results.values()),results=results,sources=fs.sources(),scope='CPU technical qualification; no model quality or profitability gate')
