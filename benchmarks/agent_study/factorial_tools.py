"""Two-factor study tools over identical point-in-time adjusted prices."""
import inspect, json, math, random
from pathlib import Path
import pandas as pd
from benchmarks.agent_study.trading_tools_v5 import text_page, rows_page, positive_integer
ARMS=('base','knowledge','tools','full')
KNOWLEDGE=('knowledge','full')
NUMERICAL=('tools','full')
SKILLS=('portfolio-optimizers','covariance-and-risk-models','corporate-actions-processing','backtest-validation','backtest-overfitting','data-quality-validation','signal-construction','trend-following-models')
ALLOW_INTERNAL_IMPORT=False

def validate(weights,tickers):
    if not isinstance(weights,dict) or not weights:return None,'weights must be a nonempty dictionary'
    clean=dict.fromkeys(tickers,0.)
    for ticker,value in weights.items():
        if ticker not in clean:return None,f'unknown ticker {ticker!r}'
        try:value=float(value)
        except (TypeError,ValueError):return None,f'invalid weight for {ticker}'
        if not math.isfinite(value) or value<0:return None,f'weight for {ticker} must be finite and >= 0'
        clean[ticker]=value
    total=sum(clean.values())
    if total>1.001+1e-12:return None,'weights sum exceeds 1.001'
    if total>1:clean={k:v/total for k,v in clean.items()}
    return clean,None

class Tools:
    def __init__(self,workspace,arm,tickers,menu_seed=0,corpus=None):
        assert arm in ARMS
        self.workspace,self.arm,self.tickers,self.menu_seed=Path(workspace),arm,tuple(tickers),menu_seed
        self.corpus,self.calls=corpus or {},[]
    @property
    def allowed(self):
        return ('read_file','read_market')+(('list_skills','read_skill') if self.arm in KNOWLEDGE else ())+(('list_algorithms','describe_algorithm','run_algorithm') if self.arm in NUMERICAL else ())
    def read_file(self,path='',offset=0,limit=4000):
        if path not in ('prices.csv','README.md'):raise ValueError('files: prices.csv, README.md')
        return text_page((self.workspace/path).read_text(),offset,limit,path=path)
    def read_market(self,tickers=None,start=None,end=None,offset=0,limit=12):
        q=pd.read_csv(self.workspace/'prices.csv');as_of=str(q.date.max())
        if tickers is not None:
            if not isinstance(tickers,list) or not tickers or set(tickers)-set(self.tickers):raise ValueError('tickers must be a nonempty list of visible asset names')
            q=q[q.ticker.isin(tickers)]
        for name,value in (('start',start),('end',end)):
            if value is not None and (pd.Timestamp(value).strftime('%Y-%m-%d')!=value or value>as_of):raise ValueError(f'{name} must be YYYY-MM-DD <= {as_of}')
        if start and end and start>end:raise ValueError('start > end')
        if start:q=q[q.date>=start]
        if end:q=q[q.date<=end]
        return rows_page(q.sort_values(['date','ticker'],ascending=[False,True]).to_dict('records'),offset,limit,'rows',as_of=as_of,adjusted=True)
    def list_skills(self,query='',offset=0,limit=8):
        rows=[dict(name=k,description=v['description']) for k,v in self.corpus.items() if str(query).lower() in (k+' '+v['description']).lower()]
        return rows_page(rows,offset,limit,'skills')
    def read_skill(self,name='',offset=0,limit=4000,reference=None):
        if name not in self.corpus:raise ValueError('unknown skill; use list_skills')
        entry=self.corpus[name]
        if reference is not None:
            if reference not in entry['references']:raise ValueError('unknown reference')
            return text_page(entry['references'][reference],offset,limit,name=name,reference=reference)
        return text_page(entry['text'],offset,limit,name=name,references=list(entry['references']))
    def list_algorithms(self):
        from fin_skills.algorithms import catalog
        from benchmarks.agent_study.trading_study import READY_ALGORITHMS
        cards=[dict(id=c['id'],task=c['task']) for c in catalog() if c['id'] in READY_ALGORITHMS]
        cards.sort(key=lambda c:c['id']);random.Random(self.menu_seed).shuffle(cards)
        return dict(ok=True,algorithms=cards)
    def describe_algorithm(self,algorithm_id=''):
        from benchmarks.agent_study.trading_tools_v5 import Tools as Prior
        result=Prior(self.workspace,'library',self.menu_seed).describe_algorithm(algorithm_id)
        if result.get('ok'):
            result['note']='Bound functions; all replies are dictionaries. Shared prices.csv is already total-return adjusted. No further adjustment is needed.'
            result['call_signature']='(algorithm_id, tickers=None, lookback=252, parameters=None)'
        return result
    def run_algorithm(self,algorithm_id,tickers=None,lookback=252,parameters=None):
        import numpy as np
        import fin_skills.algorithms as algorithms
        from benchmarks.agent_study.trading_tools_v5 import method_defaults
        from benchmarks.agent_study.trading_worker import _round
        from benchmarks.agent_study.trading_study import READY_ALGORITHMS
        lookback=positive_integer(lookback,'lookback')
        if algorithm_id not in READY_ALGORITHMS:raise ValueError('unknown or unsupported algorithm; use list_algorithms')
        cards={c['id']:c for c in algorithms.catalog()};card=cards[algorithm_id]
        if parameters is not None and not isinstance(parameters,dict):raise ValueError('parameters must be a dictionary')
        params=dict(parameters or {})
        for name in ('lookback','fast','slow','signal_span','baseline'):
            if name in params:positive_integer(params[name],name)
        effective=method_defaults(algorithm_id,card['task']);effective.update(params)
        q=pd.read_csv(self.workspace/'prices.csv');p=q.pivot(index='date',columns='ticker',values='close').sort_index()
        tickers=list(self.tickers) if tickers is None else tickers
        if not isinstance(tickers,list) or not tickers or len(set(tickers))!=len(tickers) or set(tickers)-set(self.tickers):raise ValueError('invalid ticker list')
        window=p[tickers].tail(lookback+1);returns=window.pct_change().dropna();inputs=tuple(card['inputs'])
        if inputs==('asset_returns',):
            result=algorithms.run(algorithm_id,{'asset_returns':returns},**effective)
            value={'weights':_round(pd.Series(np.asarray(result,dtype=float),index=tickers))}
        elif inputs in (('prices',),('returns',),('series',)):
            value={'per_ticker':{t:_round(algorithms.run(algorithm_id,{inputs[0]:returns[t] if inputs[0]=='returns' else window[t]},**effective)) for t in tickers}}
        else:raise ValueError('method requires inputs outside this price-only study')
        return dict(ok=True,algorithm_id=algorithm_id,result=value,effective_parameters=effective,adjusted=True,as_of=str(window.index[-1]))
    def call(self,name,arguments):
        global ALLOW_INTERNAL_IMPORT
        args={} if arguments is None else arguments
        if name not in self.allowed:result=dict(ok=False,error=f'tool {name!r} unavailable in {self.arm}')
        elif not isinstance(args,dict):result=dict(ok=False,error='arguments must be an object')
        else:
            try:
                ALLOW_INTERNAL_IMPORT=True
                result=getattr(self,name)(**args)
            except Exception as exc:result=dict(ok=False,error=f'{type(exc).__name__}: {str(exc)[:500]}',expected=f'{name}{inspect.signature(getattr(self,name))}')
            finally:ALLOW_INTERNAL_IMPORT=False
        receipt=dict(tool=name,ok=bool(result.get('ok')))
        if name=='run_algorithm':receipt.update(algorithm_id=args.get('algorithm_id') if isinstance(args,dict) else None,weights=result.get('result',{}).get('weights'))
        if name=='read_skill' and isinstance(args,dict):receipt['name']=args.get('name')
        self.calls.append(receipt);return result
    def python_bindings(self):
        def binding(name):
            sig=inspect.signature(getattr(self,name))
            def wrapped(*args,**kwargs):return self.call(name,dict(sig.bind(*args,**kwargs).arguments))
            return wrapped
        return {name:binding(name) for name in self.allowed}
