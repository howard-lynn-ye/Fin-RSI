"""Factorial controller; reuses the frozen v5 parser and feedback accounting."""
import json,os,subprocess,sys,tempfile
from pathlib import Path
from benchmarks.agent_study import trading_runtime_v5 as prior
from benchmarks.agent_study.factorial_tools import Tools,validate,KNOWLEDGE,NUMERICAL
COMMON='''Use only the visible files. prices.csv has columns date,ticker,close. close is a point-in-time total-return price index, already split/dividend adjusted; do not adjust it again. README.md describes the data. All conditions have identical prices.
Functions already bound in Python and available as JSON tools:
read_file(path, offset=0, limit=4000)
read_market(tickers=None, start=None, end=None, offset=0, limit=12)
run_python(code), submit(weights)
Readers return dictionaries: check reply["ok"]. read_market records are reply["rows"], not reply["data"]. It returns at most 100 rows per page with next_offset, not a full estimation window.
For calculations use import pandas as pd; q=pd.read_csv("prices.csv"); prices=q.pivot(index="date",columns="ticker",values="close").sort_index()
Python has NumPy/pandas/SciPy, which you must import yourself. Python state does not persist between calls. Load data and compute and submit within one call, or copy numeric weights into a later call. Files are read-only and network/subprocesses are denied. Python workers have an 8 GiB address-space limit and a 60-second wall limit.
Use one tool JSON object {"tool":"name","arguments":{...}} or ONE complete Python code fence per turn. Output budget is 4000 characters; print compact results. Oversized structured outputs are rejected and long stdout is explicitly clipped.
submit(weights) ends a decision. weights is a nonempty ticker:number dictionary, finite, nonnegative, sum <=1; cash is the remainder. Totals <=1.001 are normalized for rounding. Choose your own strategy and assets. Missing submission retains current holdings.
At most 8 turns and 1024 generated tokens per turn. Direct fin_skills imports are disabled for every condition; use the bound interfaces available below.
'''
K='''Financial skill documents are available through list_skills(query="",offset=0,limit=8), read_skill(name,offset=0,limit=4000,reference=None). They return dictionaries and next_offset for pagination. Read references listed by read_skill if useful. These documents provide guidance, not computed weights.
'''
T='''Numerical methods are available through list_algorithms(), describe_algorithm(algorithm_id), run_algorithm(algorithm_id,tickers=None,lookback=252,parameters=None). The menu is randomized, not a recommendation. Describe a method for its exact inputs, defaults and executable example. The adapter uses the same adjusted prices.csv you can read. All replies are dictionaries: check reply["ok"]. Portfolio weights are reply["result"]["weights"]; other outputs are per_ticker values and are not weights. lookback is history length; method parameters are separate. Choose whether to use a method and whether to adopt its output. No tool output is automatically submitted.
'''

class Controller:
    def __init__(self,root,workspace,arm,tickers,menu_seed,corpus):
        self.root,self.workspace=Path(root),Path(workspace);self.arm,self.tickers,self.menu_seed=arm,tuple(tickers),menu_seed
        self.corpus=corpus if arm in KNOWLEDGE else {};self.calls=[]
        self.tools=Tools(workspace,arm,tickers,menu_seed,self.corpus)
        (self.root/'tmp').mkdir(exist_ok=True,parents=True)
    def call(self,tool,arguments):
        args={} if arguments is None else arguments
        if not isinstance(args,dict):result=dict(ok=False,error='arguments must be an object')
        elif tool!='run_python' and tool not in self.tools.allowed:result=dict(ok=False,error=f'tool {tool!r} unavailable in {self.arm}')
        else:
            with tempfile.TemporaryDirectory(dir=self.root/'tmp') as td:
                box=Path(td);(box/'request.json').write_text(json.dumps(dict(arm=self.arm,tickers=self.tickers,menu_seed=self.menu_seed,tool=tool,arguments=args)))
                if self.arm in KNOWLEDGE:(box/'corpus.json').write_text(json.dumps(self.corpus))
                env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[2]),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TMPDIR=str(box),MPLCONFIGDIR=str(box))
                try:
                    proc=subprocess.run([sys.executable,'-B','-m','benchmarks.agent_study.factorial_worker',str(box),str(self.workspace)],env=env,capture_output=True,text=True,timeout=60)
                    result=json.loads((box/'result.json').read_text()) if (box/'result.json').exists() else dict(ok=False,error='worker failed: '+proc.stderr[-1500:],worker_returncode=proc.returncode)
                except subprocess.TimeoutExpired:result=dict(ok=False,error='tool timeout')
        receipt=dict(tool=tool,ok=bool(result.get('ok')))
        if tool=='run_algorithm':receipt.update(algorithm_id=args.get('algorithm_id') if isinstance(args,dict) else None,weights=result.get('result',{}).get('weights'))
        self.calls.append(receipt);self.calls.extend(result.get('tool_calls',[]));return result

def decide(backend,controller,task):
    previous_prompt,previous_validate=prior.COMMON,prior.validate_weights
    prior.COMMON=COMMON+(K if controller.arm in KNOWLEDGE else '')+(T if controller.arm in NUMERICAL else '')
    prior.validate_weights=lambda weights:validate(weights,controller.tickers)
    try:return prior.decide(backend,controller,task)
    finally:prior.COMMON,prior.validate_weights=previous_prompt,previous_validate
