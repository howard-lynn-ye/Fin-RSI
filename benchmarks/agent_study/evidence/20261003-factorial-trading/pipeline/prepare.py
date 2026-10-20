import json,sys,urllib.request,urllib.parse,datetime as dt,traceback,re
from pathlib import Path
import numpy as np,pandas as pd
import fin_skills
from benchmarks.agent_study import factorial_data as fd,factorial_study as fs
from benchmarks.agent_study import trading_study as old,market_data as md,trading_runtime_v5 as ledger
from benchmarks.agent_study.factorial_tools import SKILLS
ROOT=Path(__file__).resolve().parents[1]
assert not (ROOT/'prepared.json').exists()
for p in (ROOT/'source/benchmarks/agent_study').glob('factorial_*.py'):compile(p.read_text(),str(p),'exec')
idx={x['name']:x for x in fin_skills.catalog()}
corpus={s:dict(description=idx[s]['description'],text=fin_skills.load(s),references=fin_skills.references(s)) for s in SKILLS}
old.write(ROOT/'corpus.json',corpus)
hits=[]
for name,entry in corpus.items():
    for file,text in [('SKILL.md',entry['text']),*entry['references'].items()]:
        for i,line in enumerate(text.splitlines(),1):
            if re.search(r'\b(?:'+ '|'.join(fd.ORIGINAL+fd.TRANSFER)+r')\b',line):hits.append(dict(skill=name,file=file,line=i,text=line))
old.write(ROOT/'corpus-audit.json',dict(ticker_mentions=hits,skills=list(corpus),note='Review named asset mentions before model exposure; current documents may contain later general knowledge.'))
print('CORPUS_ASSET_MENTIONS '+json.dumps(hits),flush=True)
(ROOT/'datasets').mkdir(exist_ok=True)
original=fd.write_data(md.RAW,ROOT/'datasets/original',fd.ORIGINAL)
raw={};failure=None
for ticker in fd.TRANSFER:
    params=urllib.parse.urlencode(dict(period1=int(dt.datetime(2022,10,3,tzinfo=dt.timezone.utc).timestamp()),period2=int(dt.datetime(2026,9,26,tzinfo=dt.timezone.utc).timestamp()),interval='1d',events='div,splits'))
    url='https://query1.finance.yahoo.com/v8/finance/chart/'+ticker+'?'+params
    try:
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=35) as response:raw[ticker]=json.load(response)
        assert raw[ticker]['chart']['error'] is None and raw[ticker]['chart']['result']
        print('DOWNLOADED '+ticker,flush=True)
    except Exception as exc:
        failure=dict(ticker=ticker,error=repr(exc));break
old.write(ROOT/'transfer-download.json',dict(created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),source='Yahoo Finance chart API',requested=list(fd.TRANSFER),received=list(raw),failure=failure))
old.write(ROOT/'transfer-raw.json',raw)
baskets=['original']
if failure is None:
    transfer=fd.write_data(ROOT/'transfer-raw.json',ROOT/'datasets/transfer',fd.TRANSFER)
    assert list(transfer.index)==list(original.index),'transfer calendar differs; do not silently drop dates'
    baskets.append('transfer')
else:print('TRANSFER_PENDING '+json.dumps(failure),flush=True)
from qualify import qualify
report=qualify(ROOT,baskets,corpus)
old.write(ROOT/'qualification.json',report)
assert report['passed'], 'CPU qualification failed; inspect recorded probes'
plan=[]
for basket in baskets:
    for seed in (11,23,37):
        for family in ('7b','14b'):
            pair=ROOT/'batch'/f'{basket}-{family}-{seed}';fs.freeze(pair,basket,family,seed)
            p=old.load(pair/'protocol.json');plan.append(dict(root=str(pair),basket=basket,family=family,seed=seed,planned_decisions=p['planned_decisions'],protocol_sha256=old.sha(pair/'protocol.json')))
old.write(ROOT/'batch-plan.json',plan)
old.write(ROOT/'prepared.json',dict(complete=True,baskets=baskets,transfer_pending=failure,planned_decisions=sum(x['planned_decisions'] for x in plan),pairs=len(plan),qualification_sha256=old.sha(ROOT/'qualification.json'),plan_sha256=old.sha(ROOT/'batch-plan.json'),sources=fs.sources()))
print('PREPARED '+json.dumps(old.load(ROOT/'prepared.json')|{'sources':'see frozen map'}),flush=True)
