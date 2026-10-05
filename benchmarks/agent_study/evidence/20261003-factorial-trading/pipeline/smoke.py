import sys,json,tempfile
from pathlib import Path
from datetime import datetime,timezone
from benchmarks.agent_study import factorial_study as fs,factorial_runtime as rt,factorial_data as fd,trading_study as old,trading_study_v5 as v5
from benchmarks.agent_study.factorial_tools import ARMS
from benchmarks.agent_study.transformers_chat import TransformersChat
ROOT=Path(__file__).resolve().parents[1]
family=sys.argv[1];p=fs.verify(ROOT/'batch'/f'original-{family}-11')
q=old.load(ROOT/'qualification.json');assert q['passed'] and q['sources']==fs.sources()
review=old.load(ROOT/'corpus-reviewed.json');assert review['approved_for_exposure'] and review['sha256']==old.sha(ROOT/'corpus.json')
root=ROOT/'smoke'/family;root.mkdir(parents=True,exist_ok=False);(root/'tmp').mkdir()
backend=TransformersChat(p['model'][1],p['model'][2],max_tokens=p['max_tokens']);corpus=old.load(ROOT/'corpus.json')
inputs=old.load(ROOT/'batch'/f'original-{family}-11'/'inputs.json');date=inputs['decision_dates'][0];pick=inputs['decision_sessions'][0]
results={}
for arm in ARMS:
    backend.seed,backend.calls=11000,0
    with tempfile.TemporaryDirectory(dir=root/'tmp') as td:
        workspace=Path(td)/'visible';fd.visible(p['data_dir'],workspace,date)
        c=rt.Controller(root,workspace,arm,p['tickers'],11000,corpus)
        record=rt.decide(backend,c,v5.task(date,pick,dict.fromkeys(p['tickers'],0.)))
    old.write(root/f'{arm}.json',record)
    # Technical model transport only. Failed tasks and non-adoption remain outcomes.
    transport=bool(record['turns']) and not any(t['parse']=='backend-error' for t in record['turns']) and any(t.get('tool') is not None for t in record['turns'])
    results[arm]=dict(transport_passed=transport,submitted=record['submitted'],turns=len(record['turns']),tool_calls=record['tool_calls'],sha256=old.sha(root/f'{arm}.json'))
    print(json.dumps(dict(family=family,arm=arm,**results[arm])),flush=True)
old.write(root/'completed.json',dict(transport_passed=all(x['transport_passed'] for x in results.values()),family=family,results=results,sources=fs.sources(),completed_utc=datetime.now(timezone.utc).isoformat()))
assert all(x['transport_passed'] for x in results.values())
