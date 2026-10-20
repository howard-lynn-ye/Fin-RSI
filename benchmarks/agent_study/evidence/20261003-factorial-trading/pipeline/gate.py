import json,py_compile,sys
from pathlib import Path
from datetime import datetime,timezone
from benchmarks.agent_study import factorial_study as fs,trading_study as old
ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT/'scripts').glob('*.py'):compile(p.read_text(),str(p),'exec')
q=old.load(ROOT/'qualification.json');assert q['passed'] and q['sources']==fs.sources()
prepared=old.load(ROOT/'prepared.json');assert old.sha(ROOT/'batch-plan.json')==prepared['plan_sha256']
review=old.load(ROOT/'corpus-reviewed.json');assert review['approved_for_exposure'] and review['sha256']==old.sha(ROOT/'corpus.json')
receipt=old.load(ROOT/'pipeline-receipt.json')
for file,digest in receipt['files'].items():assert old.sha(ROOT/file)==digest,file
records={}
for family in ('7b','14b'):
    p=ROOT/'smoke'/family/'completed.json';r=old.load(p)
    assert r['transport_passed'] and r['sources']==fs.sources()
    for arm,cell in r['results'].items():assert old.sha(p.parent/f'{arm}.json')==cell['sha256']
    records[family]=dict(sha256=old.sha(p),results=r['results'])
for item in old.load(ROOT/'batch-plan.json'):fs.verify(Path(item['root']))
old.write(ROOT/'smoke-completed.json',dict(transport_passed=True,models=records,pipeline_sha256=old.sha(ROOT/'pipeline-receipt.json'),verified_utc=datetime.now(timezone.utc).isoformat(),scope='Model transport only; no minimum return, adoption or submission rate. Scripted CPU probes independently verify interface execution.'))
print('TRANSPORT_GATE_PASSED',flush=True)
