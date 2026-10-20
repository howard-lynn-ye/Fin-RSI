import sys
from pathlib import Path
from benchmarks.agent_study import factorial_study as fs,trading_study as old
ROOT=Path(__file__).resolve().parents[1]
plan=old.load(ROOT/'batch-plan.json');prepared=old.load(ROOT/'prepared.json')
assert old.sha(ROOT/'batch-plan.json')==prepared['plan_sha256']
item=plan[int(sys.argv[1])];pair=Path(item['root'])
assert old.sha(pair/'protocol.json')==item['protocol_sha256']
fs.run(pair)
