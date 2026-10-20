from pathlib import Path
from datetime import datetime,timezone
from benchmarks.agent_study import trading_study as old
ROOT=Path(__file__).resolve().parents[1]
assert not (ROOT/'smoke').exists() and not (ROOT/'pipeline-receipt.json').exists()
for p in (ROOT/'scripts').glob('*.py'):compile(p.read_text(),str(p),'exec')
from check_scoring import check
old.write(ROOT/'scoring-qualification.json',check(ROOT))
files={p.relative_to(ROOT).as_posix():old.sha(p) for p in sorted((ROOT/'scripts').iterdir()) if p.is_file()}
files['ANALYSIS_PLAN.md']=old.sha(ROOT/'ANALYSIS_PLAN.md')
old.write(ROOT/'pipeline-receipt.json',dict(files=files,created_utc=datetime.now(timezone.utc).isoformat()))
review=old.load(ROOT/'corpus-audit.json');assert not review['ticker_mentions']
old.write(ROOT/'corpus-reviewed.json',dict(approved_for_exposure=True,sha256=old.sha(ROOT/'corpus.json'),scope='Eight predeclared skills and Markdown references: no exact original/transfer ticker mentions detected. No asset-specific performance examples detected by this check; this is not proof against training contamination. Current-date documents are declared, not historical knowledge.'))
print('PIPELINE_FROZEN',flush=True)
