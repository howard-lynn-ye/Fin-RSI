"""Check frozen decision/evidence bindings; this cannot verify archival vintages."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
from benchmarks.agent_study import multisource_model_study as study
from benchmarks.agent_study import trading_capabilities as cap
root = Path(sys.argv[2])
p = study.verify(root)
inputs = study.validate_receipt(root)
records = study.old.load(root/'evidence.json')
packets = study.old.load(root/'evidence-packets.json')
checked = 0
for arm in inputs['arms']:
    for i, day in enumerate(inputs['dates']):
        snapshot = cap.evidence_snapshot(records, packets[day])
        assert snapshot['as_of'] == day
        assert all(r['eligible_date'] <= day for r in snapshot['records'])
        digest = hashlib.sha256(cap.encoded(snapshot).encode()).hexdigest()
        decision = study.old.load(root/'decisions'/arm/f'{i:02d}.json')
        assert decision['date'] == day
        assert decision['evidence_snapshot_sha256'] == digest
        checked += 1
out = dict(passed=True, decisions_checked=checked,
           protocol_sha256=study.old.sha(root/'protocol.json'),
           same_evidence_snapshot_both_arms=True, declared_eligibility_cutoffs_passed=True,
           historical_availability_verified=False,
           scope='File/receipt binding and declared dates only. Not proof of original publication '
                 'time, absence of revised content, model memorization, or current-skill hindsight.')
Path(sys.argv[3]).write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps(out))
