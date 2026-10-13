"""Public metadata and partial decision ledger; run read-only on Slurm."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

B = Path('/beacon-projects/radfm/wy891/fin-multisource-models-20261005')
OUTPUT = B/'campaign-inventory-20261013-snapshot4.json'
ROOTS = ['v9-access-matrix-20261013-r4', 'v9-access-matrix-20261013-r5',
         'v9-access-matrix-20261013-r8', 'v9-access-matrix-20261013-r9',
         'native-matrix-20261013-r2', 'native-matrix-20261013-r3',
         'native-matrix-20261013-r4', 'native-matrix-20261013-r5',
         'native-matrix-20261014-claude-r6', 'v9-access-matrix-20261013-r12',
         'v9-access-matrix-20261013-r13', 'v9-access-matrix-20261013-r15',
         'native-matrix-20261013-r7', 'native-matrix-20261013-r8', 'discovery-canary-20261013-r20']


def read(path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


rows, unstable = [], []
for batch in ROOTS:
    for root in sorted((B/batch/'pairs').glob('*')):
        if not (root/'protocol.json').is_file():
            continue
        try:
            p, digest = read(root/'protocol.json')
            row = dict(batch_root=batch, pair=root.name, model=p['model'], seed=p['seed'],
                track=p['batch'], protocol_sha256=digest, window=p['window'],
                interaction_revision=p.get('interaction_revision'), research_output=p.get('research_output'),
                planned_decisions_per_arm=p['decisions_per_arm'],
                completed_marker_present=(root/'completed.json').exists(),
                accounting_audited_in_this_inventory=False,
                phase_events=[], gate_receipts=[], infrastructure_failures=[],
                rejected_native_responses=[], partial_decisions={})
            for path in sorted((B/batch/'events').glob(root.name+'-*.json')):
                event, fingerprint = read(path)
                row['phase_events'].append(dict(phase=event['phase'], exit_code=event['exit_code'],
                    time_utc=event.get('time_utc'), job=event.get('job'), sha256=fingerprint))
            for name in ('model-access-preflight.json', 'decision-preflight.json', 'decoder-preflight.json'):
                path = root/name
                if path.exists():
                    receipt, fingerprint = read(path)
                    row['gate_receipts'].append(dict(name=name, sha256=fingerprint,
                        passed=receipt.get('passed'), cases=[{k:c.get(k) for k in
                            ('arm','required_tool','successful_deliveries','successful_receipts','decision_action','decision_completed','passed')}
                            for c in receipt.get('cases', [])]))
            for path in sorted((root/'infrastructure-failures').glob('*.json')):
                failure, fingerprint = read(path)
                response = failure.get('response', {})
                row['infrastructure_failures'].append(dict(name=path.name, sha256=fingerprint,
                    status=failure.get('status'), sampling_diagnostic=response.get('sampling_diagnostic'),
                    hardware=response.get('backend_metadata', {}).get('hardware')))
            for path in sorted((root/'native-exchanges').glob('*/response.json')):
                response, fingerprint = read(path)
                if response.get('success') is True:
                    continue
                reported_events = response.get('native_tool_events', [])
                provider_errors = [t for t in reported_events if t.get('step_update', {}).get('step_type') == 'error_message']
                tools = [t for t in reported_events if t not in provider_errors]
                row['rejected_native_responses'].append(dict(request_sha256=path.parent.name,
                    response_sha256=fingerprint, requested_model=response.get('requested_model'),
                    exit_code=response.get('exit_code'),
                    category=('provider_reported_error' if provider_errors else 'native_tool_use' if tools else 'provider_capacity'
                        if any(e.get('message') == 'Selected model is at capacity. Please try a different model.'
                               for e in response.get('provider_error_events', []))
                        else 'provider_or_transport_failure'),
                    provider_error_event_count=len(provider_errors)+len(response.get('provider_error_events', [])),
                    response_text_chars=len(response.get('text', '')),
                    provider_event_types=[e.get('type') for e in response.get('events', [])
                        if e.get('type') in ('error', 'turn.failed', 'turn.completed')],
                    native_tool_names=sorted({str(t.get('item', {}).get('tool') or
                        t.get('item', {}).get('type') or t.get('step_update', {}).get('tool_name') or
                        t.get('step_update', {}).get('step_type')) for t in tools})))
            row['access_attempts'] = {}
            for arm in ('raw', 'library'):
                path = root/'model-access'/f'{arm}.json'
                if path.is_file():
                    record, fingerprint = read(path)
                    row['access_attempts'][arm] = dict(sha256=fingerprint,
                        decision_action=record['decision_action'],
                        turns=[dict(parse=t['parse'], tool=t.get('tool'),
                            finish_reason=t['finish_reason'], ok=t['result'].get('ok'))
                            for t in record['turns']])
            for arm in ('raw', 'library'):
                decisions = []
                for path in sorted((root/'decisions'/arm).glob('*.json')):
                    decision, fingerprint = read(path)
                    decisions.append(dict(date=decision['date'], record_sha256=fingerprint,
                        action=decision['decision_action'], target=decision['target'],
                        holdings_before=decision['holdings_before'],
                        response_count=len(decision['turns']),
                        successful_tool_calls=[r['tool'] for r in decision['tool_calls'] if r.get('ok')]))
                row['partial_decisions'][arm] = sorted(decisions, key=lambda d:d['date'])
            rows.append(row)
        except (OSError, ValueError, KeyError) as exc:
            unstable.append(dict(batch_root=batch, pair=root.name, kind=type(exc).__name__,
                                 note='Snapshot omitted; possibly being written. No failure or zero return inferred.'))
out = dict(observed_utc=datetime.now(timezone.utc).isoformat(),
    scope='Preserved prior failures and in-progress records. Not a score or twenty-model completion claim.',
    publication='Complete cycles have separately audited full NAV/trade ledgers; these partial decisions must not be extrapolated or filled to the deadline.',
    private_content_excluded='Full prompts, raw news, model prose, native provider transcripts and credentials.',
    runs=rows, unstable_reads=unstable)
with OUTPUT.open('x') as stream:
    json.dump(out, stream, indent=2, allow_nan=False)
print(json.dumps(dict(output=str(OUTPUT), sha256=hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
    runs=len(rows), completed_markers=sum(r['completed_marker_present'] for r in rows),
    unstable_reads=len(unstable))), flush=True)
