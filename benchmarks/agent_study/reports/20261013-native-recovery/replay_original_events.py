import json
from pathlib import Path
from benchmarks.agent_study import native_relay as relay, trading_study as old
B=Path('/beacon-projects/radfm/wy891/fin-multisource-models-20261005')
path=B/'native-queue-20261013-v1/ea4016c3860bc1c5cfd9d235344058421265dd52352648d5fd143c112c90b885/response.json'
r=old.load(path)
parsed=relay.parse_events('codex_subscription','\n'.join(json.dumps(e) for e in r['events']),r['exit_code'])
assert r['success'] is False and parsed['success'] is True
assert parsed['text'] == r['text'] and not parsed['native_tool_events']
assert any(e.get('type')=='error' and e.get('message','').startswith('Reconnecting...') for e in parsed['diagnostic_events'])
report=dict(original_request=path.parent.name,original_response_sha256=old.sha(path),
    original_parser_success=r['success'],corrected_parser_success=parsed['success'],
    original_text_unchanged=True,native_tool_events=0,diagnostic_events=parsed['diagnostic_events'],
    relay_sha256=relay.source_digest(),scope='Read-only replay of the original provider event stream; no new model call, trade, return, or rewriting of the failed original experiment.')
old.write(B/'native-recovery-20261013-candidate6/actual-event-regression.json',report)
print(json.dumps(report),flush=True)