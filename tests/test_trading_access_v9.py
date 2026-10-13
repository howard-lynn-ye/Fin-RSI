"""Tool access must be demonstrated independently of valid investment output."""
import hashlib
import json

import pytest

from benchmarks.agent_study import trading_access_v9 as access
from benchmarks.agent_study import trading_runtime_v8 as v8, trading_runtime_v9 as v9
from test_trading_terminal_v8 import Controller, response


class AccessController(Controller):
    def __init__(self, arm, result=None):
        super().__init__(arm)
        self.result = {'ok': True, 'rows': [{'visible': True}]} if result is None else result

    def call(self, tool, args):
        self.calls.append(dict(tool=tool, ok=bool(self.result.get('ok'))))
        return dict(self.result)


def record(arm, *, tools=True, result=None, final='hold'):
    c = AccessController(arm, result)
    replies = ([response(dict(access.CASES)[arm], {'query': 'market evidence'}
                        if arm == 'library' else {})] if tools else []) + [response(final)]
    return v8.decide(lambda _: replies.pop(0), c, 'fixture',
                     prompt_factory=access.prompt, interface='v9')


@pytest.mark.parametrize('arm', ['raw', 'library'])
def test_early_hold_is_a_valid_investment_but_does_not_prove_access(arm):
    r = record(arm, tools=False)
    assert r['decision_completed']
    assert not access.inspect(r, arm)['passed']


@pytest.mark.parametrize('arm', ['raw', 'library'])
def test_successful_execution_and_delivery_qualify(arm):
    assert access.inspect(record(arm), arm)['passed']


@pytest.mark.parametrize('result', [{'ok': False, 'error': 'unavailable'},
                                 {'ok': True, 'payload': 'x' * 10000}])
def test_failed_or_undelivered_result_does_not_qualify(result):
    assert not access.inspect(record('library', result=result), 'library')['passed']


def test_missing_receipt_cannot_be_replaced_by_model_claim():
    r = record('library')
    r['tool_calls'] = []
    assert not access.inspect(r, 'library')['passed']


@pytest.fixture
def frozen_access(tmp_path, monkeypatch):
    pytest.importorskip('bs4')
    from benchmarks.agent_study import multisource_model_study as study
    day = '2025-01-02'
    p = dict(interface='v9', batch='matrix_hf', deadline='2026-09-25', seed=11,
             model=['fixture', 'fixture/model', 'no-real-weights'], max_tokens=1024,
             final_output=v8.schema.output_contract(), model_access_probe=access.contract())
    monkeypatch.setattr(study, 'verify', lambda _: p)
    study.old.write(tmp_path / 'protocol.json', p)
    study.old.write(tmp_path / 'inputs.json', {'dates': [day]})
    rows = [dict(category=c, series=c, id=c, event_date=day, eligible_date=day,
                 text='synthetic evidence') for c in ('news', 'psychology', 'behavior', 'officials')]
    packet = study.evidence_packet(rows, day)
    study.old.write(tmp_path / 'evidence.json', rows)
    study.old.write(tmp_path / 'evidence-packets.json', {day: packet})
    snapshot_sha = hashlib.sha256(v9.encoded(v9.evidence_snapshot(rows, packet)).encode()).hexdigest()
    entries = []
    for arm, _ in access.CASES:
        r = record(arm)
        r['initial_request'][1]['content'] = access.task(tmp_path, p)[1]
        r.update(arm=arm, date=day, protocol_sha256=study.old.sha(tmp_path / 'protocol.json'),
                 evidence_snapshot_sha256=snapshot_sha)
        for turn in r['turns']:
            turn['backend_metadata'] = {'identity': v8.schema.backend_identity(p)}
        path = tmp_path / 'model-access' / f'{arm}.json'
        study.old.write(path, r)
        entries.append(dict(record=path.relative_to(tmp_path).as_posix(), sha256=study.old.sha(path),
                            **access.inspect(r, arm)))
    report = dict(contract=access.contract(), protocol_sha256=study.old.sha(tmp_path / 'protocol.json'),
                  model=p['model'], cases=entries, passed=True)
    study.old.write(tmp_path / 'model-access-preflight.json', report)
    return tmp_path, p, report


def test_both_original_probe_records_required(frozen_access):
    root, _, report = frozen_access
    assert access.require(root) == report
    report['cases'].pop()
    (root / 'model-access-preflight.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='incomplete'):
        access.require(root)


@pytest.mark.parametrize('field', ['initial_request', 'evidence_snapshot_sha256', 'backend_metadata'])
def test_changed_inputs_and_model_identity_rejected(frozen_access, field):
    root, _, report = frozen_access
    row = report['cases'][1]
    path = root / row['record']
    r = v8.old.load(path)
    if field == 'initial_request':
        r[field][0]['content'] = 'different instructions'
    elif field == 'backend_metadata':
        r['turns'][0][field] = {'identity': {'model': 'different'}}
    else:
        r[field] = 'different'
    path.write_text(json.dumps(r))
    row['sha256'] = v8.old.sha(path)
    (root / 'model-access-preflight.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='inputs|identity'):
        access.require(root)


def test_no_forced_tool_usage_is_added_to_investment_prompt():
    assert access.contract()['required_during_investment'] is False
    assert 'UNSCORED INTERFACE CHECK' in access.prompt('library')
    assert 'UNSCORED INTERFACE CHECK' not in v9.system_prompt('library')
    assert 'Finish early whenever ready' in v9.system_prompt('library')

def test_unscored_instruction_survives_transition_to_final_stage():
    for index in (6, 7):
        p = access.prompt('library', index)
        assert 'UNSCORED INTERFACE CHECK' in p and 'Finish with one submit or hold' in p


def test_completed_submit_after_reader_is_not_misclassified_as_broken_access():
    c = AccessController('library')
    replies = [response('research_context', {'query': 'risk'}),
               response('submit', {'allocation': {'CASH': 1}})]
    r = v8.decide(lambda _: replies.pop(0), c, 'fixture', prompt_factory=access.prompt,
                  interface='v9')
    assert r['decision_action'] == 'submit'
    assert access.inspect(r, 'library')['passed']