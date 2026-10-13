"""Qualification gates retain failed cases and accept genuine HOLD decisions."""
import json
import pytest

from benchmarks.agent_study import multisource_model_study as study
from benchmarks.agent_study import trading_preflight_v8 as gate, trading_runtime_v8 as v8
from test_trading_terminal_v8 import Controller, response


def test_cases_cover_both_arms_and_calendar_positions_without_return_selection():
    assert gate.cases(list(range(44))) == [
        ('raw', 0), ('library', 0), ('raw', 22), ('library', 22), ('raw', 43), ('library', 43)]
    assert gate.cases([0]) == [('raw', 0), ('library', 0)]
    with pytest.raises(ValueError):
        gate.cases([])


@pytest.fixture
def preflight(tmp_path, monkeypatch):
    protocol = dict(interface='v8', model=['synthetic', 'fixture', 'no-real-model'])
    monkeypatch.setattr(study, 'verify', lambda _: protocol)
    v8.old.write(tmp_path / 'protocol.json', protocol)
    v8.old.write(tmp_path / 'inputs.json', dict(picks=[0, 1, 2]))
    rows = []
    for arm, i in gate.cases([0, 1, 2]):
        record = v8.decide(lambda _: response('hold'), Controller(arm), 'fixture')
        record.update(protocol_sha256=v8.old.sha(tmp_path / 'protocol.json'),
                      arm=arm, decision_index=i, date=f'2025-01-0{i + 1}')
        path = tmp_path / 'interface-preflight' / f'{arm}-{i:02d}.json'
        v8.old.write(path, record)
        rows.append(dict(record=path.relative_to(tmp_path).as_posix(), sha256=v8.old.sha(path),
                         arm=arm, decision_index=i, date=record['date'], decision_action='hold',
                         decision_completed=True))
    report = dict(protocol_sha256=v8.old.sha(tmp_path / 'protocol.json'), model=protocol['model'],
                  cases=rows, passed=True)
    v8.old.write(tmp_path / 'decision-preflight.json', report)
    return tmp_path, report


def test_explicit_holds_qualify_without_forcing_trades(preflight):
    root, _ = preflight
    assert gate.require(root)['passed']


def test_missing_arm_or_date_cannot_be_hidden_in_success_report(preflight):
    root, report = preflight
    report['cases'].pop()
    (root / 'decision-preflight.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='incomplete'):
        gate.require(root)


def test_failed_decision_blocks_full_run_even_if_report_claims_success(preflight):
    root, report = preflight
    row = report['cases'][0]
    path = root / row['record']
    record = v8.decide(lambda _: response('read_market'), Controller(row['arm']), 'fixture')
    record.update(protocol_sha256=report['protocol_sha256'], arm=row['arm'],
                  decision_index=row['decision_index'], date=row['date'])
    path.write_text(json.dumps(record))
    row.update(sha256=v8.old.sha(path), decision_action='failed', decision_completed=False)
    (root / 'decision-preflight.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='did not qualify'):
        gate.require(root)
    assert path.exists()


def test_changed_original_record_invalidates_qualification(preflight):
    root, report = preflight
    path = root / report['cases'][0]['record']
    path.write_text('{}')
    with pytest.raises(ValueError, match='record changed'):
        gate.require(root)
