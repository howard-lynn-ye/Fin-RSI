"""Qualification gates retain failed cases and accept genuine HOLD decisions."""
import json
import pytest

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
    pytest.importorskip('bs4', reason='multisource integration requires the study extra')
    from benchmarks.agent_study import multisource_model_study as study

    protocol = dict(interface='v8', model=['synthetic', 'fixture', 'no-real-model'],
                    max_tokens=1024, final_output=v8.schema.output_contract())
    monkeypatch.setattr(study, 'verify', lambda _: protocol)
    v8.old.write(tmp_path / 'protocol.json', protocol)
    v8.old.write(tmp_path / 'inputs.json', dict(picks=[0, 1, 2]))
    rows = []
    for arm, i in gate.cases([0, 1, 2]):
        record = v8.decide(lambda _: response('hold'), Controller(arm), 'fixture')
        for turn in record['turns']:
            turn['backend_metadata'] = {'identity': v8.schema.backend_identity(protocol)}
        record.update(protocol_sha256=v8.old.sha(tmp_path / 'protocol.json'),
                      arm=arm, decision_index=i, date=f'2025-01-0{i + 1}')
        path = tmp_path / 'interface-preflight' / f'{arm}-{i:02d}.json'
        v8.old.write(path, record)
        rows.append(dict(record=path.relative_to(tmp_path).as_posix(), sha256=v8.old.sha(path),
                         arm=arm, decision_index=i, date=record['date'], **gate.observations(record)))
    report = dict(protocol_sha256=v8.old.sha(tmp_path / 'protocol.json'), model=protocol['model'],
                  cases=rows, passed=True, interpretation=gate.interpretation(rows),
                  decoder_check=gate.require_decoder(tmp_path, protocol))
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
    for turn in record['turns']:
        turn['backend_metadata'] = {'identity': v8.schema.backend_identity(v8.old.load(root / 'protocol.json'))}
    record.update(protocol_sha256=report['protocol_sha256'], arm=row['arm'],
                  decision_index=row['decision_index'], date=row['date'])
    path.write_text(json.dumps(record))
    row.update(sha256=v8.old.sha(path), **gate.observations(record))
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


@pytest.mark.parametrize('interface', ['v8', 'v9'])
def test_run_uses_real_workspace_cutoffs_and_records_each_case(tmp_path, monkeypatch, interface):
    pytest.importorskip('bs4', reason='multisource integration requires the study extra')
    import pandas as pd
    from benchmarks.agent_study import multisource_model_study as study

    root = tmp_path / 'run'
    data = root / 'data'
    data.mkdir(parents=True)
    days = ['2025-01-02', '2025-01-03', '2025-01-06']
    protocol = dict(interface=interface, seed=11, deadline=days[-1],
                    max_tokens=1024, final_output=v8.schema.output_contract(),
                    model=['synthetic', 'fixture', 'no-real-model'])
    study.old.write(root / 'protocol.json', protocol)
    study.old.write(root / 'inputs.json', dict(picks=[0, 1, 2]))
    pd.DataFrame({t: [100., 101., 102.] for t in study.md.TICKERS}, index=days).to_csv(
        data / study.md.HIDDEN)
    pd.DataFrame([dict(date=day, ticker=t, open=100., high=101., low=99., close=100., volume=10)
                  for day in days for t in study.md.TICKERS]).to_csv(data / 'quotes.csv', index=False)
    pd.DataFrame(columns=['date', 'ticker', 'kind', 'value']).to_csv(
        data / 'corporate_actions.csv', index=False)
    records = [dict(category=c, series=c, id=c, event_date='2025-01-01',
                    eligible_date=days[0], text='synthetic evidence')
               for c in ('news', 'psychology', 'behavior', 'officials')]
    study.old.write(root / 'evidence.json', records)
    packets = {day: study.evidence_packet(records, day) for day in days}
    study.old.write(root / 'evidence-packets.json', packets)
    monkeypatch.setattr(study, 'verify', lambda _: protocol)
    runtime = study.terminal_runtime(interface)
    # Source/qualification files are outside this fixture. Workspace, truncation,
    # evidence, Controller, prompts, decision parsing and record verification are real.
    monkeypatch.setattr(runtime, 'require_qualification', lambda _: {'passed': True})
    seen = []
    def backend(messages):
        arm, index = gate.cases(days)[len(seen)]
        assert backend.seed == 11000 + index and backend.calls == 0
        visible, = (root / 'tmp').glob('*/visible')
        assert str(pd.read_csv(visible / 'quotes.csv').date.max()) == days[index]
        assert study.old.load(visible / 'evidence.json')['as_of'] == days[index]
        assert not (visible / study.md.HIDDEN).exists()
        assert messages[0]['content'] == runtime.system_prompt(arm)
        assert messages[1]['content'].endswith(study.evidence_text(packets[days[index]]))
        seen.append((arm, index))
        reply = response('hold') if index == 1 else response('submit', {
            'allocation': {'SPY': 1, 'CASH': 1}})
        reply['backend_metadata'] = {'identity': v8.schema.backend_identity(protocol)}
        return reply
    assert not (root / 'tmp').exists()
    report = gate.run(root, backend)
    assert report['passed'] and seen == gate.cases(days)
    assert len(report['cases']) == 6 and gate.require(root) == report
    assert not list((root / 'tmp').iterdir())
    assert [r['decision_action'] for r in report['cases']] == [
        'submit', 'submit', 'hold', 'hold', 'submit', 'submit']


def test_zero_tool_decisions_do_not_claim_library_use(preflight):
    root, report = preflight
    assert gate.require(root)['passed']
    assert report['interpretation']['library_execution_cases'] == 0
    assert not report['interpretation']['historical_availability_verified']
    assert all(r['research_status'] == 'not_attempted' for r in report['cases'])
    assert not any(r['library_execution_observed'] for r in report['cases'])


def test_successful_tool_delivery_is_separate_from_execution_and_submission():
    record = v8.decide(lambda _: response('hold'), Controller('library'), 'fixture')
    record['turns'].insert(0, dict(phase='research', tool='research_context',
        result=dict(ok=True, payload='x' * 5000)))
    record['tool_calls'] = [dict(tool='research_context', ok=True)]
    result = gate.observations(record)
    assert result['decision_completed'] and result['library_execution_observed']
    assert result['research_attempts'] == result['research_errors'] == 1
    assert result['research_status'] == 'all_attempts_failed'
    assert result['research_successes'] == 0


def test_discovery_and_failed_attempts_do_not_count_as_library_execution():
    record = v8.decide(lambda _: response('hold'), Controller('library'), 'fixture')
    record['tool_calls'] = [dict(tool='list_algorithms', ok=True),
                            dict(tool='library_attempt', library_tool='run_algorithm', ok=False)]
    assert not gate.observations(record)['library_execution_observed']


@pytest.mark.parametrize('tamper', ['case', 'interpretation'])
def test_cannot_upgrade_unobserved_library_use_or_unverified_history(preflight, tamper):
    root, report = preflight
    if tamper == 'case':
        report['cases'][0]['library_execution_observed'] = True
    else:
        report['interpretation']['historical_availability_verified'] = True
    (root / 'decision-preflight.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='observations|interpretation'):
        gate.require(root)
