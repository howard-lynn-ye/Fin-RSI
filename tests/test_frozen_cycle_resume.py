"""CPU fault injection only: real decision validation, ledger and independent audit.

Synthetic model answers are never reported as investment/model performance.
"""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts import frozen_cycle_resume as resume
from benchmarks.agent_study import trading_runtime_v8 as v8
from test_trading_terminal_v8 import Controller, response


@pytest.fixture(params=['v8', 'v9'])
def pair(tmp_path, monkeypatch, request):
    pytest.importorskip('bs4', reason='accounting fixtures require the optional study extra')
    from benchmarks.agent_study import multisource_model_study as study
    root, output = tmp_path / 'pair', tmp_path / 'output'
    root.mkdir()
    dates = pd.bdate_range('2025-01-02', periods=6).strftime('%Y-%m-%d')
    prices = pd.DataFrame(100., index=dates, columns=study.md.TICKERS)
    prices['SPY'] = [100., 102., 109., 103., 112., 120.]
    (root / 'data').mkdir()
    prices.to_csv(root / 'data' / study.md.HIDDEN)
    protocol = dict(interface=request.param, batch='matrix_hf', model=['fixture', 'synthetic', 'v1'],
        seed=11, max_tokens=1024, final_output={'transport': 'text'},
        universe=list(study.md.TICKERS), decisions_per_arm=2,
        initial_capital=100000., capital_currency='USD', cost_bps=study.old.COST_BPS,
        window=[dates[0], dates[-1]], limits=['Synthetic source-verification fixture'])
    inputs = dict(arms=['raw', 'library'], picks=[0, 3], dates=[dates[0], dates[3]])
    packets = {dates[pick]: dict(as_of=dates[pick], records=[]) for pick in inputs['picks']}
    resume.write_once(root / 'protocol.json', protocol)
    resume.write_once(root / 'inputs.json', inputs)
    resume.write_once(root / 'evidence.json', [])
    resume.write_once(root / 'evidence-packets.json', packets)
    for name in ('qualification/qualification.json', 'capability-qualification.json',
                 'decision-qualification.json', 'rag-qualification.json'):
        resume.write_once(root / name, {'synthetic': True})
    binds = resume.binding(root, protocol)
    resume.write_once(root / 'inference-started.json', binds)
    records = {}
    returns = prices.pct_change().fillna(0.)
    for arm in inputs['arms']:
        holdings = dict.fromkeys(study.md.TICKERS, 0.)
        for i, pick in enumerate(inputs['picks']):
            packet = packets[dates[pick]]
            # Keep an actual failed decision to ensure recovery never fabricates a hold.
            tool, args = ('submit', {'weights': {'SPY': -.5}}) if (arm, i) == ('raw', 1) else (
                ('submit', {'weights': {'SPY': .6}}) if i == 0 else ('hold', {}))
            runtime = study.terminal_runtime(request.param)
            record = runtime.decide(lambda _, t=tool, a=args: response(t, a), Controller(arm),
                                    'Synthetic decision' + study.evidence_text(packet))
            for turn in record['turns']:
                turn['backend_metadata'] = {'identity': dict(model='synthetic', revision='v1', max_tokens=1024)}
            snapshot = v8.evidence_snapshot([], packet)
            record.update(index=pick, date=dates[pick], holdings_before=dict(holdings),
                evidence_packet_sha256=study.digest(packet),
                evidence_snapshot_sha256=hashlib.sha256(v8.encoded(snapshot).encode()).hexdigest())
            records[f'decisions/{arm}/{i:02d}.json'] = record
            end = inputs['picks'][i+1]+1 if i+1 < len(inputs['picks']) else pick+1
            for k in range(pick+1, end):
                holdings, _ = study.old.drift(holdings, returns.iloc[k])
                if k == pick+1 and record['target'] is not None:
                    holdings = dict(record['target'])
    calls = []
    fail = {'after': None}
    def runner(where):
        assert where == root
        for name, record in records.items():
            if not (root / name).exists():
                calls.append(name)
                study.old.write(root / name, record)
                if len(calls) == fail['after']:
                    raise InterruptedError('simulated allocation ended')
        study.old.write(root / 'inference-receipt.json', dict(binds,
            decisions={name: resume.sha(root / name) for name in records}))
    monkeypatch.setattr(study, 'run', runner)
    # Only fixture freeze verification and the model backend are substituted. Production
    # score/validate_receipt/decision validation/independent accounting run unchanged.
    monkeypatch.setattr(study, 'verify', lambda _: protocol)
    def export(destination):
        assert resume.load(root / 'independent-model-audit.json')['passed']
        scores = resume.load(root / 'scores.json')
        ledgers = [dict(arm=arm, **resume.load(root / 'nav' / f'{arm}.json')) for arm in inputs['arms']]
        return dict(summary=dict(observed_utc=str(destination),
            all_expected_pairs_complete_and_valid=True, pairs=[scores]), ledgers=ledgers)
    def recover():
        return resume.recover(study, root, output, protocol, job='20', previous='19', export=export)
    return root, output, protocol, records, calls, fail, recover


def test_interrupted_inference_preserves_first_decision_and_exact_ledger(pair):
    from benchmarks.agent_study import multisource_model_study as study
    root, output, protocol, records, calls, fail, recover = pair
    fail['after'] = 1
    with pytest.raises(InterruptedError):
        recover()
    first = root / next(iter(records))
    original = first.read_bytes()
    fail['after'] = None
    recover()
    assert first.read_bytes() == original and len(calls) == 4
    assert resume.load(root / 'decisions/raw/01.json')['decision_action'] == 'failed'
    prices = pd.read_csv(root / 'data' / study.md.HIDDEN, index_col=0)
    for arm in ('raw', 'library'):
        nav, trades = study.runtime.ledger(prices, [0, 3],
            [records[f'decisions/{arm}/{i:02d}.json']['target'] for i in range(2)])
        assert resume.load(root / 'nav' / f'{arm}.json') == dict(nav=nav.to_dict(), trades=trades)
        expected_return = 100 * (float(nav.iloc[-1]) - 1)
        assert resume.load(root / 'scores.json')['paths'][arm]['return_rate_pct'] == pytest.approx(expected_return)
    assert len(list((output / 'recovery').glob('*-interrupted.json'))) == 1
    before = resume.inventory(root)
    public_before = {p.name: p.read_bytes() for p in (output / 'public').glob('*.json')}
    recover()
    assert before == resume.inventory(root) and len(calls) == 4
    assert public_before == {p.name: p.read_bytes() for p in (output / 'public').glob('*.json')}


def test_complete_decisions_finalize_without_model_loading(pair):
    root, output, protocol, records, calls, fail, recover = pair
    for name, record in records.items():
        resume.write_once(root / name, record)
    recover()
    assert calls == [] and resume.load(root / 'completed.json')


def test_interrupted_score_keeps_existing_nav_without_rerunning_inference(pair, monkeypatch):
    from benchmarks.agent_study import multisource_model_study as study
    root, output, protocol, records, calls, fail, recover = pair
    original_score = study.score
    def broken(where):
        prices = pd.read_csv(where / 'data' / study.md.HIDDEN, index_col=0)
        nav, trades = study.runtime.ledger(prices, [0, 3],
            [records[f'decisions/raw/{i:02d}.json']['target'] for i in range(2)])
        study.old.write(where / 'nav/raw.json', dict(nav=nav.to_dict(), trades=trades))
        raise InterruptedError('score interrupted')
    monkeypatch.setattr(study, 'score', broken)
    with pytest.raises(InterruptedError):
        recover()
    before = resume.inventory(root)
    # Reproduce the original score restart defect on the same retained evidence.
    with pytest.raises(FileExistsError):
        original_score(root)
    monkeypatch.setattr(study, 'score', original_score)
    recover()
    resume.preserve(root, before)
    assert len(calls) == 4


def test_failed_export_can_resume_without_regenerating_decisions(pair):
    from benchmarks.agent_study import multisource_model_study as study
    root, output, protocol, records, calls, fail, recover = pair
    def broken(_):
        raise InterruptedError('export interrupted')
    with pytest.raises(InterruptedError):
        resume.recover(study, root, output, protocol, job='20', previous='19', export=broken)
    before = resume.inventory(root)
    recover()
    resume.preserve(root, before)
    assert len(calls) == 4


@pytest.mark.parametrize('damage', ['hole', 'holdings', 'date', 'evidence', 'backend', 'truncated',
                                   'receipt', 'score', 'infrastructure', 'missing_start'])
def test_bad_partial_artifacts_stop_before_new_inference(pair, damage):
    root, output, protocol, records, calls, fail, recover = pair
    first = next(iter(records))
    record = records[first]
    if damage == 'hole':
        first = 'decisions/raw/01.json'
    elif damage == 'holdings':
        record['holdings_before']['SPY'] = .8
    elif damage == 'date':
        record['date'] = '1900-01-01'
    elif damage == 'evidence':
        record['evidence_packet_sha256'] = 'wrong'
    elif damage == 'backend':
        record['turns'][0]['backend_metadata']['identity']['model'] = 'another model'
    elif damage in ('receipt', 'score', 'infrastructure'):
        name = {'receipt': 'inference-receipt.json', 'score': 'scores.json',
                'infrastructure': 'infrastructure-failures/error.json'}[damage]
        resume.write_once(root / name, {'failed': True})
    elif damage == 'missing_start':
        (root / 'inference-started.json').unlink()
    resume.write_once(root / first, record)
    if damage == 'truncated':
        (root / first).write_text('{"date":')
    with pytest.raises(ValueError):
        recover()
    assert calls == []


def test_sealed_records_cannot_be_changed_between_recoveries(pair):
    root, output, protocol, records, calls, fail, recover = pair
    fail['after'] = 1
    with pytest.raises(InterruptedError):
        recover()
    first = root / next(iter(records))
    first.write_bytes(first.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='previously recorded artifact changed'):
        recover()
    assert len(calls) == 1


def test_unpublished_temporary_file_survives_without_becoming_a_decision(pair):
    root, output, protocol, records, calls, fail, recover = pair
    temporary = root / 'decisions/raw/.resume-write-interrupted'
    temporary.parent.mkdir(parents=True)
    temporary.write_bytes(b'{"incomplete":')
    recover()
    assert temporary.read_bytes() == b'{"incomplete":' and len(calls) == 4


def test_published_artifact_tampering_blocks_repeat_export(pair):
    root, output, protocol, records, calls, fail, recover = pair
    recover()
    path = output / 'public/ledgers.json'
    path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='previously published export changed'):
        recover()
    assert len(calls) == 4


def test_atomic_publication_and_nonoverwrite(tmp_path, monkeypatch):
    path = tmp_path / 'record.json'
    original_link = resume.os.link
    def broken(*_):
        raise OSError('filesystem failure')
    monkeypatch.setattr(resume.os, 'link', broken)
    with pytest.raises(OSError):
        resume.write_once(path, {'decision': 1})
    assert not path.exists() and not list(tmp_path.glob('.resume-write-*'))
    monkeypatch.setattr(resume.os, 'link', original_link)
    path.write_text('{"decision":1}\n')
    resume.write_once(path, {'decision': 1})
    assert path.read_text() == '{"decision":1}\n'
    with pytest.raises(ValueError, match='differs'):
        resume.write_once(path, {'decision': 2})


@pytest.mark.parametrize('value', ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '{"a":1e999}', '{"a":'])
def test_invalid_json_is_never_repaired_silently(tmp_path, value):
    path = tmp_path / 'record.json'
    path.write_text(value)
    with pytest.raises(ValueError):
        resume.load(path)


def test_single_writer_lock_releases_after_exception(tmp_path):
    with pytest.raises(InterruptedError):
        with resume.locked(tmp_path):
            with pytest.raises(RuntimeError, match='another recovery writer'):
                with resume.locked(tmp_path):
                    pass
            raise InterruptedError()
    with resume.locked(tmp_path):
        pass


@pytest.mark.parametrize('rows', ['19|fin-32b-cycle33|PENDING\n20|fin-recovery|RUNNING',
    '20|fin-recovery|RUNNING\n21|fin-other|PENDING', '20|fin-recovery|PENDING', '', 'bad'])
def test_scheduler_rejects_duplicate_or_unallocated_work(rows):
    with pytest.raises(ValueError):
        resume.sole_job(rows, '20', '19')


def test_scheduler_accepts_single_allocation_without_touching_other_projects():
    resume.sole_job('20|fin-recovery|RUNNING\n30|unrelated-project|RUNNING', '20', '19')


@pytest.mark.parametrize('changed', ['protocol', 'verification', 'source', 'exporter'])
def test_changed_frozen_configuration_is_rejected_before_source_import(tmp_path, monkeypatch, changed):
    root, source = tmp_path / 'pair', tmp_path / 'source'
    root.mkdir()
    source.mkdir()
    resume.write_once(root / 'protocol.json', {'frozen': True})
    program = source / 'original.py'
    program.write_text('unchanged source')
    verification, exporter = tmp_path / 'verification.json', tmp_path / 'exporter.py'
    resume.write_once(verification, {'sha256': {'original.py': resume.sha(program)}})
    exporter.write_text('unchanged exporter')
    config = dict(root=str(root), source=str(source), output=str(tmp_path / 'output'),
        original_job='19', protocol_sha256=resume.sha(root / 'protocol.json'),
        verification=str(verification), verification_sha256=resume.sha(verification),
        exporter=str(exporter), exporter_sha256=resume.sha(exporter))
    config_file = tmp_path / 'config.json'
    resume.write_once(config_file, config)
    target = dict(protocol=root / 'protocol.json', verification=verification,
                  source=program, exporter=exporter)[changed]
    target.write_text(target.read_text() + '\n')
    monkeypatch.setenv('SLURM_JOB_ID', '20')
    monkeypatch.setattr(resume.sys, 'argv', ['resume', '--config', str(config_file), '--previous-job', '19'])
    monkeypatch.setattr(resume.subprocess, 'check_output', lambda *a, **k: '20|fin-recovery|RUNNING')
    def forbidden(*_):
        pytest.fail('changed source must never be imported')
    monkeypatch.setattr(resume.importlib, 'import_module', forbidden)
    with pytest.raises(ValueError, match='differs'):
        resume.main()


def test_scoring_only_does_not_require_torch_or_gpu(monkeypatch):
    monkeypatch.setattr(resume.subprocess, 'check_output', lambda *a, **k: 'AllocTRES=cpu=8,mem=64G')
    monkeypatch.setitem(resume.sys.modules, 'torch', None)
    assert resume.allocation('20', False)['devices'] == []


@pytest.mark.parametrize('sizes', [[], [48], [48, 48], [80], [141]])
def test_allocation_keeps_frozen_single_device_layout(monkeypatch, sizes):
    monkeypatch.setattr(resume.subprocess, 'check_output', lambda *a, **k: 'fixture allocation')
    cuda = SimpleNamespace(device_count=lambda: len(sizes),
        get_device_properties=lambda i: SimpleNamespace(name='synthetic GPU', total_memory=sizes[i]*1024**3))
    monkeypatch.setitem(resume.sys.modules, 'torch', SimpleNamespace(cuda=cuda))
    if len(sizes) == 1 and sizes[0] >= 80:
        assert len(resume.allocation('20', True)['devices']) == 1
    else:
        with pytest.raises(ValueError, match='single-device'):
            resume.allocation('20', True)
