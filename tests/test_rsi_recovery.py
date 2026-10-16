"""RSI crash recovery preserves evidence, budget reservations and approved versions."""
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from fin_skills.rsi import Policy, RSILoop, Trace

PAST = '2025-01-01T00:00:00Z'
NOW = '2025-02-01T00:00:00Z'


def make(path, **kwargs):
    def forbidden(*_):
        pytest.fail('recovery must not call a model or evaluator')
    defaults = dict(model_id='fixture', protocol_id='fixture', available_at=PAST,
                    policy=Policy(enabled=True), proposer=forbidden, evaluator=forbidden)
    defaults.update(kwargs)
    return RSILoop(path, **defaults)


def observed(path):
    loop = make(path)
    loop.observe(Trace('t1', 'dev1', PAST, 'tool_contract', 'fixture failure', 'source-fixture'))
    return loop


@pytest.mark.parametrize('phase', ['proposal', 'evaluation'])
def test_real_crash_releases_writer_and_preserves_call_budget(tmp_path, phase):
    directory, marker = tmp_path / 'archive', tmp_path / 'ready'
    child = r'''
import json, os, sys, time
from pathlib import Path
from fin_skills.rsi import RSILoop, Policy, Trace
phase = sys.argv[3]
def block():
    Path(sys.argv[2]).write_text(str(os.getpid()), encoding='utf-8')
    time.sleep(30)
def propose(_):
    if phase == 'proposal': block()
    return json.dumps(dict(key='fix', surface='tool_usage', text='Inspect shapes.', rationale='fixture'))
def evaluate(*_): block()
loop = RSILoop(sys.argv[1], model_id='fixture', protocol_id='fixture',
    available_at='2025-01-01T00:00:00Z',
    policy=Policy(enabled=True, max_proposals=1 if phase == 'proposal' else 3, max_evaluations=4),
    proposer=propose, evaluator=evaluate)
loop.observe(Trace('t1', 'dev1', '2025-01-01T00:00:00Z', 'tool_contract', 'fixture failure', 'fixture'))
loop.improve(as_of='2025-02-01T00:00:00Z')
'''
    child = 'import sys\nsys.path[:] = ' + repr(sys.path) + '\n' + child
    interpreter = sys._base_executable if sys.platform == 'win32' else sys.executable
    process = subprocess.Popen([interpreter, '-S', '-c', child, str(directory), str(marker), phase],
        cwd=Path(__file__).resolve().parents[1], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 15
        while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(.05)
        assert marker.exists(), 'owned RSI fixture did not reach the callback'
        assert int(marker.read_text()) == process.pid
        loop = make(directory, policy=Policy(enabled=True,
                    max_proposals=1 if phase == 'proposal' else 3, max_evaluations=4))
        with pytest.raises(FileExistsError, match='writer'):
            loop.recover()
        original = (directory / 'events.jsonl').read_bytes()
        process.kill()
        process.wait(timeout=5)
        report = loop.recover()
        assert report['status'] == 'ready' and report['proposals_started'] == 1
        assert report['evaluations_started'] == (1 if phase == 'evaluation' else 0)
        assert (directory / 'events.jsonl').read_bytes() == original
        assert loop.instructions(as_of=NOW) == ''
        assert loop.improve(as_of=NOW).status == 'budget_exhausted'
        assert (directory / '.writer.lock.v2').exists()
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)


def test_legacy_marker_needs_explicit_host_confirmation_and_is_preserved(tmp_path):
    loop = observed(tmp_path)
    marker = tmp_path / '.writer.lock'
    marker.write_text('12345', encoding='utf-8')
    before = (tmp_path / 'events.jsonl').read_bytes()
    with pytest.raises(FileExistsError, match='legacy'):
        loop.recover()
    assert (tmp_path / 'events.jsonl').read_bytes() == before
    report = loop.recover(confirmed_legacy_writer_stopped=True)
    assert report['archived_legacy_lock']
    assert (tmp_path / report['legacy_backup']).read_text() == '12345'
    assert not marker.exists()
    assert loop.audit()[-1]['kind'] == 'legacy_writer_recovered'


@pytest.mark.parametrize('tail', [b'{"partial":', b'\xf0\x9f'])
def test_partial_tail_is_quarantined_only_by_explicit_recovery(tmp_path, tail):
    loop = observed(tmp_path)
    path = tmp_path / 'events.jsonl'
    prefix = path.read_bytes()
    path.write_bytes(prefix + tail)
    with pytest.raises(ValueError):
        loop.audit()
    report = loop.recover()
    assert (tmp_path / report['backup']).read_bytes() == prefix + tail
    assert path.read_bytes().startswith(prefix)
    assert loop.audit()[-1]['kind'] == 'journal_recovered'
    recovered = path.read_bytes()
    assert loop.recover()['status'] == 'ready'
    assert path.read_bytes() == recovered


def test_valid_final_record_without_newline_is_kept(tmp_path):
    loop = observed(tmp_path)
    path = tmp_path / 'events.jsonl'
    original = path.read_bytes()
    path.write_bytes(original[:-1])
    with pytest.raises(ValueError, match='unterminated'):
        loop.audit()
    loop.recover()
    assert path.read_bytes().startswith(original)
    assert len([e for e in loop.audit() if e['kind'] == 'trace']) == 1


@pytest.mark.parametrize('corruption', ['hash', 'complete_bad_record', 'header'])
def test_recovery_never_overrides_corruption_or_protocol(tmp_path, corruption):
    loop = observed(tmp_path)
    path = tmp_path / 'events.jsonl'
    original = path.read_bytes()
    if corruption == 'hash':
        path.write_bytes(original.replace(b'fixture failure', b'forged failure') + b'{')
    elif corruption == 'complete_bad_record':
        path.write_bytes(original + b'{"broken":\n')
    else:
        loop = make(tmp_path, protocol_id='changed')
        path.write_bytes(original + b'{')
    before = path.read_bytes()
    with pytest.raises(ValueError):
        loop.recover()
    assert path.read_bytes() == before
    assert not list(tmp_path.glob('recovery-*.bin'))


def test_recovery_does_not_unfreeze_or_create_disabled_archive(tmp_path):
    loop = observed(tmp_path / 'enabled')
    loop.freeze(as_of=NOW)
    assert loop.recover()['frozen']
    assert loop.improve(as_of=NOW).status == 'frozen'
    disabled = make(tmp_path / 'disabled', policy=Policy())
    assert disabled.recover()['status'] == 'disabled'
    assert not (tmp_path / 'disabled').exists()
