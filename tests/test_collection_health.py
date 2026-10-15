"""Collector failures must remain visible across CLI, restarts and agent retrieval."""
import json
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from fin_skills.collect import Batch, Collector, FetchError, Store, Watch
from fin_skills.collect.__main__ import main
from fin_skills.collect.feeds import RSSSource


def configure(path, *, enabled=True):
    with Store(path) as store:
        store.put_watch(Watch('news', 'rss', 'https://example.com/feed', 10, enabled=enabled))


def backend(monkeypatch, outcome):
    def fetch(*_):
        if outcome == 'error':
            raise FetchError('fixture unavailable', retry_after=3600)
        return Batch([], warnings=['fixture incomplete'] if outcome == 'partial' else [])
    monkeypatch.setattr(RSSSource, 'fetch', fetch)


@pytest.mark.parametrize('command', [['once'], ['run', '--cycles', '1']])
@pytest.mark.parametrize('outcome,expected', [('error', 1), ('partial', 1), ('ok', 0)])
def test_cli_reports_source_outcome(tmp_path, monkeypatch, capsys, command, outcome, expected):
    path = tmp_path / 'events.db'
    configure(path)
    backend(monkeypatch, outcome)
    assert main(['--database', str(path), *command]) == expected
    output = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    if command[0] == 'run':
        assert output[-1]['type'] == 'run_summary'
        assert output[-1]['failed_cycles'] == expected
        assert output[-1]['exit_code'] == expected


@pytest.mark.parametrize('command', [['once'], ['run', '--cycles', '1']])
def test_backoff_skip_cannot_hide_prior_failure(tmp_path, monkeypatch, capsys, command):
    path = tmp_path / 'events.db'
    configure(path)
    backend(monkeypatch, 'error')
    assert main(['--database', str(path), 'once']) == 1
    capsys.readouterr()
    monkeypatch.setattr(RSSSource, 'fetch', lambda *_: pytest.fail('backoff was ignored'))
    assert main(['--database', str(path), *command]) == 1
    output = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert output[0]['watches'] == []
    assert output[-1]['health']['watches'][0]['status'] == 'error'


@pytest.mark.parametrize('configured', [False, True])
def test_idle_does_not_report_success(tmp_path, capsys, configured):
    path = tmp_path / 'events.db'
    if configured:
        configure(path, enabled=False)
    assert main(['--database', str(path), 'run', '--cycles', '1']) == 1
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert summary['health']['status'] == 'idle'


def test_bounded_recovery_does_not_erase_earlier_failure(tmp_path, monkeypatch, capsys):
    path = tmp_path / 'events.db'
    configure(path)
    calls = []
    def fetch(*_):
        calls.append(1)
        if len(calls) == 1:
            raise FetchError('first attempt failed')
        return Batch([])
    monkeypatch.setattr(RSSSource, 'fetch', fetch)
    monkeypatch.setattr(Store, 'due', lambda *_: True)
    monkeypatch.setattr('fin_skills.collect.runtime.threading.Event.wait', lambda *_: None)
    assert main(['--database', str(path), 'run', '--cycles', '2']) == 1
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert summary['failed_cycles'] == 1 and summary['cycles'] == 2
    assert summary['health']['data_ready']


def test_tool_reports_skipped_failure_and_mixed_source_health(tmp_path, monkeypatch):
    from fin_skills.tools.collection import collect_once
    path = tmp_path / 'events.db'
    configure(path)
    with Store(path) as store:
        store.put_watch(Watch('good', 'rss', 'https://example.com/other'))
    def fetch(_, watch, state):
        if watch.id == 'news':
            raise FetchError('fixture unavailable', retry_after=3600)
        return Batch([])
    monkeypatch.setattr(RSSSource, 'fetch', fetch)
    first = collect_once(str(path))
    assert {r['status'] for r in first['watches']} == {'ok', 'error'}
    skipped = collect_once(str(path))
    assert skipped['watches'] == [] and not skipped['health']['data_ready']
    assert {w['status'] for w in skipped['health']['watches']} == {'ok', 'error'}


def test_poll_freshness_error_precedence_and_recovery(tmp_path, monkeypatch):
    path = tmp_path / 'events.db'
    configure(path)
    now = 1_700_000_000
    monkeypatch.setattr('fin_skills.collect.store.utcnow',
                        lambda: datetime.fromtimestamp(now, timezone.utc).isoformat())
    with Store(path) as store:
        watch = store.watches()[0]
        assert store.status(now=now)[0]['status'] == 'never_succeeded'
        # An empty response (including an unchanged feed) is a successful source check.
        store.save(watch, Batch([]), next_due=now + 10)
        assert store.status(now=now + 60)[0]['status'] == 'ok'
        assert store.status(now=now + 60.01)[0]['status'] == 'stale'
        assert store.status(now=now - 1)[0]['status'] == 'unknown'
        store.failure('news', 'fixture unavailable', now + 3600)
        current = store.status(now=now)[0]
        assert current['freshness'] == 'fresh' and current['status'] == 'error'
        assert store.status(now=now + 61)[0]['freshness'] == 'stale'
        store.save(watch, Batch([], warnings=['incomplete']), next_due=now + 3600)
        assert store.status(now=now)[0]['status'] == 'partial'
        store.save(watch, Batch([]), next_due=now + 10)
        assert store.health(now=now)['data_ready']
        assert not store.health(now=now)['healthy']


def test_health_requires_continuous_runner_and_success(tmp_path, monkeypatch):
    path = tmp_path / 'events.db'
    configure(path)
    backend(monkeypatch, 'ok')
    seen = []
    with Store(path) as store:
        collector = Collector(store)
        collector.run(cycles=1, on_cycle=lambda _: seen.append(store.health()))
        assert seen[0]['healthy'] and seen[0]['runner']['active']
        after = store.health()
        assert after['data_ready'] and not after['healthy']
        assert after['runner']['status'] == 'stopped'
        backend(monkeypatch, 'error')
        store.db.execute('UPDATE watches SET next_due=0')
        store.db.commit()
        collector.run(cycles=1, on_cycle=lambda _: seen.append(store.health()))
        assert seen[-1]['runner']['active'] and not seen[-1]['healthy']


def test_nested_collectors_refused_before_fetch_and_exceptions_release_lock(tmp_path, monkeypatch):
    path = tmp_path / 'events.db'
    configure(path)
    backend(monkeypatch, 'ok')
    def callback(_):
        with Store(path) as other:
            with pytest.raises(ValueError, match='another collector'):
                Collector(other).once(force=True)
        raise RuntimeError('fixture sink failure')
    with Store(path) as store:
        with pytest.raises(RuntimeError, match='sink failure'):
            Collector(store).run(cycles=1, on_cycle=callback)
        assert store.health()['runner']['status'] == 'interrupted'
        assert not store.health()['runner']['active']
        assert Collector(store).once(force=True)['watches'][0]['status'] == 'ok'


def test_killed_process_is_interrupted_and_restartable(tmp_path):
    path, marker = tmp_path / 'events.db', tmp_path / 'ready'
    configure(path)
    child = r'''
import sys, time
from pathlib import Path
from fin_skills.collect import Batch, Collector, Store
class Source:
    def fetch(self, *_):
        return Batch([])
def ready(_):
    Path(sys.argv[2]).write_text('ready', encoding='utf-8')
    time.sleep(30)
with Store(sys.argv[1]) as store:
    Collector(store, backends={'rss': Source()}).run(on_cycle=ready)
'''
    # Windows venv python.exe can be a launcher whose exit leaves the real worker alive.
    # Launch the actual interpreter with this test environment's import paths instead.
    executable = sys._base_executable if sys.platform == 'win32' else sys.executable
    child = 'import sys\nsys.path[:] = ' + repr(sys.path) + '\n' + child
    process = subprocess.Popen([executable, '-S', '-c', child, str(path), str(marker)],
                               cwd=Path(__file__).resolve().parents[1],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 15
        while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(.05)
        assert marker.exists(), 'owned fixture collector did not become ready'
        with Store(path, read_only=True) as store:
            assert store.health()['runner']['pid'] == process.pid
            assert store.health()['healthy']
        # Only this test's child is terminated. Its kernel lock must disappear on exit.
        process.kill()
        process.wait(timeout=5)
        with Store(path) as store:
            health = store.health()
            assert health['runner']['status'] == 'interrupted'
            assert health['runner']['active'] is False
            assert health['data_ready'] and not health['healthy']
            class Source:
                def fetch(self, *_):
                    return Batch([])
            assert Collector(store, backends={'rss': Source()}).once(force=True)['watches']
        assert Path(str(path) + '.collector.lock').exists()
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)


def test_health_legacy_database_read_only_and_other_host_unknown(tmp_path):
    from fin_skills.tools.collection import collection_status
    path = tmp_path / 'events.db'
    configure(path)
    with Store(path) as store:
        store.db.execute('DROP TABLE collector_runtime')
        store.db.commit()
    before = path.read_bytes()
    report = collection_status(str(path))
    assert report['runner']['status'] == 'unobserved' and not report['healthy']
    assert path.read_bytes() == before
    assert not Path(str(path) + '.collector.lock').exists()
    with Store(path) as store:
        store.record_runtime(dict(host=socket.gethostname() + '-other', mode='run', state='running'))
        assert store.health()['runner']['status'] == 'unknown'


def test_health_cli_missing_db_no_creation_and_stopped_returns_failure(tmp_path, monkeypatch, capsys):
    path = tmp_path / 'missing.db'
    assert main(['--database', str(path), 'health']) == 1
    assert not path.exists()
    capsys.readouterr()
    configure(path)
    backend(monkeypatch, 'ok')
    assert main(['--database', str(path), 'once']) == 0
    capsys.readouterr()
    assert main(['--database', str(path), 'health']) == 1
    report = json.loads(capsys.readouterr().out)
    assert report['data_ready'] and report['runner']['status'] == 'stopped'


def test_search_surfaces_current_health_but_does_not_project_it_into_history(tmp_path, monkeypatch):
    from fin_skills.tools.collection import collection_search
    path = tmp_path / 'events.db'
    configure(path)
    backend(monkeypatch, 'error')
    with Store(path) as store:
        Collector(store).once()
    monkeypatch.setattr(RSSSource, 'fetch', lambda *_: pytest.fail('search must not fetch'))
    current = collection_search(str(path), 'news')
    assert current['collection_health'][0]['status'] == 'error'
    assert not current['collection_readiness']['data_ready']
    assert current['collection_runtime']['status'] == 'stopped'
    past = collection_search(str(path), 'news', as_of='2020-01-01T00:00:00Z')
    assert all(past[key] is None for key in
               ('collection_health', 'collection_runtime', 'collection_readiness'))
