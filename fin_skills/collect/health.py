"""Local collector ownership and read-only health probes; no background thread."""
from __future__ import annotations

import errno
import os
import socket
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from .model import utcnow


def _lock(handle, *, shared=False):
    handle.seek(0)
    if os.name == 'nt':
        import msvcrt
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBRLCK if shared else msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)


def _unlock(handle):
    handle.seek(0)
    if os.name == 'nt':
        import msvcrt
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle, fcntl.LOCK_UN)


def _busy(exc):
    return exc.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK)


@contextmanager
def ownership(store, mode):
    """Serialize collectors on one local DB. The OS releases ownership on a crash.

    Keep the sidecar inode: deleting a lock file could allow two owners. Network
    filesystems and copies on another host are not a supported liveness oracle.
    """
    if getattr(store, '_collector_active', False):
        raise ValueError('another collector already owns this database')
    handle = None
    if store.path != ':memory:':
        handle = open(store.path + '.collector.lock', 'a+b')
        try:
            _lock(handle)
        except OSError as exc:
            handle.close()
            if _busy(exc):
                raise ValueError('another collector already owns this database') from exc
            raise
    store._collector_active = True
    record = dict(session=uuid4().hex, mode=mode, state='running', host=socket.gethostname(),
                  pid=os.getpid(), started_at=utcnow(), finished_at=None)
    try:
        if handle is not None and Path(store.path + '.collector.lock').stat().st_size == 0:
            handle.write(b'0')
            handle.flush()
        store.record_runtime(record)
        try:
            yield
        except BaseException:
            record['state'] = 'interrupted'
            raise
        else:
            record['state'] = 'stopped'
        finally:
            record['finished_at'] = utcnow()
            store.record_runtime(record)
    finally:
        store._collector_active = False
        if handle is not None:
            _unlock(handle)
            handle.close()


def runner_status(store):
    """Probe an existing sidecar without creating, writing, or trusting a PID."""
    record = store.runtime_record()
    if record is None:
        return {'status': 'unobserved', 'active': None,
                'reason': 'no instrumented collector has run against this database'}
    if record.get('host') != socket.gethostname():
        return {**record, 'status': 'unknown', 'active': None,
                'reason': 'runtime belongs to another host; inspect it on that host'}
    active = getattr(store, '_collector_active', False)
    if store.path != ':memory:':
        try:
            with open(store.path + '.collector.lock', 'rb') as handle:
                try:
                    _lock(handle, shared=True)
                except OSError as exc:
                    if not _busy(exc):
                        raise
                    active = True
                else:
                    _unlock(handle)
                    active = False
        except OSError as exc:
            return {**record, 'status': 'unknown', 'active': None,
                    'reason': f'cannot inspect collector ownership: {type(exc).__name__}'}
    status = ('running' if record['state'] == 'running' else 'starting_or_stopping') if active else (
        'interrupted' if record['state'] == 'running' else record['state'])
    return {**record, 'status': status, 'active': active}
