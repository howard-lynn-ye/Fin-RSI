"""Crash-released local RSI writer ownership, independent of model callbacks."""
from contextlib import contextmanager
import errno
import os


@contextmanager
def writer_lock(directory):
    # Never unlink this inode: another process may already have opened it.
    with (directory / '.writer.lock.v2').open('a+b') as stream:
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            acquire = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            release = lambda: (stream.seek(0), msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1))
        else:
            import fcntl
            acquire = lambda: fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            release = lambda: fcntl.flock(stream, fcntl.LOCK_UN)
        try:
            acquire()
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                raise FileExistsError('another RSI writer owns this archive') from exc
            raise
        try:
            yield
        finally:
            release()
