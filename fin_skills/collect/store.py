"""Transactional local SQLite storage: watch state, revisions, and an alert outbox."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

from .model import Batch, Watch, utcnow


class Store:
    def __init__(self, path, *, read_only=False):
        self.path = str(path)
        if read_only:
            existing = Path(path).expanduser().resolve(strict=True)
            self.path = str(existing)
            self.db = sqlite3.connect(existing.as_uri() + '?mode=ro', uri=True, timeout=20)
            self.db.row_factory = sqlite3.Row
            return
        if self.path != ":memory:":
            Path(self.path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
            self.path = str(Path(self.path).expanduser().resolve())
        self.db = sqlite3.connect(self.path, timeout=20)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS watches (
                id TEXT PRIMARY KEY, config TEXT NOT NULL, state TEXT NOT NULL DEFAULT '{}',
                last_attempt TEXT, last_success TEXT, error TEXT, next_due REAL NOT NULL DEFAULT 0,
                failures INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS events (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, watch_id TEXT NOT NULL,
                event_id TEXT NOT NULL, hash TEXT NOT NULL, record TEXT NOT NULL,
                observed_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS latest (
                watch_id TEXT NOT NULL, event_id TEXT NOT NULL, seq INTEGER NOT NULL,
                PRIMARY KEY(watch_id, event_id));
            CREATE TABLE IF NOT EXISTS alerts (
                seq INTEGER PRIMARY KEY REFERENCES events(seq), acknowledged_at TEXT);
        """)

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def put_watch(self, watch: Watch):
        self.put_watches([watch])

    def put_watches(self, watches):
        watches = list(watches)
        if len({w.id for w in watches}) != len(watches):
            raise ValueError('watch ids must be unique within a configuration batch')
        with self.db:
            for watch in watches:
                encoded = json.dumps(asdict(watch), sort_keys=True)
                old = self.db.execute("SELECT config FROM watches WHERE id=?", (watch.id,)).fetchone()
                if old:
                    prior = json.loads(old[0])
                    if (prior['source'], prior['target'], prior['options']) != (
                            watch.source, watch.target, watch.options):
                        raise ValueError("source/target/options changed: use a new watch id to preserve provenance")
                self.db.execute("INSERT INTO watches(id,config) VALUES(?,?) ON CONFLICT(id) "
                                "DO UPDATE SET config=excluded.config", (watch.id, encoded))

    def watches(self):
        return [Watch(**json.loads(r[0])) for r in self.db.execute("SELECT config FROM watches ORDER BY id")]

    def state(self, watch_id):
        row = self.db.execute("SELECT state FROM watches WHERE id=?", (watch_id,)).fetchone()
        if row is None:
            raise KeyError(watch_id)
        return json.loads(row[0])

    def due(self, watch_id, now):
        return self.db.execute("SELECT next_due FROM watches WHERE id=?", (watch_id,)).fetchone()[0] <= now

    def save(self, watch: Watch, batch: Batch, *, next_due: float):
        """Commit records, outbox and cursor together; a failed batch advances nothing."""
        added = []
        with self.db:
            for event in batch.events:
                prior = self.db.execute("SELECT e.hash FROM latest l JOIN events e ON l.seq=e.seq "
                                        "WHERE l.watch_id=? AND l.event_id=?", (watch.id, event.id)).fetchone()
                if prior and prior[0] == event.content_hash:
                    continue
                cur = self.db.execute("INSERT INTO events(watch_id,event_id,hash,record,observed_at) "
                                      "VALUES(?,?,?,?,?)", (watch.id, event.id, event.content_hash,
                                      json.dumps(event.to_dict(), sort_keys=True, allow_nan=False), event.observed_at))
                seq = cur.lastrowid
                self.db.execute("INSERT INTO latest VALUES(?,?,?) ON CONFLICT(watch_id,event_id) "
                                "DO UPDATE SET seq=excluded.seq", (watch.id, event.id, seq))
                self.db.execute("INSERT INTO alerts(seq) VALUES(?)", (seq,))
                added.append({"seq": seq, "watch_id": watch.id, **event.to_dict()})
            now = utcnow()
            self.db.execute("UPDATE watches SET state=?,last_attempt=?,"
                            "last_success=CASE WHEN ? THEN ? ELSE last_success END,error=?,next_due=?,failures=0 WHERE id=?",
                            (json.dumps(batch.state, allow_nan=False), now, not batch.warnings, now,
                             '; '.join(batch.warnings)[:1000] or None, next_due, watch.id))
        return added

    def failure(self, watch_id, error, next_due):
        with self.db:
            self.db.execute("UPDATE watches SET last_attempt=?,error=?,next_due=?,failures=failures+1 WHERE id=?",
                            (utcnow(), str(error)[:1000], next_due, watch_id))

    def events(self, *, watch_id=None, after=0, limit=100, latest=False, pending=False):
        if not 1 <= limit <= 1000 or after < 0:
            raise ValueError("limit must be 1..1000 and after must be non-negative")
        query = "SELECT e.* FROM events e "
        if latest:
            query += "JOIN latest l ON e.seq=l.seq "
        if pending:
            query += "JOIN alerts a ON e.seq=a.seq AND a.acknowledged_at IS NULL "
        query += "WHERE e.seq>? "
        params = [after]
        if watch_id:
            query += "AND e.watch_id=? "
            params.append(watch_id)
        query += "ORDER BY e.seq LIMIT ?"
        return [{"seq": r['seq'], "watch_id": r['watch_id'], **json.loads(r['record'])}
                for r in self.db.execute(query, [*params, limit])]

    def acknowledge(self, sequences):
        with self.db:
            self.db.executemany("UPDATE alerts SET acknowledged_at=? WHERE seq=?",
                                [(utcnow(), int(s)) for s in sequences])

    def retrieval_events(self, *, as_of, watch_id=None, max_records=10000):
        """Latest eligible revision per event, including old revisions for past queries.

        Event normalizes stored dates to UTC ISO strings. A fixed microsecond cutoff
        preserves sub-second ordering without SQLite's rounded Julian-day conversion.
        Refuse an oversized snapshot rather than silently searching only its first page.
        """
        from fin_skills.rag.documents import timestamp
        if type(max_records) is not int or not 1 <= max_records <= 10000:
            raise ValueError('max_records must be 1..10000')
        cutoff = timestamp(as_of, 'as_of').isoformat(timespec='microseconds')
        conditions = ('observed_at <= ? AND (json_extract(record, \'$.published_at\') IS NULL '
                      'OR json_extract(record, \'$.published_at\') <= ?)')
        params = [cutoff, cutoff]
        if watch_id is not None:
            conditions += ' AND watch_id=?'
            params.append(watch_id)
        rows = self.db.execute(
            'SELECT seq,watch_id,record FROM events WHERE seq IN '
            '(SELECT max(seq) FROM events WHERE ' + conditions +
            ' GROUP BY watch_id,event_id) ORDER BY seq LIMIT ?', [*params, max_records + 1]).fetchall()
        if len(rows) > max_records:
            raise ValueError('eligible collection exceeds max_records; narrow watch_id or raise the limit')
        return [dict(json.loads(r['record']), seq=r['seq'], watch_id=r['watch_id']) for r in rows]

    def status(self):
        return [dict(r) for r in self.db.execute(
            "SELECT id,last_attempt,last_success,error,next_due,failures FROM watches ORDER BY id")]
