"""Idempotency store — AGENT_BRIEF §7.

The same txn_id arriving twice must never produce two attempts. We use a SQLite
table with txn_id as PRIMARY KEY and a PENDING reservation row: the first arrival
INSERTs the reservation and wins; any concurrent duplicate hits the primary-key
constraint and is turned away. WAL mode + a busy timeout let concurrent writers
serialise cleanly (see DECISIONS.md for why WAL over an application mutex).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime

_SCHEMA = """
CREATE TABLE IF NOT EXISTS attempts (
    txn_id        TEXT PRIMARY KEY,
    status        TEXT NOT NULL,          -- PENDING | DONE
    n_attempts    INTEGER NOT NULL DEFAULT 0,
    last_processor TEXT,
    created_ts    TEXT NOT NULL,
    updated_ts    TEXT NOT NULL,
    last_attempt_ts TEXT
);
"""


class IdempotencyStore:
    def __init__(self, path: str, busy_timeout_ms: int = 5000):
        self.path = path
        self._conn = self._connect(busy_timeout_ms)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def _connect(self, busy_timeout_ms: int) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=busy_timeout_ms / 1000.0, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute(f"PRAGMA busy_timeout={busy_timeout_ms};")
        return conn

    def reserve(self, txn_id: str, now: datetime | None = None) -> bool:
        """Atomically reserve a txn for processing. Returns True exactly once per
        txn_id (the winner); False for every duplicate."""
        ts = (now or datetime.now()).isoformat()
        try:
            self._conn.execute(
                "INSERT INTO attempts (txn_id, status, n_attempts, created_ts, updated_ts) "
                "VALUES (?, 'PENDING', 0, ?, ?)",
                (txn_id, ts, ts),
            )
            return True
        except sqlite3.IntegrityError:
            return False  # already reserved — idempotent no-op

    def record_attempt(self, txn_id: str, processor: str, now: datetime | None = None) -> int:
        ts = (now or datetime.now()).isoformat()
        cur = self._conn.execute(
            "UPDATE attempts SET n_attempts = n_attempts + 1, last_processor = ?, "
            "last_attempt_ts = ?, updated_ts = ? WHERE txn_id = ? RETURNING n_attempts",
            (processor, ts, ts, txn_id),
        )
        row = cur.fetchone()
        return int(row[0]) if row else 0

    def finish(self, txn_id: str, now: datetime | None = None) -> None:
        ts = (now or datetime.now()).isoformat()
        self._conn.execute(
            "UPDATE attempts SET status = 'DONE', updated_ts = ? WHERE txn_id = ?", (ts, txn_id)
        )

    def get(self, txn_id: str) -> dict | None:
        cur = self._conn.execute(
            "SELECT txn_id, status, n_attempts, last_processor, last_attempt_ts "
            "FROM attempts WHERE txn_id = ?", (txn_id,))
        row = cur.fetchone()
        if row is None:
            return None
        return {"txn_id": row[0], "status": row[1], "n_attempts": row[2],
                "last_processor": row[3], "last_attempt_ts": row[4]}

    def last_attempt_ts(self, txn_id: str) -> datetime | None:
        row = self.get(txn_id)
        if row and row["last_attempt_ts"]:
            return datetime.fromisoformat(row["last_attempt_ts"])
        return None

    def close(self) -> None:
        self._conn.close()
