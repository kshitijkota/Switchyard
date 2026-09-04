"""§7 / §9: the same txn_id arriving 10 times concurrently must never produce two
attempts. We test the store directly and end-to-end through the engine, each
thread using its own connection to the shared db (as separate workers would)."""

from __future__ import annotations

import threading
from datetime import datetime

import numpy as np

from events import Context
from recovery.loop import RecoveryEngine
from recovery.store import IdempotencyStore
from sim.gateway import SimGateway


def _ctx():
    return Context("dup-1", datetime(2026, 2, 1, 12), "upi", "hdfc", 120000, 12)


def test_store_reserve_wins_exactly_once(tmp_path):
    db = str(tmp_path / "idem.db")
    IdempotencyStore(db).close()  # create schema once
    results = []
    barrier = threading.Barrier(10)

    def worker():
        store = IdempotencyStore(db)
        barrier.wait()
        results.append(store.reserve("dup-1", now=datetime(2026, 2, 1, 12)))
        store.close()

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(results) == 1, f"expected exactly one winner, got {sum(results)}"


class _FakeBrain:
    name = "fake"

    def expected_reward_for(self, ctx):
        return np.array([1000.0, 1000.0, 1000.0])


def test_engine_processes_duplicate_once(tmp_path):
    db = str(tmp_path / "idem2.db")
    audit = str(tmp_path / "audit.jsonl")
    IdempotencyStore(db).close()
    outcomes = []
    barrier = threading.Barrier(10)

    def worker(i):
        store = IdempotencyStore(db)
        engine = RecoveryEngine(_FakeBrain(), SimGateway(), store, audit_path=audit)
        rng = np.random.default_rng(i)
        barrier.wait()
        res = engine.handle_failure(_ctx(), "pa", "U30", rng, start_ts=datetime(2026, 2, 1, 12))
        outcomes.append(res.stopped_reason)
        store.close()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # exactly one thread actually processed; the other nine were duplicates
    assert outcomes.count("duplicate") == 9
    non_dup = [r for r in outcomes if r != "duplicate"]
    assert len(non_dup) == 1

    store = IdempotencyStore(db)
    row = store.get("dup-1")
    store.close()
    assert row is not None and row["status"] == "DONE"
