"""Recovery evaluation — AGENT_BRIEF §7.

Money recovered = incremental successful re-attempts versus the legacy retry
policy, on ≥1,000 failures, with a bootstrap CI. We report the EXPECTED value of
the best re-attempt (ground-truth success prob, low noise) as the headline, and
separately run the live engine (sampled outcomes + SQLite + audit trail) to show
the stopping rules and idempotency working end to end.
"""

from __future__ import annotations

import json
import os

import numpy as np

from estimators.build import build_all, load_datasets
from events import read_jsonl
from policy.legacy import probs as legacy_probs
from recovery.loop import HARD_DECLINE_CODES, RecoveryEngine, AUDIT_PATH
from recovery.store import IdempotencyStore
from sim import economics as ec
from sim.gateway import SimGateway
from sim.ground_truth import success_prob
from events import PROCESSORS

_ROOT = os.path.dirname(os.path.dirname(__file__))
LOGS_PATH = os.path.join(_ROOT, "data", "logs.jsonl")
DB_PATH = os.path.join(_ROOT, "data", "recovery.db")


def load_failures(logs_path: str = LOGS_PATH, max_failures: int = 5000):
    out = []
    for ev in read_jsonl(logs_path):
        if not ev.outcome.success:
            out.append((ev.context, ev.decision.processor, ev.outcome.failure_code))
            if len(out) >= max_failures:
                break
    return out


def _true_expected(context, proc):
    return success_prob(context, proc) * ec.reward_if_success_paise(proc, context.amount_paise)


def chowk_best_retry_expected(context, failed_proc, failed_code, method) -> float:
    """Expected recovered reward (paise) of chowk's best single re-attempt."""
    if failed_code in HARD_DECLINE_CODES:
        return 0.0
    er = method.expected_reward_for(context)  # model estimate (no ground truth)
    best_proc, best_er = None, 0.0
    for p, proc in enumerate(PROCESSORS):
        if proc == failed_proc:
            continue
        if er[p] > best_er:   # must be > 0 to retry
            best_er, best_proc = er[p], proc
    if best_proc is None:
        return 0.0
    return _true_expected(context, best_proc)   # true expected value of that reroute


def legacy_retry_expected(context, failed_code) -> float:
    """Expected recovered reward (paise) of the legacy retry policy (reroute via
    the same skewed legacy distribution — it may even re-pick the failed one)."""
    if failed_code in HARD_DECLINE_CODES:
        return 0.0
    p = legacy_probs(context)
    return sum(p[proc] * _true_expected(context, proc) for proc in PROCESSORS)


def money_recovered(method, failures, seed: int = 0) -> dict:
    chowk = np.array([chowk_best_retry_expected(c, fp, fc, method) for c, fp, fc in failures])
    legacy = np.array([legacy_retry_expected(c, fc) for c, fp, fc in failures])
    incr = chowk - legacy
    n = len(failures)
    rng = np.random.default_rng(seed)
    boot = np.array([incr[rng.integers(0, n, n)].mean() for _ in range(1000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "n_failures": n,
        "chowk_expected_recovered_rupees": round(float(chowk.sum()) / 100, 2),
        "legacy_expected_recovered_rupees": round(float(legacy.sum()) / 100, 2),
        "incremental_per_failure_paise": round(float(incr.mean()), 2),
        "incremental_total_rupees": round(float(incr.sum()) / 100, 2),
        "incremental_ci_per_failure_paise": [round(float(lo), 2), round(float(hi), 2)],
        "distinguishable_from_zero": bool(lo > 0 or hi < 0),
    }


def run_live_engine(method, failures, seed: int = 0) -> dict:
    """Drive the real engine (sampled gateway + SQLite + audit) to show stopping
    rules and idempotency working. Rebuilds the db and audit from scratch."""
    from datetime import datetime, timedelta
    for pth in (DB_PATH, DB_PATH + "-wal", DB_PATH + "-shm"):
        if os.path.exists(pth):
            os.remove(pth)
    open(AUDIT_PATH, "w").close()  # truncate audit

    store = IdempotencyStore(DB_PATH)
    engine = RecoveryEngine(method, SimGateway(), store)
    rng = np.random.default_rng(seed)
    recovered = 0
    reasons: dict[str, int] = {}
    base_ts = datetime(2026, 2, 1, 12, 0)
    for i, (context, failed_proc, failed_code) in enumerate(failures):
        res = engine.handle_failure(context, failed_proc, failed_code, rng,
                                    start_ts=base_ts + timedelta(seconds=i))
        recovered += int(res.recovered)
        reasons[res.stopped_reason] = reasons.get(res.stopped_reason, 0) + 1
    # idempotency spot check: replay the first failure — must be a no-op duplicate
    c0, fp0, fc0 = failures[0]
    dup = engine.handle_failure(c0, fp0, fc0, rng, start_ts=base_ts)
    store.close()
    return {
        "n_processed": len(failures),
        "realized_recovered": recovered,
        "stopped_reasons": dict(sorted(reasons.items())),
        "duplicate_replay_created_attempt": dup.recovered or dup.n_attempts > 0,
        "audit_path": AUDIT_PATH,
    }


def evaluate() -> dict:
    legacy_ds, explore_ds = load_datasets()
    method = build_all(legacy_ds, explore_ds)["chowk"]
    failures = load_failures()
    return {
        "money_recovered": money_recovered(method, failures),
        "live_engine": run_live_engine(method, failures),
    }


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
