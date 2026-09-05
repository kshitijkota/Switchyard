"""Recovery loop — AGENT_BRIEF §7.

When a payment fails, decide which processor to re-attempt on, and whether to
re-attempt at all. Same routing brain (switchyard), second attempt.

Stopping rules:
  - never re-attempt if expected net reward ≤ 0,
  - at most MAX_ATTEMPTS total (the original attempt counts as #1),
  - never re-attempt a hard decline (U16 risk-reject / expired instrument).

Compliance: an attempt cap and a minimum inter-attempt delay for e-mandate-style
failures (modelled here as netbanking debits). The RBI 24-hour pre-debit
notification rule is NOT implemented — that gap is stated in the README.

Idempotency is enforced by the injected IdempotencyStore. Every attempt is
appended to the audit trail. The engine never imports ground truth; the injected
gateway is the only thing that produces outcomes.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from events import PROCESSORS, Context, Decision
from recovery.store import IdempotencyStore

MAX_ATTEMPTS = 3
HARD_DECLINE_CODES = frozenset({"Z9"})   # insufficient funds — retrying elsewhere cannot create funds
EMANDATE_MIN_DELAY = timedelta(hours=1)
POLICY_VERSION = "switchyard-recovery-v1"

_ROOT = os.path.dirname(os.path.dirname(__file__))
AUDIT_PATH = os.path.join(_ROOT, "artifacts", "audit.jsonl")


def is_emandate(context: Context) -> bool:
    """Stand-in for e-mandate/recurring debits: netbanking."""
    return context.method == "netbanking"


@dataclass
class RecoveryResult:
    txn_id: str
    recovered: bool
    recovered_reward_paise: int
    n_attempts: int
    stopped_reason: str
    retries: list = field(default_factory=list)


class ComplianceError(RuntimeError):
    pass


class RecoveryEngine:
    def __init__(self, method, gateway, store: IdempotencyStore,
                 audit_path: str = AUDIT_PATH, emandate_delay: timedelta = EMANDATE_MIN_DELAY):
        self.method = method
        self.gateway = gateway
        self.store = store
        self.audit_path = audit_path
        self.emandate_delay = emandate_delay
        os.makedirs(os.path.dirname(audit_path), exist_ok=True)

    def _best_alternative(self, context: Context, exclude: str):
        er = self.method.expected_reward_for(context)
        best_proc, best_val = None, -np.inf
        for p, proc in enumerate(PROCESSORS):
            if proc == exclude:
                continue
            if er[p] > best_val:
                best_val, best_proc = float(er[p]), proc
        return best_proc, best_val

    def _audit(self, context: Context, decision: Decision, outcome, expected_reward_paise: int):
        rec = {
            "context": context.to_dict(),
            "decision": decision.to_dict(),
            "expected_reward_paise": expected_reward_paise,
            "outcome": outcome.to_dict(),
            "policy_version": POLICY_VERSION,
        }
        with open(self.audit_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n")

    def can_attempt_now(self, context: Context, now: datetime) -> bool:
        """Compliance gate: e-mandate retries must respect the minimum delay."""
        if not is_emandate(context):
            return True
        last = self.store.last_attempt_ts(context.txn_id)
        return last is None or (now - last) >= self.emandate_delay

    def handle_failure(self, context: Context, first_processor: str,
                       first_failure_code: str | None, rng: np.random.Generator,
                       start_ts: datetime) -> RecoveryResult:
        # Idempotency: exactly one reservation wins; duplicates are no-ops.
        if not self.store.reserve(context.txn_id, now=start_ts):
            return RecoveryResult(context.txn_id, False, 0, 0, "duplicate")

        self.store.record_attempt(context.txn_id, first_processor, now=start_ts)  # original = #1
        last_proc, last_code, last_ts = first_processor, first_failure_code, start_ts
        recovered, recovered_reward, reason = False, 0, ""
        retries = []

        while True:
            n = self.store.get(context.txn_id)["n_attempts"]
            if n >= MAX_ATTEMPTS:
                reason = "max_attempts"; break
            if last_code in HARD_DECLINE_CODES:
                reason = "hard_decline"; break

            best_proc, best_er = self._best_alternative(context, exclude=last_proc)
            if best_er <= 0:
                reason = "nonpositive_expected_reward"; break

            attempt_ts = last_ts + self.emandate_delay if is_emandate(context) else last_ts
            if not self.can_attempt_now(context, attempt_ts):
                reason = "compliance_delay"; break

            outcome = self.gateway.attempt(context, best_proc, rng)
            decision = Decision(context.txn_id, best_proc, 1.0, POLICY_VERSION, int(round(best_er)))
            self.store.record_attempt(context.txn_id, best_proc, now=attempt_ts)
            self._audit(context, decision, outcome, int(round(best_er)))
            retries.append({"processor": best_proc, "expected_reward_paise": int(round(best_er)),
                            "success": outcome.success, "failure_code": outcome.failure_code})
            last_proc, last_ts = best_proc, attempt_ts
            if outcome.success:
                recovered, recovered_reward, reason = True, outcome.reward_paise, "recovered"
                break
            last_code = outcome.failure_code

        self.store.finish(context.txn_id, now=last_ts)
        n_final = self.store.get(context.txn_id)["n_attempts"]
        return RecoveryResult(context.txn_id, recovered, recovered_reward, n_final, reason, retries)
