"""Frozen event schema for Switchyard.

This is the contract every other module composes around. Field names are frozen
(see AGENT_BRIEF §3) and must not change.

ALL MONEY IS INTEGER PAISE. No floats for money, anywhere. The dataclasses below
enforce this at construction time so a float can never silently enter a log.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Iterable, Iterator

# --- Frozen domain vocabularies -------------------------------------------------
# Kept here so every module imports the same tuples and no string is re-typed.
METHODS = ("upi", "card", "netbanking")
ISSUERS = ("hdfc", "sbi", "icici", "axis", "kotak")
PROCESSORS = ("pa", "pb", "pc")

# NPCI / Razorpay-style failure codes (AGENT_BRIEF §3.1). The mapping from code
# to *true cause class* lives in sim.ground_truth ONLY; it must never appear in
# an event log the models consume.
FAILURE_CODES = (
    # NPCI UPI response codes (real; see DECISIONS.md)
    "U28",               # remitter/customer bank (PSP) down  -> issuer
    "Z9",                # insufficient funds                 -> customer
    "U69",               # collect request expired            -> customer
    "U30",               # "debit failed: bank down OR debit issue" -> AMBIGUOUS
    # Razorpay card/netbanking error codes (real)
    "BAD_REQUEST_ERROR",  # invalid request (integration)     -> merchant
    "GATEWAY_ERROR",     # transient gateway/bank error       -> network
    "SERVER_ERROR",      # transient internal error           -> network
)


def _require_paise(name: str, value: object, allow_none: bool = False) -> None:
    """Reject anything that is not a plain int (bool is not money either)."""
    if allow_none and value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{name} must be integer paise, got {type(value).__name__}: {value!r}"
        )


@dataclass(frozen=True)
class Context:
    txn_id: str
    ts: datetime
    method: str        # 'upi' | 'card' | 'netbanking'
    issuer: str        # 'hdfc' | 'sbi' | 'icici' | 'axis' | 'kotak'
    amount_paise: int
    hour: int          # 0-23

    def __post_init__(self) -> None:
        _require_paise("amount_paise", self.amount_paise)
        if self.amount_paise <= 0:
            raise ValueError(f"amount_paise must be positive, got {self.amount_paise}")
        if self.method not in METHODS:
            raise ValueError(f"unknown method {self.method!r}")
        if self.issuer not in ISSUERS:
            raise ValueError(f"unknown issuer {self.issuer!r}")
        if not isinstance(self.hour, int) or isinstance(self.hour, bool):
            raise TypeError("hour must be int")
        if not 0 <= self.hour <= 23:
            raise ValueError(f"hour out of range: {self.hour}")

    def to_dict(self) -> dict:
        return {
            "txn_id": self.txn_id,
            "ts": self.ts.isoformat(),
            "method": self.method,
            "issuer": self.issuer,
            "amount_paise": self.amount_paise,
            "hour": self.hour,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Context":
        return cls(
            txn_id=d["txn_id"],
            ts=datetime.fromisoformat(d["ts"]),
            method=d["method"],
            issuer=d["issuer"],
            amount_paise=int(d["amount_paise"]),
            hour=int(d["hour"]),
        )


@dataclass(frozen=True)
class Decision:
    txn_id: str
    processor: str     # 'pa' | 'pb' | 'pc'
    propensity: float  # probability THIS processor was chosen, recorded AT
                       # DECISION TIME. Never reconstructed afterwards.
    policy_version: str
    expected_reward_paise: int | None   # None for the legacy policy

    def __post_init__(self) -> None:
        if self.processor not in PROCESSORS:
            raise ValueError(f"unknown processor {self.processor!r}")
        if not isinstance(self.propensity, float):
            raise TypeError("propensity must be a float")
        if not 0.0 < self.propensity <= 1.0:
            raise ValueError(f"propensity must be in (0, 1], got {self.propensity}")
        _require_paise("expected_reward_paise", self.expected_reward_paise, allow_none=True)

    def to_dict(self) -> dict:
        return {
            "txn_id": self.txn_id,
            "processor": self.processor,
            "propensity": self.propensity,
            "policy_version": self.policy_version,
            "expected_reward_paise": self.expected_reward_paise,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Decision":
        er = d["expected_reward_paise"]
        return cls(
            txn_id=d["txn_id"],
            processor=d["processor"],
            propensity=float(d["propensity"]),
            policy_version=d["policy_version"],
            expected_reward_paise=None if er is None else int(er),
        )


@dataclass(frozen=True)
class Outcome:
    txn_id: str
    success: bool
    failure_code: str | None   # NPCI/Razorpay-style code, see events.FAILURE_CODES
    revenue_paise: int         # 0 if failed
    cost_paise: int            # 0 if failed
    reward_paise: int          # revenue - cost, 0 if failed

    def __post_init__(self) -> None:
        if not isinstance(self.success, bool):
            raise TypeError("success must be bool")
        _require_paise("revenue_paise", self.revenue_paise)
        _require_paise("cost_paise", self.cost_paise)
        _require_paise("reward_paise", self.reward_paise)
        if self.success:
            if self.failure_code is not None:
                raise ValueError("successful outcome must not carry a failure_code")
            if self.reward_paise != self.revenue_paise - self.cost_paise:
                raise ValueError("reward_paise must equal revenue_paise - cost_paise on success")
        else:
            if self.failure_code not in FAILURE_CODES:
                raise ValueError(f"failed outcome needs a known failure_code, got {self.failure_code!r}")
            if (self.revenue_paise, self.cost_paise, self.reward_paise) != (0, 0, 0):
                raise ValueError("failed outcome must have zero revenue/cost/reward")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Outcome":
        return cls(
            txn_id=d["txn_id"],
            success=bool(d["success"]),
            failure_code=d["failure_code"],
            revenue_paise=int(d["revenue_paise"]),
            cost_paise=int(d["cost_paise"]),
            reward_paise=int(d["reward_paise"]),
        )


@dataclass(frozen=True)
class LoggedEvent:
    """One joined row: context + decision + outcome. This is a JSONL line."""

    context: Context
    decision: Decision
    outcome: Outcome

    def __post_init__(self) -> None:
        ids = {self.context.txn_id, self.decision.txn_id, self.outcome.txn_id}
        if len(ids) != 1:
            raise ValueError(f"txn_id mismatch across context/decision/outcome: {ids}")

    @property
    def txn_id(self) -> str:
        return self.context.txn_id

    def to_dict(self) -> dict:
        return {
            "context": self.context.to_dict(),
            "decision": self.decision.to_dict(),
            "outcome": self.outcome.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LoggedEvent":
        return cls(
            context=Context.from_dict(d["context"]),
            decision=Decision.from_dict(d["decision"]),
            outcome=Outcome.from_dict(d["outcome"]),
        )


def write_jsonl(path: str, events: Iterable[LoggedEvent]) -> int:
    """Write logged events to JSONL. Returns the number of lines written.

    Uses sort_keys and no extra whitespace so the same data serialises
    byte-identically every run (determinism requirement, §9).
    """
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps(ev.to_dict(), sort_keys=True, separators=(",", ":")))
            fh.write("\n")
            n += 1
    return n


def read_jsonl(path: str) -> Iterator[LoggedEvent]:
    """Stream logged events from a JSONL file."""
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield LoggedEvent.from_dict(json.loads(line))
