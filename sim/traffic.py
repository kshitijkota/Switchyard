"""Traffic generation — AGENT_BRIEF §4.1.

- amount_paise: log-normal, median ≈ ₹1,200, fat tail reaching ₹50,000+,
  hard cap at ₹200,000.
- method: 65% upi, 25% card, 10% netbanking.
- issuer: roughly uniform, hdfc slightly heavier.
- hour: diurnal, peaks 11–14 and 19–22.

The array path (`generate_traffic_arrays`) is the fast core used by eval
rollouts; `generate_traffic` wraps it into Context objects for log generation.
Both are fully determined by (n, seed).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterator

import numpy as np

from events import ISSUERS, METHODS, Context
from sim.ground_truth import SIM_START

# amount ~ lognormal(mu, sigma), median = exp(mu).
_MEDIAN_PAISE = 120_000                 # ₹1,200
_MU = math.log(_MEDIAN_PAISE)
_SIGMA = 1.25                           # fat enough to reach ₹50k+ in the tail
_AMOUNT_FLOOR = 100                     # ₹1
_AMOUNT_CAP = 20_000_000                # ₹200,000

_METHOD_P = np.array([0.65, 0.25, 0.10])          # upi / card / netbanking
_ISSUER_P = np.array([0.24, 0.19, 0.19, 0.19, 0.19])  # hdfc heavier

# Traffic profiles. "main" reproduces the frozen distribution exactly (same draws,
# same order → byte-identical logs). "heldout" is a deliberately DIFFERENT mix
# (more card, uniform issuers, fatter tail, fewer days) evaluated exactly once.
_PROFILES = {
    "main": {"method_p": _METHOD_P, "issuer_p": _ISSUER_P, "sigma": 1.25, "num_days": 28},
    "heldout": {"method_p": np.array([0.50, 0.35, 0.15]),
                "issuer_p": np.array([0.20, 0.20, 0.20, 0.20, 0.20]),
                "sigma": 1.45, "num_days": 21},
}

# Diurnal hour weights (index = hour). Twin peaks at 11–14 and 19–22.
_HOUR_W = np.array([
    0.20, 0.15, 0.10, 0.10, 0.15, 0.25, 0.50, 0.80, 1.20, 1.60, 1.90, 2.40,
    2.60, 2.50, 2.30, 1.80, 1.70, 1.80, 2.00, 2.50, 2.70, 2.60, 2.20, 1.00,
])
_HOUR_P = _HOUR_W / _HOUR_W.sum()

DEFAULT_NUM_DAYS = 28   # 4 weeks, so day_index % 7 sweeps every residue repeatedly


@dataclass(frozen=True)
class TrafficArrays:
    """Column-oriented traffic for the fast paths. dtype '<U…' string arrays for
    the categoricals, int64 for the numerics."""

    txn_ids: np.ndarray
    methods: np.ndarray
    issuers: np.ndarray
    amounts: np.ndarray        # int64 paise
    hours: np.ndarray          # int64 0-23
    day_indices: np.ndarray    # int64 0-based
    minutes: np.ndarray        # int64, for ts reconstruction only
    seconds: np.ndarray

    def __len__(self) -> int:
        return len(self.txn_ids)


def generate_traffic_arrays(
    n: int, seed: int, prefix: str = "txn_", num_days: int | None = None, profile: str = "main"
) -> TrafficArrays:
    cfg = _PROFILES[profile]
    if num_days is None:
        num_days = cfg["num_days"]
    rng = np.random.default_rng(seed)
    methods = np.array(METHODS)[rng.choice(len(METHODS), size=n, p=cfg["method_p"])]
    issuers = np.array(ISSUERS)[rng.choice(len(ISSUERS), size=n, p=cfg["issuer_p"])]

    raw = rng.lognormal(mean=_MU, sigma=cfg["sigma"], size=n)
    amounts = np.rint(raw).astype(np.int64)
    amounts = np.clip(amounts, _AMOUNT_FLOOR, _AMOUNT_CAP)

    hours = rng.choice(24, size=n, p=_HOUR_P).astype(np.int64)
    day_indices = rng.integers(0, num_days, size=n).astype(np.int64)
    minutes = rng.integers(0, 60, size=n).astype(np.int64)
    seconds = rng.integers(0, 60, size=n).astype(np.int64)

    txn_ids = np.array([f"{prefix}{i:08d}" for i in range(n)])
    return TrafficArrays(txn_ids, methods, issuers, amounts, hours, day_indices, minutes, seconds)


def ts_of(day_index: int, hour: int, minute: int, second: int) -> datetime:
    return datetime(SIM_START.year, SIM_START.month, SIM_START.day) + timedelta(
        days=int(day_index), hours=int(hour), minutes=int(minute), seconds=int(second)
    )


def context_at(ta: TrafficArrays, i: int) -> Context:
    return Context(
        txn_id=str(ta.txn_ids[i]),
        ts=ts_of(ta.day_indices[i], ta.hours[i], ta.minutes[i], ta.seconds[i]),
        method=str(ta.methods[i]),
        issuer=str(ta.issuers[i]),
        amount_paise=int(ta.amounts[i]),
        hour=int(ta.hours[i]),
    )


def generate_traffic(
    n: int, seed: int, prefix: str = "txn_", num_days: int = DEFAULT_NUM_DAYS
) -> Iterator[Context]:
    ta = generate_traffic_arrays(n, seed, prefix=prefix, num_days=num_days)
    for i in range(n):
        yield context_at(ta, i)
