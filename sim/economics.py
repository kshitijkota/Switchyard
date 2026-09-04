"""Economics: revenue, per-processor cost, and reward — AGENT_BRIEF §4.3.

    revenue_paise  = round(0.02 * amount_paise)          # illustrative 2% take
    cost_pa(amount) = 400                                 # flat ₹4
    cost_pb(amount) = round(0.0025 * amount_paise)        # 0.25%
    cost_pc(amount) = 200 + round(0.0010 * amount_paise)  # ₹2 + 0.10%
    reward = success ? (revenue - cost) : 0

This module is deliberately PUBLIC: a router legitimately knows the aggregator's
own take rate and each processor's fee schedule. It contains NO latent ground
truth about success probabilities, so policy/estimator code may import it.

All returned money is integer paise. The rate constants are dimensionless; the
`round(rate * amount)` pattern is exactly as specified by the brief, and the
scalar and vectorised paths are tested to agree byte-for-byte.
"""

from __future__ import annotations

import numpy as np

from events import PROCESSORS

TAKE_RATE = 0.02          # aggregator revenue = 2% of ticket
PB_RATE = 0.0025          # pb charges 0.25%
PC_RATE = 0.0010          # pc charges ₹2 flat + 0.10%
PA_FLAT_PAISE = 400       # pa charges a flat ₹4
PC_FLAT_PAISE = 200       # pc's ₹2 flat component


def revenue_paise(amount_paise: int) -> int:
    return int(round(TAKE_RATE * amount_paise))


def cost_paise(processor: str, amount_paise: int) -> int:
    if processor == "pa":
        return PA_FLAT_PAISE
    if processor == "pb":
        return int(round(PB_RATE * amount_paise))
    if processor == "pc":
        return PC_FLAT_PAISE + int(round(PC_RATE * amount_paise))
    raise ValueError(f"unknown processor {processor!r}")


def reward_paise(processor: str, amount_paise: int, success: bool) -> int:
    """Realised reward: revenue - cost on success, else 0 (brief §4.3)."""
    if not success:
        return 0
    return revenue_paise(amount_paise) - cost_paise(processor, amount_paise)


def reward_if_success_paise(processor: str, amount_paise: int) -> int:
    """revenue - cost assuming success. May be NEGATIVE at tiny tickets where a
    flat fee exceeds the 2% take — this is the fee crossover the project turns on.
    """
    return revenue_paise(amount_paise) - cost_paise(processor, amount_paise)


def expected_reward_paise(processor: str, amount_paise: int, p_success: float) -> int:
    """Expected net reward given a success probability, as integer paise.

    Used both by the direct model (to turn P(success) into an objective) and by
    the segment value machinery. Rounded to int per the schema.
    """
    return int(round(p_success * reward_if_success_paise(processor, amount_paise)))


# --- Vectorised variants (numpy int64) for the hot eval/rollout paths ----------

def revenue_paise_vec(amount_paise: np.ndarray) -> np.ndarray:
    return np.rint(TAKE_RATE * amount_paise.astype(np.float64)).astype(np.int64)


def cost_paise_vec(processor: str, amount_paise: np.ndarray) -> np.ndarray:
    a = amount_paise.astype(np.float64)
    if processor == "pa":
        return np.full(a.shape, PA_FLAT_PAISE, dtype=np.int64)
    if processor == "pb":
        return np.rint(PB_RATE * a).astype(np.int64)
    if processor == "pc":
        return (PC_FLAT_PAISE + np.rint(PC_RATE * a)).astype(np.int64)
    raise ValueError(f"unknown processor {processor!r}")


def reward_if_success_paise_vec(processor: str, amount_paise: np.ndarray) -> np.ndarray:
    return revenue_paise_vec(amount_paise) - cost_paise_vec(processor, amount_paise)


def reward_paise_vec(processor: str, amount_paise: np.ndarray, success: np.ndarray) -> np.ndarray:
    r = reward_if_success_paise_vec(processor, amount_paise)
    return np.where(success, r, 0).astype(np.int64)
