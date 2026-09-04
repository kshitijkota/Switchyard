"""Simulator gateway — the 'real world' that executes a routing attempt.

The recovery engine decides WHO to route to; the gateway produces the OUTCOME.
Only the gateway (simulator side) touches ground truth, so the engine stays a
pure policy. Deterministic given the rng.
"""

from __future__ import annotations

import numpy as np

from events import Outcome
from sim import economics as ec
from sim import ground_truth as gt


class SimGateway:
    def attempt(self, context, processor: str, rng: np.random.Generator) -> Outcome:
        success, code, _cause = gt.sample_outcome(context, processor, rng)
        return Outcome(
            txn_id=context.txn_id,
            success=success,
            failure_code=code,
            revenue_paise=ec.revenue_paise(context.amount_paise) if success else 0,
            cost_paise=ec.cost_paise(processor, context.amount_paise) if success else 0,
            reward_paise=ec.reward_paise(processor, context.amount_paise, success),
        )
