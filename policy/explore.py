"""ε-exploration policy — AGENT_BRIEF §6 (chowk) / §6.2.

A budgeted slice of traffic routed by ε-greedy on top of a DETERMINISTIC exploit
policy (the legacy mode). With probability 1−ε follow the exploit choice; with
probability ε route uniformly over the three processors. The recorded propensity
of the chosen processor is, at decision time:

    propensity(p) = (1−ε)·[p == exploit] + ε/3

so the exploit choice gets 0.98 and each other processor gets exactly ε/3 = 0.01
(the "minimum propensity 0.01" of the brief). That 0.01 floor is what restores
coverage in the cells the legacy policy starved. No ground truth is used here.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from events import PROCESSORS, Context, Decision

EPSILON = 0.03
POLICY_VERSION = "explore-v1"


class EpsilonGreedyPolicy:
    policy_version = POLICY_VERSION

    def __init__(self, exploit_proc_fn: Callable[[Context], str], epsilon: float = EPSILON):
        self.exploit_proc_fn = exploit_proc_fn
        self.epsilon = epsilon

    def propensity_of(self, chosen: str, exploit: str) -> float:
        return (1.0 - self.epsilon) * (1.0 if chosen == exploit else 0.0) + self.epsilon / len(PROCESSORS)

    def decide(self, context: Context, rng: np.random.Generator) -> tuple[Decision, bool, str]:
        """Return (decision, was_explore, exploit_proc). was_explore/exploit_proc
        are recorded to a separate aux file for the §6.2 exploration accounting;
        they never enter the event log the models read."""
        exploit = self.exploit_proc_fn(context)
        was_explore = bool(rng.random() < self.epsilon)
        chosen = PROCESSORS[int(rng.integers(len(PROCESSORS)))] if was_explore else exploit
        decision = Decision(
            txn_id=context.txn_id,
            processor=chosen,
            propensity=self.propensity_of(chosen, exploit),
            policy_version=POLICY_VERSION,
            expected_reward_paise=None,
        )
        return decision, was_explore, exploit
