"""Legacy routing policy — AGENT_BRIEF §5.

Generates the confounded logs. Stochastic but heavily skewed, so some
(segment, processor) pairs get near-zero coverage:

    if amount_paise > 500_000:   pb 0.95, pa 0.03, pc 0.02
    elif method == 'upi':        pa 0.90, pc 0.07, pb 0.03
    else:                        pc 0.85, pa 0.10, pb 0.05

This matches §5 exactly. Note that >₹5,000 traffic goes almost entirely to pb,
starving pa and pc on large tickets — which is precisely where the latent truth
hides pa's large-ticket weakness (sim/ground_truth.py), so a model trained on
these logs never learns it.

The propensity of the chosen processor is recorded AT DECISION TIME. It is never
reconstructed later — that is ABSOLUTE RULE 2.
"""

from __future__ import annotations

import numpy as np

from events import Decision, Context

POLICY_VERSION = "legacy-v1"

_HIGH_AMOUNT_PAISE = 500_000  # ₹5,000


def probs(context: Context) -> dict[str, float]:
    """Selection probability per processor for this context."""
    if context.amount_paise > _HIGH_AMOUNT_PAISE:
        return {"pa": 0.03, "pb": 0.95, "pc": 0.02}
    if context.method == "upi":
        return {"pa": 0.90, "pb": 0.03, "pc": 0.07}
    return {"pa": 0.10, "pb": 0.05, "pc": 0.85}


def legacy_mode(context: Context) -> str:
    """The deterministic argmax of the legacy selection probabilities. Used as
    the (deployment) exploit base for the ε-exploration slice."""
    p = probs(context)
    return max(p, key=p.__getitem__)


class LegacyPolicy:
    policy_version = POLICY_VERSION

    def decide(self, context: Context, rng: np.random.Generator) -> Decision:
        p = probs(context)
        procs = list(p)
        weights = np.array([p[x] for x in procs], dtype=np.float64)
        chosen = procs[int(rng.choice(len(procs), p=weights))]
        return Decision(
            txn_id=context.txn_id,
            processor=chosen,
            propensity=float(p[chosen]),  # recorded at decision time
            policy_version=POLICY_VERSION,
            expected_reward_paise=None,   # the legacy policy computes no estimate
        )
