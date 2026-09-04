"""Generate confounded logs — AGENT_BRIEF §5 / build step 3.

Joins traffic -> legacy decision (with propensity) -> latent outcome -> economics
and streams two files:

  data/logs.jsonl          one {context, decision, outcome} per line. The models
                           consume ONLY this. It carries the observable failure
                           CODE but never the latent cause class.
  data/ground_truth.jsonl  one {txn_id, success, true_cause, latent_regime} per
                           line. The grader/diagnosis side reads this; models
                           never do.

Determinism: all randomness derives from one master seed via SeedSequence, so
`same seed -> byte-identical files` (§9).
"""

from __future__ import annotations

import json
import os

import numpy as np

from events import LoggedEvent, Outcome, write_jsonl  # noqa: F401 (write_jsonl kept for API parity)
from policy.legacy import LegacyPolicy
from sim import economics as ec
from sim import ground_truth as gt
from sim.traffic import context_at, generate_traffic_arrays

_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(_ROOT, "data")
LOGS_PATH = os.path.join(DATA_DIR, "logs.jsonl")
GROUND_TRUTH_PATH = os.path.join(DATA_DIR, "ground_truth.jsonl")

N_TXNS = 200_000
MASTER_SEED = 20260904


def generate(
    n: int = N_TXNS,
    master_seed: int = MASTER_SEED,
    logs_path: str = LOGS_PATH,
    ground_truth_path: str = GROUND_TRUTH_PATH,
    prefix: str = "txn_",
) -> dict:
    """Write logs + ground truth. Returns summary stats (all script-produced)."""
    os.makedirs(os.path.dirname(logs_path), exist_ok=True)
    s_traffic, s_decide, s_outcome = np.random.SeedSequence(master_seed).spawn(3)
    ta = generate_traffic_arrays(n, seed=s_traffic, prefix=prefix)
    rng_decide = np.random.default_rng(s_decide)
    rng_outcome = np.random.default_rng(s_outcome)
    policy = LegacyPolicy()

    n_success = 0
    proc_counts = {"pa": 0, "pb": 0, "pc": 0}
    regime_counts: dict[str, int] = {}

    with open(logs_path, "w", encoding="utf-8") as lf, open(
        ground_truth_path, "w", encoding="utf-8"
    ) as gf:
        for i in range(n):
            ctx = context_at(ta, i)
            decision = policy.decide(ctx, rng_decide)
            success, code, cause = gt.sample_outcome(ctx, decision.processor, rng_outcome)
            reward = ec.reward_paise(decision.processor, ctx.amount_paise, success)
            outcome = Outcome(
                txn_id=ctx.txn_id,
                success=success,
                failure_code=code,
                revenue_paise=ec.revenue_paise(ctx.amount_paise) if success else 0,
                cost_paise=ec.cost_paise(decision.processor, ctx.amount_paise) if success else 0,
                reward_paise=reward,
            )
            ev = LoggedEvent(ctx, decision, outcome)
            lf.write(json.dumps(ev.to_dict(), sort_keys=True, separators=(",", ":")))
            lf.write("\n")

            reg = gt.regime(ctx, decision.processor)
            gf.write(
                json.dumps(
                    {"txn_id": ctx.txn_id, "success": success, "true_cause": cause,
                     "latent_regime": reg},
                    sort_keys=True, separators=(",", ":"),
                )
            )
            gf.write("\n")

            n_success += int(success)
            proc_counts[decision.processor] += 1
            regime_counts[reg] = regime_counts.get(reg, 0) + 1

    return {
        "n": n,
        "master_seed": master_seed,
        "success_rate": round(n_success / n, 4),
        "processor_share": {k: round(v / n, 4) for k, v in proc_counts.items()},
        "regime_share": {k: round(v / n, 4) for k, v in sorted(regime_counts.items())},
        "logs_path": logs_path,
        "ground_truth_path": ground_truth_path,
    }


if __name__ == "__main__":
    stats = generate()
    print(json.dumps(stats, indent=2))
