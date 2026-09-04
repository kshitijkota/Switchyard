"""Exploration-slice generation — AGENT_BRIEF §6 (switchyard) / §6.2 / build step 6.

Runs one more operating epoch under the ε-greedy policy (exploit = legacy mode,
ε = 0.03). Of these decisions ~3% are routed uniformly at random — that 3% is the
"explicitly budgeted slice of randomly-routed traffic" that restores coverage.

Writes:
  data/explore.jsonl      {context, decision, outcome} lines, same schema as the
                          legacy logs; switchyard consumes legacy + this.
  data/explore_aux.jsonl  {txn_id, was_explore, exploit_proc} — the extra facts
                          the §6.2 accounting needs. Kept OUT of the event log.

Randomness derives from one master seed; byte-identical on re-run.
"""

from __future__ import annotations

import json
import os

import numpy as np

from events import LoggedEvent, Outcome
from policy.explore import EpsilonGreedyPolicy
from policy.legacy import legacy_mode
from sim import economics as ec
from sim import ground_truth as gt
from sim.traffic import context_at, generate_traffic_arrays

_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(_ROOT, "data")
EXPLORE_PATH = os.path.join(DATA_DIR, "explore.jsonl")
EXPLORE_AUX_PATH = os.path.join(DATA_DIR, "explore_aux.jsonl")

EXPLORE_N = 200_000
MASTER_SEED = 815


def generate(
    n: int = EXPLORE_N,
    master_seed: int = MASTER_SEED,
    logs_path: str = EXPLORE_PATH,
    aux_path: str = EXPLORE_AUX_PATH,
    prefix: str = "exp_",
) -> dict:
    os.makedirs(os.path.dirname(logs_path), exist_ok=True)
    s_traffic, s_decide, s_outcome = np.random.SeedSequence(master_seed).spawn(3)
    ta = generate_traffic_arrays(n, seed=s_traffic, prefix=prefix)
    rng_decide = np.random.default_rng(s_decide)
    rng_outcome = np.random.default_rng(s_outcome)
    policy = EpsilonGreedyPolicy(exploit_proc_fn=legacy_mode)

    n_explore = 0
    n_success = 0
    with open(logs_path, "w", encoding="utf-8") as lf, open(aux_path, "w", encoding="utf-8") as af:
        for i in range(n):
            ctx = context_at(ta, i)
            decision, was_explore, exploit = policy.decide(ctx, rng_decide)
            success, code, _cause = gt.sample_outcome(ctx, decision.processor, rng_outcome)
            outcome = Outcome(
                txn_id=ctx.txn_id,
                success=success,
                failure_code=code,
                revenue_paise=ec.revenue_paise(ctx.amount_paise) if success else 0,
                cost_paise=ec.cost_paise(decision.processor, ctx.amount_paise) if success else 0,
                reward_paise=ec.reward_paise(decision.processor, ctx.amount_paise, success),
            )
            ev = LoggedEvent(ctx, decision, outcome)
            lf.write(json.dumps(ev.to_dict(), sort_keys=True, separators=(",", ":")) + "\n")
            af.write(json.dumps(
                {"txn_id": ctx.txn_id, "was_explore": was_explore, "exploit_proc": exploit},
                sort_keys=True, separators=(",", ":")) + "\n")
            n_explore += int(was_explore)
            n_success += int(success)

    return {
        "n": n, "master_seed": master_seed,
        "explore_rate": round(n_explore / n, 4),
        "success_rate": round(n_success / n, 4),
        "logs_path": logs_path, "aux_path": aux_path,
    }


if __name__ == "__main__":
    print(json.dumps(generate(), indent=2))
