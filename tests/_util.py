"""Shared test helpers. Tests are grader-side, so they MAY use ground truth."""

from __future__ import annotations

import numpy as np

from estimators.core import LoggedDataset
from events import PROCESSORS, LoggedEvent, Outcome
from policy.legacy import LegacyPolicy
from sim import economics as ec
from sim import ground_truth as gt
from sim.traffic import context_at, generate_traffic_arrays


def gen_events(n: int, seed: int, policy: str = "uniform") -> list[LoggedEvent]:
    ta = generate_traffic_arrays(n, seed=seed, prefix=f"{policy[:3]}{seed}_")
    s_dec, s_out = np.random.SeedSequence(seed + 1).spawn(2)
    rng_dec = np.random.default_rng(s_dec)
    rng_out = np.random.default_rng(s_out)
    legacy = LegacyPolicy()
    events = []
    for i in range(n):
        ctx = context_at(ta, i)
        if policy == "uniform":
            proc = PROCESSORS[int(rng_dec.integers(len(PROCESSORS)))]
            propensity = 1.0 / len(PROCESSORS)
            decision = _decision(ctx.txn_id, proc, propensity, "uniform-v1")
        else:
            decision = legacy.decide(ctx, rng_dec)
            proc = decision.processor
        success, code, _cause = gt.sample_outcome(ctx, proc, rng_out)
        outcome = Outcome(
            ctx.txn_id, success, code,
            ec.revenue_paise(ctx.amount_paise) if success else 0,
            ec.cost_paise(proc, ctx.amount_paise) if success else 0,
            ec.reward_paise(proc, ctx.amount_paise, success),
        )
        events.append(LoggedEvent(ctx, decision, outcome))
    return events


def _decision(txn_id, proc, propensity, version):
    from events import Decision
    return Decision(txn_id, proc, float(propensity), version, None)


def gen_dataset(n: int, seed: int, policy: str = "uniform") -> LoggedDataset:
    return LoggedDataset.from_events(gen_events(n, seed, policy))
