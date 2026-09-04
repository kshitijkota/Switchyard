"""Cohort construction — the GRADER side of §8 (may read ground truth).

Builds labelled cohorts of failed transactions from the logs + the ground-truth
regime file. Clear cohorts have a known dominant cause; ambiguous cohorts are
deliberately un-diagnosable (too few failures, or an even blend of two regimes)
and their correct answer is INSUFFICIENT_EVIDENCE — that is what makes abstention
scoreable and rewarded. The diagnoser never sees any of this labelling.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass

from diagnose.schema import CAUSE_CUSTOMER, INSUFFICIENT
from events import read_jsonl
from sim.ground_truth import REGIME_TRUE_CAUSE

_ROOT = os.path.dirname(os.path.dirname(__file__))
LOGS_PATH = os.path.join(_ROOT, "data", "logs.jsonl")
GT_PATH = os.path.join(_ROOT, "data", "ground_truth.jsonl")

CLEAR_SIZE = 80
SMALL_SIZE = 18       # deliberately too few to conclude
N_CLEAR_PER_CAUSE = 4


@dataclass
class Cohort:
    label: str
    window: str
    counts: dict          # code -> count
    true_cause: str
    kind: str             # "clear" | "ambiguous"


def _load_failures_by_regime():
    import json
    regime = {}
    with open(GT_PATH) as fh:
        for line in fh:
            d = json.loads(line)
            if not d["success"]:
                regime[d["txn_id"]] = d["latent_regime"]
    by_regime: dict[str, list[str]] = {}
    all_codes: list[str] = []
    for ev in read_jsonl(LOGS_PATH):
        if ev.outcome.success:
            continue
        reg = regime.get(ev.txn_id, "baseline")
        by_regime.setdefault(reg, []).append(ev.outcome.failure_code)
        all_codes.append(ev.outcome.failure_code)
    return by_regime, Counter(all_codes)


def build_cohorts(seed: int = 0):
    import random
    rng = random.Random(seed)
    by_regime, baseline_counts = _load_failures_by_regime()
    for codes in by_regime.values():
        rng.shuffle(codes)

    cohorts: list[Cohort] = []

    # Clear cohorts: one cause per regime (baseline regime -> customer-side).
    regime_to_cause = dict(REGIME_TRUE_CAUSE)
    regime_to_cause["baseline"] = CAUSE_CUSTOMER
    for regime, cause in regime_to_cause.items():
        codes = by_regime.get(regime, [])
        for k in range(N_CLEAR_PER_CAUSE):
            chunk = codes[k * CLEAR_SIZE:(k + 1) * CLEAR_SIZE]
            if len(chunk) < CLEAR_SIZE:
                break
            cohorts.append(Cohort(f"{regime} cohort #{k+1}", "a 4h window",
                                  dict(Counter(chunk)), cause, "clear"))

    # Ambiguous #1: too few failures to conclude (drawn from real regimes).
    for regime in ("issuer_degraded", "network_incident", "merchant_glitch", "baseline"):
        codes = by_regime.get(regime, [])
        tail = codes[-SMALL_SIZE:]
        if len(tail) >= SMALL_SIZE:
            cohorts.append(Cohort(f"{regime} tiny sample", "a 20m window",
                                  dict(Counter(tail)), INSUFFICIENT, "ambiguous"))

    # Ambiguous #2: even blends of two anomalous regimes (genuinely two causes).
    blends = [("network_incident", "merchant_glitch"), ("issuer_degraded", "network_incident")]
    for a, b in blends:
        ca, cb = by_regime.get(a, [])[:40], by_regime.get(b, [])[:40]
        if len(ca) >= 40 and len(cb) >= 40:
            cohorts.append(Cohort(f"{a}+{b} even blend", "a 4h window",
                                  dict(Counter(ca + cb)), INSUFFICIENT, "ambiguous"))

    return cohorts, dict(baseline_counts)
