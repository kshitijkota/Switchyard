"""TASK B regression: the failure-code taxonomy is a cosmetic label and must NOT
perturb the economic (success/reward) stream.

Once the ambiguous-U30 override drew from the OUTCOME rng, which shifted the whole
success stream and silently moved the TASK A live-race result. This test locks the
invariant: toggling the ambiguity fraction (or any code-selection logic that keeps
the success/cause/code-index draws) leaves the sequence of successes identical.
"""

from __future__ import annotations

import numpy as np

import sim.ground_truth as gt
from sim.traffic import context_at, generate_traffic_arrays


def _success_sequence(prob: float) -> list[bool]:
    ta = generate_traffic_arrays(600, seed=7, prefix="inv_")
    rng = np.random.default_rng(123)
    out = []
    for i in range(len(ta)):
        ctx = context_at(ta, i)
        success, _code, _cause = gt.sample_outcome(ctx, "pa", rng)
        out.append(success)
    return out


def test_failure_code_taxonomy_does_not_perturb_economics(monkeypatch):
    base = _success_sequence(gt.AMBIGUOUS_EMIT_PROB)
    # Drive the ambiguity fraction to both extremes; the success stream must not move.
    monkeypatch.setattr(gt, "AMBIGUOUS_EMIT_PROB", 0.0)
    assert _success_sequence(0.0) == base
    monkeypatch.setattr(gt, "AMBIGUOUS_EMIT_PROB", 0.95)
    assert _success_sequence(0.95) == base


def test_ambiguous_override_changes_only_codes(monkeypatch):
    # With the override off, U30 never appears; with it on, some issuer/network
    # failures relabel to U30 — but the SET of failed txns is unchanged.
    def failures(prob):
        monkeypatch.setattr(gt, "AMBIGUOUS_EMIT_PROB", prob)
        ta = generate_traffic_arrays(600, seed=7, prefix="inv_")
        rng = np.random.default_rng(123)
        failed_ids, codes = [], []
        for i in range(len(ta)):
            ctx = context_at(ta, i)
            s, code, _c = gt.sample_outcome(ctx, "pa", rng)
            if not s:
                failed_ids.append(ctx.txn_id); codes.append(code)
        return failed_ids, codes

    ids0, codes0 = failures(0.0)
    ids1, codes1 = failures(0.9)
    assert ids0 == ids1                         # exactly the same txns fail
    assert "U30" not in codes0                  # override off -> no U30
    assert codes1.count("U30") > 0              # override on -> some U30
