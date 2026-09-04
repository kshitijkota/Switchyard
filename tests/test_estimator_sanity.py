"""§9: on UNIFORMLY RANDOM logs, direct/ips/snips/dr must all agree with the true
per-cell value within tolerance. If they don't, an estimator is broken."""

from __future__ import annotations

import numpy as np

from estimators.core import QModel, estimate_cell_values
from estimators.segments import CELL_INDEX, cell_of
from events import PROCESSORS
from sim import economics as ec
from sim.ground_truth import success_prob_batch
from sim.traffic import generate_traffic_arrays
from tests._util import gen_dataset

N = 40_000
SEED = 21


def _true_cell_values(ta):
    """True expected reward per (cell, proc), averaged over each cell's traffic."""
    n = len(ta)
    cell_idx = np.array([CELL_INDEX[cell_of(str(ta.methods[i]), str(ta.issuers[i]), int(ta.amounts[i]))]
                         for i in range(n)])
    from estimators.segments import N_CELLS
    true = np.full((N_CELLS, 3), np.nan)
    counts = np.bincount(cell_idx, minlength=N_CELLS)
    for p, proc in enumerate(PROCESSORS):
        sp = success_prob_batch(ta.methods, ta.issuers, ta.amounts, ta.hours, ta.day_indices, proc)
        er = sp * ec.reward_if_success_paise_vec(proc, ta.amounts).astype(float)
        summed = np.bincount(cell_idx, weights=er, minlength=N_CELLS)
        true[:, p] = np.where(counts > 0, summed / np.where(counts > 0, counts, 1), np.nan)
    return true, counts


def test_uniform_logs_all_estimators_agree_with_truth():
    ds = gen_dataset(N, seed=SEED, policy="uniform")
    ta = generate_traffic_arrays(N, seed=SEED, prefix=f"uni{SEED}_")   # same traffic as gen_dataset
    true, counts = _true_cell_values(ta)
    qhat = QModel().fit(ds).expected_reward(ds)

    ests = {
        "direct": estimate_cell_values(ds, "direct", qhat=qhat)[0],
        "ips": estimate_cell_values(ds, "ips")[0],
        "snips": estimate_cell_values(ds, "snips")[0],
        "dr": estimate_cell_values(ds, "dr", qhat=qhat)[0],
    }

    # Only judge well-covered cells (≥300 logs ⇒ ~100 per processor under uniform),
    # and only where the true reward is large enough for a relative error to mean
    # something (avoid dividing by ~0 near a crossover).
    well = counts >= 300
    for name, V in ests.items():
        rel_errs = []
        for c in np.where(well)[0]:
            for p in range(3):
                t = true[c, p]
                if np.isnan(t) or abs(t) < 500 or np.isnan(V[c, p]):
                    continue
                rel_errs.append(abs(V[c, p] - t) / abs(t))
        rel_errs = np.array(rel_errs)
        assert len(rel_errs) > 30, f"{name}: too few comparable cells ({len(rel_errs)})"
        assert np.median(rel_errs) < 0.12, f"{name}: median rel error {np.median(rel_errs):.3f}"
        assert np.mean(rel_errs) < 0.20, f"{name}: mean rel error {np.mean(rel_errs):.3f}"
