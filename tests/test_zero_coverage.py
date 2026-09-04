"""§9: zero-coverage segment handling must not crash or silently divide by zero."""

from __future__ import annotations

import numpy as np

from estimators.core import QModel, estimate_cell_values
from estimators.segments import N_CELLS
from events import PROCESSORS
from tests._util import gen_dataset


def test_no_crash_and_nan_on_zero_coverage():
    # legacy routing starves many (cell, processor) pairs.
    ds = gen_dataset(3000, seed=11, policy="legacy")
    qhat = QModel().fit(ds).expected_reward(ds)

    # None of these may raise (ZeroDivisionError / RuntimeWarning-as-error).
    v_ips, n_c, _ = estimate_cell_values(ds, "ips", weight_clip=None)
    v_snips, _, _ = estimate_cell_values(ds, "snips")
    v_dr, _, _ = estimate_cell_values(ds, "dr", qhat=qhat)
    v_direct, _, _ = estimate_cell_values(ds, "direct", qhat=qhat)

    # There is at least one covered cell with a starved processor: IPS is nan
    # there while DR stays finite (it falls back to the model term).
    starved = False
    for c in range(N_CELLS):
        if n_c[c] == 0:
            continue
        for p in range(len(PROCESSORS)):
            if np.isnan(v_ips[c, p]):
                starved = True
                assert np.isfinite(v_dr[c, p]), "DR must fall back to the model in starved cells"
    assert starved, "test needs at least one starved (cell, processor) to be meaningful"

    # Empty cells (no logs at all) are all-nan for every estimator.
    empty = np.where(n_c == 0)[0]
    if len(empty):
        assert np.all(np.isnan(v_direct[empty]))


def test_policy_derivation_survives_all_nan_cells():
    from estimators.core import Method
    ds = gen_dataset(2000, seed=12, policy="legacy")
    m = Method("ips", "ips").fit(ds)          # ips has the most nan cells
    assert m.policy_idx.shape[0] == N_CELLS
    assert set(np.unique(m.policy_idx)).issubset({0, 1, 2})   # always a valid processor
    # recommend never raises, even for a never-seen cell
    from estimators.segments import ALL_CELLS
    for cell in ALL_CELLS:
        assert m.recommend_cell(cell) in PROCESSORS
