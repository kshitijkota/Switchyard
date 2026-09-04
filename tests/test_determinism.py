"""§9: same seed produces byte-identical artifacts and identical learned policies."""

from __future__ import annotations

import hashlib

import numpy as np

from estimators.build import build_all
from sim.generate_logs import generate
from tests._util import gen_dataset


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def test_log_generation_byte_identical(tmp_path):
    a_logs, a_gt = str(tmp_path / "a.jsonl"), str(tmp_path / "a_gt.jsonl")
    b_logs, b_gt = str(tmp_path / "b.jsonl"), str(tmp_path / "b_gt.jsonl")
    generate(n=6000, logs_path=a_logs, ground_truth_path=a_gt, prefix="d_")
    generate(n=6000, logs_path=b_logs, ground_truth_path=b_gt, prefix="d_")
    assert _sha(a_logs) == _sha(b_logs)
    assert _sha(a_gt) == _sha(b_gt)


def test_learned_policies_are_deterministic():
    legacy = gen_dataset(6000, seed=31, policy="legacy")
    explore = gen_dataset(6000, seed=32, policy="uniform")
    m1 = build_all(legacy, explore)
    legacy2 = gen_dataset(6000, seed=31, policy="legacy")
    explore2 = gen_dataset(6000, seed=32, policy="uniform")
    m2 = build_all(legacy2, explore2)
    for name in m1:
        assert np.array_equal(m1[name].policy_idx, m2[name].policy_idx), f"{name} policy differs"
        assert m1[name].n_clipped == m2[name].n_clipped
