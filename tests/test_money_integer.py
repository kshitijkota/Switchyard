"""§9: money is always integer paise; no float arithmetic on money."""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pytest

from events import Context, Decision, Outcome, PROCESSORS
from sim import economics as ec
from tests._util import gen_events


def test_economics_return_integers():
    for amt in [1, 100, 999, 120000, 500001, 20_000_000]:
        assert type(ec.revenue_paise(amt)) is int
        for p in PROCESSORS:
            assert type(ec.cost_paise(p, amt)) is int
            assert type(ec.reward_paise(p, amt, True)) is int
            assert type(ec.reward_if_success_paise(p, amt)) is int


def test_scalar_and_vector_economics_agree():
    amts = np.array([1, 100, 999, 120000, 500001, 1_000_000, 20_000_000], dtype=np.int64)
    for p in PROCESSORS:
        vec = ec.reward_if_success_paise_vec(p, amts)
        sca = np.array([ec.reward_if_success_paise(p, int(a)) for a in amts])
        assert np.array_equal(vec, sca)


def test_schema_rejects_float_money():
    with pytest.raises(TypeError):
        Context("t", datetime(2026, 1, 1), "upi", "hdfc", 120000.0, 11)
    with pytest.raises(TypeError):
        Outcome("t", True, None, 2400.0, 400, 2000)
    with pytest.raises(TypeError):
        Decision("t", "pa", 0.9, "v", 100.5)  # expected_reward must be int|None


def test_generated_logs_are_integer_paise():
    for ev in gen_events(500, seed=3, policy="legacy"):
        d = ev.to_dict()
        for field in ("amount_paise",):
            assert isinstance(d["context"][field], int)
        for field in ("revenue_paise", "cost_paise", "reward_paise"):
            v = d["outcome"][field]
            assert isinstance(v, int) and not isinstance(v, bool)
        # round-trips through JSON as an int, never a float
        assert isinstance(json.loads(json.dumps(d))["outcome"]["reward_paise"], int)
