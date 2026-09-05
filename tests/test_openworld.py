"""TASK C: open-world diagnosis — the deterministic (no-LLM) guarantees.

The rule and the trained classifier win closed-world and fail open-world; the
open-world ground-truth labels are fixed in the module; the trained classifier is
reproducible. The LLM path is exercised by diagnose/threeway.py against the
committed cache, not here.
"""

from __future__ import annotations

from diagnose.cohorts import build_cohorts
from diagnose.node import Diagnoser, DiagnosisInput, StatisticalProvider
from diagnose.openworld import build_open_world_cohorts
from diagnose.schema import CODE_MEANING, INSUFFICIENT
from diagnose.threeway import _run, _score
from diagnose.trained import TrainedClassifier


def test_open_world_cohorts_are_fixed_and_large_enough():
    cohorts = build_open_world_cohorts()
    assert len(cohorts) >= 8
    for c in cohorts:
        total = sum(c.cohort_counts.values()) + sum(c.unknown_codes.values()) + len(c.messages)
        assert total >= 40, f"{c.label} below guardrail"
        assert c.kind in ("clear", "ambiguous")
        # withheld/free-text signals must NOT be documented codes (else not open-world)
        for code in c.unknown_codes:
            assert code not in CODE_MEANING


def test_rule_wins_closed_fails_open():
    closed, baseline = build_cohorts(0)
    openw = build_open_world_cohorts()
    rule = Diagnoser(StatisticalProvider(), cache_namespace="stat_t", cache_dir="/tmp/sw_stat_t")
    c = _score(_run(rule, closed, baseline, use_cache=False))
    o = _score(_run(rule, openw, baseline, use_cache=False))
    assert c["accuracy_on_clear"] >= 0.85            # strong closed-world
    assert o["accuracy_on_clear"] <= 0.25            # collapses open-world


def test_trained_wins_closed_fails_open():
    closed, baseline = build_cohorts(0)
    openw = build_open_world_cohorts()
    clf = Diagnoser(TrainedClassifier().fit(), cache_namespace="tr_t", cache_dir="/tmp/sw_tr_t")
    c = _score(_run(clf, closed, baseline, use_cache=False))
    o = _score(_run(clf, openw, baseline, use_cache=False))
    assert c["accuracy_on_clear"] >= 0.85            # learns the closed-world map well
    assert o["accuracy_on_clear"] <= 0.25            # collapses open-world


def test_trained_classifier_is_reproducible():
    a = TrainedClassifier(seed=0).fit()
    b = TrainedClassifier(seed=0).fit()
    inp = DiagnosisInput("x", "w", {"U28": 60, "Z9": 20}, {})
    assert a.diagnose(inp)["cause"] == b.diagnose(inp)["cause"]


def test_rule_abstains_on_free_text_openworld():
    # A pure free-text cohort has no documented codes -> a lookup rule cannot
    # interpret it and must abstain (not guess).
    inp = DiagnosisInput("ft", "w", {}, {}, messages=tuple(["Issuer bank down, retry"] * 55))
    out = StatisticalProvider().diagnose(inp)
    assert out["cause"] == INSUFFICIENT
