"""Trained diagnosis classifier — TASK C2 (the third method, between rule and LLM).

A small multinomial logistic regression over the SHARE of each documented failure
code. It is trained on many labelled windows resampled from the closed-world log
(the true cause of each window is known from the ground-truth regime file, which
only the grader/trainer sees — never the diagnoser at inference on real cohorts).

Design (fixed before running, TASK C):
  - features: share of each code in `diagnose.schema.CODES` (the documented codes
    only) — a 7-vector. Unknown codes and free-text messages are NOT features, by
    construction: this is exactly the closed-world representation, so the model
    should do well closed-world and have nothing to stand on open-world.
  - classes: the four causes plus INSUFFICIENT_EVIDENCE (trained on blended /
    tiny / ambiguous-code windows), so it can learn to abstain from the data.
  - deterministic: fixed windows from the deterministic log + fixed random_state,
    so it reproduces with no cache and no key.
"""

from __future__ import annotations

import os
from collections import Counter

import numpy as np

from diagnose.schema import (
    CAUSE_CUSTOMER, CAUSE_ISSUER, CAUSE_MERCHANT, CAUSE_NETWORK, CODES, INSUFFICIENT,
)
from events import read_jsonl
from sim.ground_truth import AMBIGUOUS_CODE, REGIME_TRUE_CAUSE

_ROOT = os.path.dirname(os.path.dirname(__file__))
LOGS_PATH = os.path.join(_ROOT, "data", "logs.jsonl")
GT_PATH = os.path.join(_ROOT, "data", "ground_truth.jsonl")

WINDOW = 80
N_WINDOWS_PER_CLASS = 60
ABSTAIN_THRESH = 0.45   # max class prob below this -> abstain (chosen up front)


def _shares(counts: dict) -> list[float]:
    total = sum(counts.get(c, 0) for c in CODES)
    if total == 0:
        return [0.0] * len(CODES)
    return [counts.get(c, 0) / total for c in CODES]


def _failures_by_regime():
    import json
    regime = {}
    with open(GT_PATH) as fh:
        for line in fh:
            d = json.loads(line)
            if not d["success"]:
                regime[d["txn_id"]] = d["latent_regime"]
    by: dict[str, list[str]] = {}
    for ev in read_jsonl(LOGS_PATH):
        if ev.outcome.success:
            continue
        reg = regime.get(ev.txn_id, "baseline")
        by.setdefault(reg, []).append(ev.outcome.failure_code)
    return by


def _training_set(rng: np.random.Generator):
    by = _failures_by_regime()
    regime_cause = dict(REGIME_TRUE_CAUSE)
    regime_cause["baseline"] = CAUSE_CUSTOMER
    X, y = [], []
    # clear windows: sample WINDOW codes from a single regime, label = its cause
    for regime, cause in regime_cause.items():
        pool = by.get(regime, [])
        if len(pool) < WINDOW:
            continue
        for _ in range(N_WINDOWS_PER_CLASS):
            w = [pool[i] for i in rng.integers(0, len(pool), WINDOW)]
            X.append(_shares(Counter(w))); y.append(cause)
    # INSUFFICIENT windows: even blends of two regimes, and U30-dominated windows
    blends = [("issuer_degraded", "network_incident"), ("network_incident", "merchant_glitch"),
              ("issuer_degraded", "merchant_glitch")]
    for a, b in blends:
        pa, pb = by.get(a, []), by.get(b, [])
        if len(pa) < WINDOW or len(pb) < WINDOW:
            continue
        for _ in range(N_WINDOWS_PER_CLASS // 2):
            w = ([pa[i] for i in rng.integers(0, len(pa), WINDOW // 2)]
                 + [pb[i] for i in rng.integers(0, len(pb), WINDOW // 2)])
            X.append(_shares(Counter(w))); y.append(INSUFFICIENT)
    u30 = [c for codes in by.values() for c in codes if c == AMBIGUOUS_CODE]
    if len(u30) >= WINDOW:
        for _ in range(N_WINDOWS_PER_CLASS):
            w = [u30[i] for i in rng.integers(0, len(u30), WINDOW)]
            X.append(_shares(Counter(w))); y.append(INSUFFICIENT)
    return np.array(X), np.array(y)


class TrainedClassifier:
    """Multinomial logistic regression on documented-code shares."""

    name = "trained"

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.clf = None
        self.last_in = self.last_out = 0

    def fit(self) -> "TrainedClassifier":
        from sklearn.linear_model import LogisticRegression
        rng = np.random.default_rng(self.seed)
        X, y = _training_set(rng)
        self.clf = LogisticRegression(max_iter=2000, C=2.0, random_state=self.seed)
        self.clf.fit(X, y)
        return self

    def diagnose(self, inp) -> dict:
        if self.clf is None:
            self.fit()
        x = np.array([_shares(inp.cohort_counts)])
        proba = self.clf.predict_proba(x)[0]
        classes = list(self.clf.classes_)
        top = int(np.argmax(proba))
        cause = classes[top]
        conf = float(proba[top])
        if conf < ABSTAIN_THRESH:
            return {"cause": INSUFFICIENT, "confidence": round(conf, 2), "_abstain_kind": "model",
                    "evidence": [f"trained model max class prob {conf:.2f} < {ABSTAIN_THRESH}; abstains"]}
        return {"cause": cause, "confidence": round(conf, 2),
                "evidence": [f"trained LR on code shares -> {cause} (p={conf:.2f})"]}
