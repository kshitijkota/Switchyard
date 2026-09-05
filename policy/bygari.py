"""bygari_baseline — a faithful reimplementation of the published Razorpay router
(Bygari et al., "An AI-powered Smart Routing Solution for Payment Systems," IEEE
Big Data 2021). Added as a baseline alongside `direct`; `direct` is kept.

Two modules, as in the paper:

STATIC
  - eligibility table (all three processors are contractually eligible for every
    txn in this sim — no artificial restriction; documented in DECISIONS.md);
  - a logistic-regression downtime circuit breaker over each processor's recent
    error velocity — a processor it flags is removed from consideration.

DYNAMIC
  - a random forest predicting P(success) per processor from payment attributes
    (method, issuer, amount, hour) plus per-processor rolling success rates;
  - route to the eligible, non-broken processor with the highest predicted
    SUCCESS probability (NOT expected net revenue — as the paper describes);
  - an adaptive time-decay (EWMA) feedback loop updates the rolling success rate
    and error velocity after each outcome, recent events weighted more.

This module never imports sim.ground_truth (enforced by the isolation test).
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from estimators.core import LoggedDataset
from estimators.segments import amount_bucket
from events import ISSUERS, METHODS, PROCESSORS

_M_IDX = {m: i for i, m in enumerate(METHODS)}
_S_IDX = {s: i for i, s in enumerate(ISSUERS)}
_P_IDX = {p: i for i, p in enumerate(PROCESSORS)}

# Adaptive feedback + circuit-breaker constants (documented in DECISIONS.md).
SUCCESS_DECAY = 0.99          # EWMA weight on history for rolling success rate
ERRVEL_DECAY = 0.98          # EWMA weight on history for error velocity
BREAKER_FAIL_PROB = 0.60     # LR-predicted failure prob above which a processor is broken


def _row(method, issuer, amount, hour, proc, roll):
    """One RF feature row for (context, processor) given current rolling success."""
    m = np.zeros(len(METHODS)); m[_M_IDX[method]] = 1.0
    s = np.zeros(len(ISSUERS)); s[_S_IDX[issuer]] = 1.0
    p = np.zeros(len(PROCESSORS)); p[_P_IDX[proc]] = 1.0
    return np.concatenate([m, s, [np.log1p(amount), float(hour)], p,
                           [roll[0], roll[1], roll[2]]])


class BygariRouter:
    name = "bygari_baseline"

    def __init__(self, success_decay: float = SUCCESS_DECAY, errvel_decay: float = ERRVEL_DECAY,
                 breaker_fail_prob: float = BREAKER_FAIL_PROB, random_state: int = 0):
        self.success_decay = success_decay
        self.errvel_decay = errvel_decay
        self.breaker_fail_prob = breaker_fail_prob
        self.rf = RandomForestClassifier(n_estimators=60, max_depth=14, min_samples_leaf=50,
                                         random_state=random_state, n_jobs=-1)
        self.lr = LogisticRegression(max_iter=1000)
        # online state (per processor)
        self.roll = np.full(len(PROCESSORS), 0.87)     # rolling success rate
        self.errvel = np.full(len(PROCESSORS), 0.13)   # rolling error velocity
        self.n_broken = 0

    # --- training on the legacy log --------------------------------------------
    def fit(self, ds: LoggedDataset) -> "BygariRouter":
        n = len(ds)
        # causal rolling success + error velocity in arrival (index) order.
        roll = np.full(len(PROCESSORS), 0.87)
        errvel = np.full(len(PROCESSORS), 0.13)
        X = np.empty((n, len(METHODS) + len(ISSUERS) + 2 + len(PROCESSORS) + 3))
        y = ds.success.astype(int)
        lr_X = np.empty((n, len(PROCESSORS) + 1)); lr_y = (~ds.success).astype(int)
        for i in range(n):
            p = int(ds.proc_idx[i])
            X[i] = _row(str(ds.method[i]), str(ds.issuer[i]), int(ds.amount[i]), int(ds.hour[i]),
                        PROCESSORS[p], roll)
            ph = np.zeros(len(PROCESSORS)); ph[p] = 1.0
            lr_X[i] = np.concatenate([ph, [errvel[p]]])
            succ = bool(ds.success[i])
            roll[p] = self.success_decay * roll[p] + (1 - self.success_decay) * succ
            errvel[p] = self.errvel_decay * errvel[p] + (1 - self.errvel_decay) * (0.0 if succ else 1.0)
        self.rf.fit(X, y)
        self.lr.fit(lr_X, lr_y)
        self.roll = roll.copy()          # carry the legacy end-state into online use
        self.errvel = errvel.copy()
        return self

    # --- online decision + feedback --------------------------------------------
    def _eligible(self, context) -> list[str]:
        return list(PROCESSORS)   # all contractually eligible in this sim

    def _broken(self) -> set[str]:
        feats = np.zeros((len(PROCESSORS), len(PROCESSORS) + 1))
        for p in range(len(PROCESSORS)):
            feats[p, p] = 1.0
            feats[p, -1] = self.errvel[p]
        probs = self.lr.predict_proba(feats)[:, 1]
        return {PROCESSORS[p] for p in range(len(PROCESSORS)) if probs[p] > self.breaker_fail_prob}

    def route(self, context) -> str:
        candidates = [p for p in self._eligible(context) if p not in self._broken()]
        if not candidates:
            candidates = list(PROCESSORS)   # never leave a txn unroutable
        rows = np.array([_row(context.method, context.issuer, context.amount_paise,
                              context.hour, p, self.roll) for p in candidates])
        succ = self.rf.predict_proba(rows)[:, 1]
        return candidates[int(np.argmax(succ))]     # highest predicted SUCCESS

    def predicted_success(self, context, proc) -> float:
        row = _row(context.method, context.issuer, context.amount_paise, context.hour,
                   proc, self.roll).reshape(1, -1)
        return float(self.rf.predict_proba(row)[0, 1])

    def update(self, proc: str, success: bool) -> None:
        p = _P_IDX[proc]
        self.roll[p] = self.success_decay * self.roll[p] + (1 - self.success_decay) * (1.0 if success else 0.0)
        self.errvel[p] = self.errvel_decay * self.errvel[p] + (1 - self.errvel_decay) * (0.0 if success else 1.0)

    # --- vectorised mini-batch routing (tractability; rolling frozen per chunk) --
    def route_chunk(self, methods, issuers, amounts, hours) -> list[str]:
        """Route a chunk of txns with the rolling state frozen (updated between
        chunks). One vectorised RF predict instead of one call per txn."""
        broken = self._broken()
        cand = [p for p in PROCESSORS if p not in broken] or list(PROCESSORS)
        k = len(cand)
        n = len(methods)
        Moh = np.eye(len(METHODS))[[_M_IDX[m] for m in methods]]
        Soh = np.eye(len(ISSUERS))[[_S_IDX[s] for s in issuers]]
        num = np.column_stack([np.log1p(amounts.astype(float)), hours.astype(float)])
        base = np.repeat(np.hstack([Moh, Soh, num]), k, axis=0)          # (n*k) × 10
        Poh = np.tile(np.eye(len(PROCESSORS))[[_P_IDX[p] for p in cand]], (n, 1))  # (n*k) × 3
        roll = np.tile(self.roll, (n * k, 1))                            # (n*k) × 3
        rows = np.hstack([base, Poh, roll])
        succ = self.rf.predict_proba(rows)[:, 1].reshape(n, k)
        return [cand[j] for j in succ.argmax(axis=1)]

    def update_chunk(self, procs: list[str], successes) -> None:
        for proc, s in zip(procs, successes):
            self.update(proc, bool(s))

    def predicted_success_vec(self, methods, issuers, amounts, hours, proc_idx) -> np.ndarray:
        """Vectorised P(success) for each (context, given processor) at the current
        rolling state — for the batch off-policy eval table (TASK A3)."""
        n = len(methods)
        Moh = np.eye(len(METHODS))[[_M_IDX[m] for m in methods]]
        Soh = np.eye(len(ISSUERS))[[_S_IDX[s] for s in issuers]]
        num = np.column_stack([np.log1p(np.asarray(amounts, float)), np.asarray(hours, float)])
        Poh = np.eye(len(PROCESSORS))[np.asarray(proc_idx)]
        roll = np.tile(self.roll, (n, 1))
        return self.rf.predict_proba(np.hstack([Moh, Soh, num, Poh, roll]))[:, 1]
