"""Shared estimator core — AGENT_BRIEF §6.

All four methods estimate an expected-reward value V(cell, processor), then route
by per-cell argmax. They differ ONLY in the estimator (and, for switchyard, the data):

  direct : model plug-in    V = mean_i∈c  q̂(x_i, p)                 [GBM only]
  ips    : inverse propensity V = mean_i∈c  1(a_i=p)/μ_i · r_i
  snips  : self-normalised   V = Σ 1(a_i=p)/μ_i·r_i / Σ 1(a_i=p)/μ_i
  dr     : doubly robust      V = direct + mean_i∈c 1(a_i=p)/μ_i·(r_i − q̂(x_i,p))

Because DR's correction term is zero wherever an action was never logged in a
cell, DR *collapses to direct* in starved cells — which is exactly why switchyard's
ε-exploration (restoring coverage) is what lets the correction actually fire.

This module imports economics (a public fee schedule) but NEVER
sim.ground_truth — enforced by tests/test_ground_truth_isolation.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from events import METHODS, ISSUERS, PROCESSORS, LoggedEvent, read_jsonl
from sim import economics as ec
from estimators.segments import (
    CELL_INDEX,
    N_CELLS,
    PROC_INDEX,
    Cell,
    amount_bucket,
    cell_of,
    legacy_mode_proc,
    ALL_CELLS,
)

_METHOD_IDX = {m: i for i, m in enumerate(METHODS)}
_ISSUER_IDX = {s: i for i, s in enumerate(ISSUERS)}


@dataclass
class LoggedDataset:
    """Column arrays over N logged transactions. Feature arrays feed the Q-model;
    (proc_idx, propensity, reward) feed the reweighting estimators."""

    method: np.ndarray
    issuer: np.ndarray
    amount: np.ndarray        # int64 paise
    hour: np.ndarray          # int64
    proc_idx: np.ndarray      # 0/1/2 chosen processor
    propensity: np.ndarray    # float
    reward: np.ndarray        # float paise (0 if failed)
    success: np.ndarray       # bool
    cell_idx: np.ndarray      # 0..N_CELLS-1

    def __len__(self) -> int:
        return len(self.amount)

    @classmethod
    def from_events(cls, events: list[LoggedEvent]) -> "LoggedDataset":
        n = len(events)
        method = np.empty(n, dtype="<U10")
        issuer = np.empty(n, dtype="<U10")
        amount = np.empty(n, dtype=np.int64)
        hour = np.empty(n, dtype=np.int64)
        proc_idx = np.empty(n, dtype=np.int64)
        propensity = np.empty(n, dtype=np.float64)
        reward = np.empty(n, dtype=np.float64)
        success = np.empty(n, dtype=bool)
        cell_idx = np.empty(n, dtype=np.int64)
        for i, ev in enumerate(events):
            c, d, o = ev.context, ev.decision, ev.outcome
            method[i] = c.method
            issuer[i] = c.issuer
            amount[i] = c.amount_paise
            hour[i] = c.hour
            proc_idx[i] = PROC_INDEX[d.processor]
            propensity[i] = d.propensity
            reward[i] = float(o.reward_paise)
            success[i] = o.success
            cell_idx[i] = CELL_INDEX[cell_of(c.method, c.issuer, c.amount_paise)]
        return cls(method, issuer, amount, hour, proc_idx, propensity, reward, success, cell_idx)

    @classmethod
    def from_file(cls, path: str) -> "LoggedDataset":
        return cls.from_events(list(read_jsonl(path)))

    @staticmethod
    def concat(*parts: "LoggedDataset") -> "LoggedDataset":
        fields = ("method", "issuer", "amount", "hour", "proc_idx", "propensity",
                  "reward", "success", "cell_idx")
        return LoggedDataset(*[np.concatenate([getattr(p, f) for p in parts]) for f in fields])

    # --- Q-model feature matrix ------------------------------------------------
    def base_features(self) -> np.ndarray:
        """One-hot method (3) + issuer (5) + numeric [log1p(amount), hour]."""
        n = len(self)
        m = np.zeros((n, len(METHODS)))
        s = np.zeros((n, len(ISSUERS)))
        m[np.arange(n), [_METHOD_IDX[x] for x in self.method]] = 1.0
        s[np.arange(n), [_ISSUER_IDX[x] for x in self.issuer]] = 1.0
        num = np.column_stack([np.log1p(self.amount.astype(np.float64)), self.hour.astype(np.float64)])
        return np.hstack([m, s, num])


def _proc_onehot(n: int, p: int) -> np.ndarray:
    oh = np.zeros((n, len(PROCESSORS)))
    oh[:, p] = 1.0
    return oh


class QModel:
    """GBM classifier for P(success | context, processor), turned into expected
    net reward via the public fee schedule. Shared by direct, dr and switchyard."""

    def __init__(self, random_state: int = 0):
        self.clf = HistGradientBoostingClassifier(
            max_depth=4, max_iter=200, learning_rate=0.08,
            min_samples_leaf=200, random_state=random_state,
        )

    def fit(self, ds: LoggedDataset) -> "QModel":
        base = ds.base_features()
        X = np.hstack([base, _proc_onehot(len(ds), 0)])  # placeholder proc cols
        # overwrite proc one-hot with the actually-chosen processor per row
        proc_oh = np.zeros((len(ds), len(PROCESSORS)))
        proc_oh[np.arange(len(ds)), ds.proc_idx] = 1.0
        X[:, base.shape[1]:] = proc_oh
        self.clf.fit(X, ds.success.astype(int))
        self._n_features_base = base.shape[1]
        return self

    def success_proba(self, methods, issuers, amounts, hours) -> np.ndarray:
        """Predicted P(success) as (N, 3) for routing each context to each
        processor. Used by the naive success-rate router (§1's fee-blind
        strawman) — it maximises predicted success, ignoring fees."""
        n = len(amounts)
        m = np.zeros((n, len(METHODS))); s = np.zeros((n, len(ISSUERS)))
        m[np.arange(n), [_METHOD_IDX[x] for x in methods]] = 1.0
        s[np.arange(n), [_ISSUER_IDX[x] for x in issuers]] = 1.0
        num = np.column_stack([np.log1p(np.asarray(amounts, dtype=np.float64)),
                               np.asarray(hours, dtype=np.float64)])
        base = np.hstack([m, s, num])
        out = np.empty((n, len(PROCESSORS)))
        for p in range(len(PROCESSORS)):
            out[:, p] = self.clf.predict_proba(np.hstack([base, _proc_onehot(n, p)]))[:, 1]
        return out

    def expected_reward_for(self, context) -> np.ndarray:
        """Expected net reward (paise) for one context routed to each processor."""
        sp = self.success_proba([context.method], [context.issuer],
                                [context.amount_paise], [context.hour])[0]
        return np.array([sp[p] * ec.reward_if_success_paise(proc, context.amount_paise)
                         for p, proc in enumerate(PROCESSORS)], dtype=np.float64)

    def expected_reward(self, ds: LoggedDataset) -> np.ndarray:
        """Return (N, 3) expected net reward in paise: P̂(success|x_i,p) times the
        realised-on-success reward for routing x_i to each processor p."""
        base = ds.base_features()
        out = np.empty((len(ds), len(PROCESSORS)), dtype=np.float64)
        for p, proc in enumerate(PROCESSORS):
            X = np.hstack([base, _proc_onehot(len(ds), p)])
            phat = self.clf.predict_proba(X)[:, 1]
            r_succ = ec.reward_if_success_paise_vec(proc, ds.amount).astype(np.float64)
            out[:, p] = phat * r_succ
        return out


# --- Per-cell value estimation --------------------------------------------------

_ESTIMATORS = ("direct", "ips", "snips", "dr")


def estimate_cell_values(
    ds: LoggedDataset,
    estimator: str,
    qhat: np.ndarray | None = None,
    weight_clip: float | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (V, n_c, n_clipped).

    V         : (N_CELLS, 3) estimated expected reward per (cell, processor);
                np.nan where the estimate is undefined (no data and no model term).
    n_c       : (N_CELLS,) logged count per cell.
    n_clipped : number of importance weights clipped (0 if weight_clip is None).
    """
    n_c = np.bincount(ds.cell_idx, minlength=N_CELLS).astype(np.float64)
    safe_nc = np.where(n_c > 0, n_c, 1.0)
    V = np.full((N_CELLS, len(PROCESSORS)), np.nan, dtype=np.float64)

    w = 1.0 / ds.propensity
    n_clipped = 0
    if weight_clip is not None:
        clipped_mask = w > weight_clip
        n_clipped = int(clipped_mask.sum())
        w = np.minimum(w, weight_clip)

    for p in range(len(PROCESSORS)):
        chose_p = ds.proc_idx == p

        if estimator == "direct":
            model_term = np.bincount(ds.cell_idx, weights=qhat[:, p], minlength=N_CELLS) / safe_nc
            V[:, p] = np.where(n_c > 0, model_term, np.nan)

        elif estimator == "ips":
            cnt_p = np.bincount(ds.cell_idx[chose_p], minlength=N_CELLS)
            num = np.bincount(ds.cell_idx[chose_p], weights=(w * ds.reward)[chose_p], minlength=N_CELLS)
            # nan (unknown), NOT 0, where the action was never tried in the cell —
            # you cannot evaluate what you never tried.
            V[:, p] = np.where(cnt_p > 0, num / safe_nc, np.nan)

        elif estimator == "snips":
            num = np.bincount(ds.cell_idx[chose_p], weights=(w * ds.reward)[chose_p], minlength=N_CELLS)
            wsum = np.bincount(ds.cell_idx[chose_p], weights=w[chose_p], minlength=N_CELLS)
            V[:, p] = np.where(wsum > 0, num / np.where(wsum > 0, wsum, 1.0), np.nan)

        elif estimator == "dr":
            model_term = np.bincount(ds.cell_idx, weights=qhat[:, p], minlength=N_CELLS) / safe_nc
            resid = (ds.reward - qhat[:, p])
            corr = np.bincount(ds.cell_idx[chose_p], weights=(w * resid)[chose_p], minlength=N_CELLS) / safe_nc
            V[:, p] = np.where(n_c > 0, model_term + corr, np.nan)

        else:
            raise ValueError(f"unknown estimator {estimator!r}")

    return V, n_c, n_clipped


class Method:
    """A learned routing method: an estimator over a dataset, yielding a per-cell
    value table, a per-cell greedy policy, and a self-estimate of its value."""

    def __init__(self, name: str, estimator: str, weight_clip: float | None = None):
        assert estimator in _ESTIMATORS
        self.name = name
        self.estimator = estimator
        self.weight_clip = weight_clip
        self.value_table: np.ndarray | None = None    # (N_CELLS, 3)
        self.n_c: np.ndarray | None = None
        self.n_clipped: int = 0
        self.policy_idx: np.ndarray | None = None      # (N_CELLS,) proc idx
        self._qmodel: QModel | None = None

    def fit(self, ds: LoggedDataset, qmodel: QModel | None = None) -> "Method":
        qhat = None
        if self.estimator in ("direct", "dr"):
            if qmodel is None:
                qmodel = QModel().fit(ds)
            self._qmodel = qmodel
            qhat = qmodel.expected_reward(ds)
        self.value_table, self.n_c, self.n_clipped = estimate_cell_values(
            ds, self.estimator, qhat=qhat, weight_clip=self.weight_clip
        )
        self.policy_idx = self._derive_policy()
        return self

    def _derive_policy(self) -> np.ndarray:
        V = self.value_table
        policy = np.empty(N_CELLS, dtype=np.int64)
        for c in range(N_CELLS):
            row = V[c]
            if np.all(np.isnan(row)):
                policy[c] = PROC_INDEX[legacy_mode_proc(ALL_CELLS[c])]
            else:
                policy[c] = int(np.nanargmax(row))
        return policy

    def recommend_cell(self, cell: Cell) -> str:
        return PROCESSORS[self.policy_idx[CELL_INDEX[cell]]]

    def recommend(self, context) -> str:
        return self.recommend_cell(cell_of(context.method, context.issuer, context.amount_paise))

    def expected_reward_for(self, context) -> np.ndarray:
        """Per-processor expected net reward (paise) for one context — the routing
        brain used by the recovery loop. Uses the Q-model when present (direct/dr/
        switchyard); otherwise falls back to this method's per-cell value table."""
        if self._qmodel is not None:
            return self._qmodel.expected_reward_for(context)
        ci = CELL_INDEX[cell_of(context.method, context.issuer, context.amount_paise)]
        return self.value_table[ci]

    def policy_value_per_cell(self) -> np.ndarray:
        """V[c, π(c)] for each cell (paise/txn), nan where cell empty."""
        pv = np.full(N_CELLS, np.nan)
        for c in range(N_CELLS):
            if self.n_c[c] > 0:
                pv[c] = self.value_table[c, self.policy_idx[c]]
        return pv

    def estimated_value_per_1k(self) -> float:
        """Self-estimated value of this method's policy, ₹ per 1,000 txns, using
        the empirical cell frequencies of its own dataset."""
        return self.value_of_policy(self.policy_idx)

    def value_of_policy(self, policy_idx: np.ndarray) -> float:
        """This estimator's estimate of the value of an ARBITRARY per-cell policy
        (₹/1k txns), using its own data's cell frequencies. This is the off-policy
        *evaluation* the honesty benchmark turns on: point a starved-cell-visiting
        policy at each estimator and see which stays honest against the truth."""
        mask = self.n_c > 0
        freq = self.n_c[mask] / self.n_c[mask].sum()
        chosen_vals = self.value_table[np.arange(N_CELLS), policy_idx][mask]
        per_txn_paise = float(np.nansum(freq * chosen_vals))
        return per_txn_paise * 10.0

    def policy_coverage(self, policy_idx: np.ndarray) -> float:
        """Fraction of (frequency-weighted) cells where this estimator can even
        evaluate the given policy (non-nan). Low ⇒ it is guessing."""
        mask = self.n_c > 0
        freq = self.n_c[mask] / self.n_c[mask].sum()
        defined = ~np.isnan(self.value_table[np.arange(N_CELLS), policy_idx][mask])
        return float(freq[defined].sum())
