"""Bootstrap confidence intervals — AGENT_BRIEF §6.1.

≥1,000 resamples, common resample indices across methods so improvements are
paired. This module is pure numpy and sees no ground truth.
"""

from __future__ import annotations

import numpy as np

N_RESAMPLES = 1000


def paired_bootstrap(
    per_txn: dict[str, np.ndarray],
    baseline: np.ndarray,
    n_resamples: int = N_RESAMPLES,
    seed: int = 0,
    scale: float = 10.0,
) -> dict[str, dict]:
    """Bootstrap value and improvement-over-baseline CIs.

    per_txn : {name -> per-transaction reward array (paise), all same length N}
    baseline: per-transaction reward array for the baseline policy (paise)
    scale   : paise/txn -> ₹/1k txns is ×10.

    Same resample indices are used for every method and the baseline (common
    random numbers), so the improvement CI is properly paired.
    """
    rng = np.random.default_rng(seed)
    names = list(per_txn)
    n = len(baseline)
    val_samples = {name: np.empty(n_resamples) for name in names}
    imp_samples = {name: np.empty(n_resamples) for name in names}
    base_samples = np.empty(n_resamples)
    for b in range(n_resamples):
        idx = rng.integers(0, n, n)
        base_mean = baseline[idx].mean()
        base_samples[b] = base_mean * scale
        for name in names:
            m = per_txn[name][idx].mean()
            val_samples[name][b] = m * scale
            imp_samples[name][b] = (m - base_mean) * scale

    out: dict[str, dict] = {}
    for name in names:
        vlo, vhi = np.percentile(val_samples[name], [2.5, 97.5])
        ilo, ihi = np.percentile(imp_samples[name], [2.5, 97.5])
        out[name] = {
            "true_value": float(per_txn[name].mean() * scale),
            "value_ci": (float(vlo), float(vhi)),
            "improvement": float((per_txn[name].mean() - baseline.mean()) * scale),
            "improvement_ci": (float(ilo), float(ihi)),
            "distinguishable_from_baseline": bool(ilo > 0 or ihi < 0),
        }
    out["_baseline"] = {
        "true_value": float(baseline.mean() * scale),
        "value_ci": (float(np.percentile(base_samples, 2.5)), float(np.percentile(base_samples, 97.5))),
    }
    return out
