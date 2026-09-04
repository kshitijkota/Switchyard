"""`dr` — doubly robust (AGENT_BRIEF §6.1).

The direct model plus IPW-weighted residuals: unbiased if EITHER the model or the
propensities are right, and lower-variance than plain IPS. On the legacy logs its
correction term still cannot fire in starved cells (no logged action there), so
in exactly those cells it collapses back to the (biased) direct model — which is
what switchyard's exploration slice fixes.
"""

from __future__ import annotations

from estimators.core import LoggedDataset, Method, QModel


def build(ds: LoggedDataset, qmodel: QModel | None = None, weight_clip: float | None = None) -> Method:
    return Method("dr", "dr", weight_clip=weight_clip).fit(ds, qmodel=qmodel)
