"""`ips` — inverse propensity scoring, and its self-normalised variant SNIPS
(AGENT_BRIEF §6.1).

Reweight each logged reward by 1/propensity to undo the legacy policy's skew.
Unbiased where there is coverage, but high-variance — and simply undefined in
cells where an action was never logged. SNIPS divides by the sum of weights
instead of the count, trading a little bias for much lower variance.
"""

from __future__ import annotations

from estimators.core import LoggedDataset, Method


def build(ds: LoggedDataset, weight_clip: float | None = None) -> Method:
    return Method("ips", "ips", weight_clip=weight_clip).fit(ds)


def build_snips(ds: LoggedDataset, weight_clip: float | None = None) -> Method:
    return Method("snips", "snips", weight_clip=weight_clip).fit(ds)
