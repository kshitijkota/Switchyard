"""`direct` — the industry-standard method (AGENT_BRIEF §6.1).

A gradient-boosted classifier for P(success | context, processor) trained on the
legacy logs; route greedily by predicted expected reward. Its per-cell value is
the model plug-in mean. This is the method expected to fail: its self-estimate
extrapolates into starved cells with no signal that it is guessing.
"""

from __future__ import annotations

from estimators.core import LoggedDataset, Method, QModel


def build(ds: LoggedDataset, qmodel: QModel | None = None) -> Method:
    return Method("direct", "direct").fit(ds, qmodel=qmodel)
