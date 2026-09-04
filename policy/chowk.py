"""`chowk` — the method this project argues for (AGENT_BRIEF §6, item 4).

Doubly-robust estimation computed over the legacy logs PLUS a small ε-exploration
slice (ε = 0.03, minimum propensity 0.01). The exploration restores coverage in
the cells the legacy policy starved, so the DR correction can finally fire there
— the estimate becomes honest and the policy picks right. Importance weights are
clipped at 50 and the number of clipped samples is reported.

chowk fits its own Q-model on the *combined* data (legacy + exploration), so the
model term too benefits from the restored coverage. It never imports ground
truth — it sees only logged events.
"""

from __future__ import annotations

from estimators.core import LoggedDataset, Method, QModel

WEIGHT_CLIP = 50.0


def build(combined: LoggedDataset, qmodel: QModel | None = None) -> Method:
    """`combined` must be the legacy logs concatenated with the exploration slice."""
    if qmodel is None:
        qmodel = QModel().fit(combined)
    return Method("chowk", "dr", weight_clip=WEIGHT_CLIP).fit(combined, qmodel=qmodel)
