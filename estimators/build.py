"""Assemble the four methods from logged data — shared by eval and recovery.

direct and dr share ONE Q-model fit on the legacy logs; chowk fits its own on the
legacy+exploration data. Everything is deterministic (fixed random_state).
"""

from __future__ import annotations

import os

from estimators.core import LoggedDataset, QModel
from estimators import direct, dr, ips
from policy import chowk

_ROOT = os.path.dirname(os.path.dirname(__file__))
LOGS_PATH = os.path.join(_ROOT, "data", "logs.jsonl")
EXPLORE_PATH = os.path.join(_ROOT, "data", "explore.jsonl")


def load_datasets(logs_path: str = LOGS_PATH, explore_path: str = EXPLORE_PATH):
    legacy = LoggedDataset.from_file(logs_path)
    explore = LoggedDataset.from_file(explore_path)
    return legacy, explore


def build_all(legacy_ds: LoggedDataset, explore_ds: LoggedDataset) -> dict:
    """Return {name -> fitted Method} for direct, ips, snips, dr, chowk."""
    qmodel = QModel().fit(legacy_ds)
    combined = LoggedDataset.concat(legacy_ds, explore_ds)
    # chowk shares the SAME (legacy) Q-model as direct/dr, so the only thing that
    # separates chowk from dr is the exploration data in its DR correction term.
    return {
        "direct": direct.build(legacy_ds, qmodel=qmodel),
        "ips": ips.build(legacy_ds),
        "snips": ips.build_snips(legacy_ds),
        "dr": dr.build(legacy_ds, qmodel=qmodel),
        "chowk": chowk.build(combined, qmodel=qmodel),
    }
