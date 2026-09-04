"""Context discretisation into cells — the hypothesis space shared by all four
methods (AGENT_BRIEF §6 interpretation, logged in NOTES 2026-09-04).

A cell is (method, issuer, amount_bucket). Amount-bucket boundaries sit at
economically meaningful paise (the fee crossovers and the ₹10k interaction),
chosen BEFORE seeing any result — never tuned to outcomes.

This module sees only observable context, never ground truth.
"""

from __future__ import annotations

import bisect
from datetime import datetime
from typing import NamedTuple

from events import ISSUERS, METHODS, PROCESSORS, Context

# Bucket boundaries (paise): <₹300, ₹300–1k, ₹1k–3k, ₹3k–10k, >₹10k.
AMOUNT_BOUNDS = (30_000, 100_000, 300_000, 1_000_000)
BUCKET_LABELS = ("<₹300", "₹300–1k", "₹1k–3k", "₹3k–10k", ">₹10k")
N_BUCKETS = len(BUCKET_LABELS)
# A representative amount per bucket for building canonical contexts (segment
# table + direct's per-cell display). Interior buckets use a mid-ish value;
# the open top bucket uses ₹12k where the pb+card interaction is live.
BUCKET_REPR_PAISE = (15_000, 60_000, 180_000, 600_000, 1_200_000)

PROC_INDEX = {p: i for i, p in enumerate(PROCESSORS)}


def amount_bucket(amount_paise: int) -> int:
    return bisect.bisect_right(AMOUNT_BOUNDS, amount_paise)


class Cell(NamedTuple):
    method: str
    issuer: str
    bucket: int

    def label(self) -> str:
        return f"{self.method}×{self.issuer}×{BUCKET_LABELS[self.bucket]}"


def cell_of(method: str, issuer: str, amount_paise: int) -> Cell:
    return Cell(method, issuer, amount_bucket(amount_paise))


def cell_of_context(ctx: Context) -> Cell:
    return Cell(ctx.method, ctx.issuer, amount_bucket(ctx.amount_paise))


ALL_CELLS: list[Cell] = [
    Cell(m, i, b) for m in METHODS for i in ISSUERS for b in range(N_BUCKETS)
]
CELL_INDEX: dict[Cell, int] = {c: k for k, c in enumerate(ALL_CELLS)}
N_CELLS = len(ALL_CELLS)


def representative_context(cell: Cell, hour: int = 12) -> Context:
    """Canonical context for a cell: representative amount, midday, day 0 (never a
    degradation day). Used only for display/segment-table lookups."""
    return Context(
        txn_id=f"repr-{cell.method}-{cell.issuer}-{cell.bucket}",
        ts=datetime(2026, 1, 1, hour, 0),
        method=cell.method,
        issuer=cell.issuer,
        amount_paise=BUCKET_REPR_PAISE[cell.bucket],
        hour=hour,
    )


def legacy_mode_proc(cell: Cell) -> str:
    """Deterministic 'exploit' choice mirroring the legacy policy's mode; used as
    a fallback for cells with no logged coverage. Depends only on method and a
    high-amount flag — no ground truth."""
    if cell.bucket == N_BUCKETS - 1 or cell.bucket == N_BUCKETS - 2:
        # top two buckets straddle ₹5k; legacy sends >₹5k to pb. Use pb for the
        # top bucket, and for ₹3k–10k default to pb as the covered high-amount pick.
        return "pb"
    if cell.method == "upi":
        return "pa"
    return "pc"
