"""Crossover plot — AGENT_BRIEF §4.3.

Expected net reward vs ticket size, one line per processor, crossovers marked.
Rendered for the headline `hdfc × upi` segment because that is the segment the
results table turns on: pa has the highest *success rate* there (the +0.07
boost) yet is NOT the best *expected-reward* choice below the crossover, because
its flat ₹4 fee eats the thin margin on small tickets.

For a fixed segment each processor's expected reward is *linear* in amount
(success prob is constant in amount here), so crossovers are solved exactly
rather than eyeballed, and written to artifacts/crossover.json for the README.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from events import PROCESSORS, Context  # noqa: E402
from sim import economics as ec  # noqa: E402
from sim.ground_truth import success_prob  # noqa: E402

_ARTIFACTS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")
_PROC_COLOR = {"pa": "#2f6db3", "pb": "#c65a2e", "pc": "#4c9a52"}


def _segment_context(amount_paise: int) -> Context:
    # hdfc × upi, midday, a non-degraded day (day_index 0). Only pb+card+amount
    # and the degradation window depend on amount/time, neither active here, so
    # success prob is constant across amount for this segment.
    return Context("seg", datetime(2026, 1, 1, 12, 0), "upi", "hdfc", amount_paise, 12)


def _line_params(processor: str) -> tuple[float, float]:
    """Return (slope, intercept) of expected reward (paise) vs amount (paise).

    exp_p(A) = s_p * (revenue(A) - cost_p(A)); revenue and cost are linear in A,
    so this is m*A + b. Derived numerically from two points to stay in lockstep
    with economics.py rather than re-deriving the algebra by hand.
    """
    s = success_prob(_segment_context(100_000), processor)
    a0, a1 = 100_000, 200_000
    y0 = s * ec.reward_if_success_paise(processor, a0)
    y1 = s * ec.reward_if_success_paise(processor, a1)
    m = (y1 - y0) / (a1 - a0)
    b = y0 - m * a0
    return m, b


def _crossovers(amt_lo: int, amt_hi: int) -> list[dict]:
    params = {p: _line_params(p) for p in PROCESSORS}

    def top(a: float) -> str:
        return max(PROCESSORS, key=lambda p: params[p][0] * a + params[p][1])

    switches: list[dict] = []
    for i, p in enumerate(PROCESSORS):
        for q in PROCESSORS[i + 1 :]:
            (mp, bp), (mq, bq) = params[p], params[q]
            if mp == mq:
                continue
            a_star = (bq - bp) / (mp - mq)
            if not (amt_lo < a_star < amt_hi):
                continue
            # Only a real crossover if the two lines are the top choice on
            # either side of the crossing (an actual argmax switch).
            lo_top, hi_top = top(a_star - 1), top(a_star + 1)
            if {lo_top, hi_top} == {p, q}:
                switches.append(
                    {
                        "amount_paise": int(round(a_star)),
                        "amount_rupees": round(a_star / 100, 2),
                        "from_processor": lo_top,
                        "to_processor": hi_top,
                    }
                )
    switches.sort(key=lambda s: s["amount_paise"])
    return switches


def make_crossover_plot(out_png: str | None = None, out_json: str | None = None) -> dict:
    out_png = out_png or os.path.join(_ARTIFACTS, "crossover.png")
    out_json = out_json or os.path.join(_ARTIFACTS, "crossover.json")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    amt_lo, amt_hi = 5_000, 500_000     # ₹50 .. ₹5,000
    import numpy as np
    from sim import plotstyle as ps

    grid = np.linspace(amt_lo, amt_hi, 400)
    fig, ax = ps.new_fig(8.8, 5.0)
    for p in PROCESSORS:
        m, b = _line_params(p)
        y = (m * grid + b) / 100
        ax.plot(grid / 100, y, label=p, color=ps.PROC[p], lw=2.6)
        ax.annotate(p, xy=(grid[-1] / 100, y[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=10.5, fontweight="bold", color=ps.PROC[p])

    crossings = _crossovers(amt_lo, amt_hi)
    for c in crossings:
        ax.axvline(c["amount_rupees"], color=ps.FAINT, ls="--", lw=1.2)
        ax.annotate(
            f"₹{c['amount_rupees']:.0f}\n{c['from_processor']} → {c['to_processor']}",
            xy=(c["amount_rupees"], ax.get_ylim()[1] * 0.9),
            ha="center", va="top", fontsize=9.5, color=ps.MUTED,
        )

    ax.axhline(0, color=ps.FAINT, lw=1.0)
    ax.set_xlabel("ticket size (₹)")
    ax.set_ylabel("expected net reward (₹)")
    ax.margins(x=0.03)
    ps.titled(ax, "Why fees flip the ranking",
              "expected net reward vs ticket size · segment hdfc × upi · pa's flat fee costs it the small tickets")
    fig.subplots_adjust(top=0.84, left=0.10, right=0.94, bottom=0.12)
    fig.savefig(out_png, metadata=ps.METADATA)
    plt.close(fig)

    summary = {
        "segment": {"issuer": "hdfc", "method": "upi", "hour": 12},
        "success_prob": {p: success_prob(_segment_context(100_000), p) for p in PROCESSORS},
        "crossovers": crossings,
        "note": "Below the crossover the flat-fee processor is not the best "
        "expected-reward choice despite the best success rate.",
    }
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    return summary


def make_extrapolation_plot(evidence: dict, out_png: str | None = None) -> str:
    """Plot direct's predicted vs TRUE pa success across ticket sizes (the
    extrapolation failure). `evidence` is eval.harness.pa_extrapolation_evidence
    output — no model needed here, so this stays a pure plotter."""
    out_png = out_png or os.path.join(_ARTIFACTS, "extrapolation.png")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    import numpy as np
    from sim import plotstyle as ps
    rows = evidence["rows"]
    amt = np.array([r["amount_rupees"] for r in rows], dtype=float)
    pred = np.array([r["predicted_pa_success"] for r in rows])
    true = np.array([r["true_pa_success"] for r in rows])

    fig, ax = ps.new_fig(8.8, 5.0)
    # shade the extrapolation gap where the model overstates the truth
    ax.fill_between(amt, true, pred, where=(pred >= true), color=ps.RED, alpha=0.14,
                    label="model overstates the truth")
    ax.plot(amt, pred, "o-", color=ps.BYGARI, lw=2.6, ms=6, label="direct model's predicted pa success")
    ax.plot(amt, true, "s--", color=ps.SWITCHYARD, lw=2.6, ms=6, label="true pa success")
    ax.axvline(5000, color=ps.FAINT, ls=":", lw=1.4)
    ax.annotate("legacy sends every >₹5k ticket to pb\n— pa is never observed here",
                xy=(5200, 0.70), xytext=(9000, 0.585), fontsize=9.5, color=ps.MUTED,
                arrowprops=dict(arrowstyle="->", color=ps.FAINT, lw=1.2))
    ax.set_xscale("log")
    ax.set_xlabel("ticket size (₹, log scale)")
    ax.set_ylabel("pa success probability")
    ax.set_ylim(0.5, 1.0)
    ax.margins(x=0.02)
    ax.legend(loc="lower left")
    ps.titled(ax, "You cannot learn what you never tried",
              "in the starved region the model predicts pa ~0.78 when the truth is 0.68 — confidently wrong")
    fig.subplots_adjust(top=0.84, left=0.10, right=0.96, bottom=0.12)
    fig.savefig(out_png, metadata=ps.METADATA)
    plt.close(fig)
    return out_png


if __name__ == "__main__":
    print(json.dumps(make_crossover_plot(), indent=2))
