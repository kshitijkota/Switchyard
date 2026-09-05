"""Shared plot styling — one consistent, polished look across every figure.

Import `apply_style()` (or use `new_fig`) at the top of any plotting function and
finish axes with `despine` / `titled`. Colours are fixed so a series (switchyard,
bygari, a processor) is the same hue in every chart.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# --- palette -------------------------------------------------------------------
INK = "#222b35"        # primary text / data ink
MUTED = "#6b7683"      # secondary text, ticks, subtitles
FAINT = "#aab3bd"      # reference lines
GRID = "#e6eaef"       # gridlines
PANEL = "#f7f9fb"      # subtle fill

SWITCHYARD = "#2e6fb7"   # blue
BYGARI = "#e07a3e"       # warm orange
GREEN = "#3f9e5a"
RED = "#d1495b"
GOLD = "#c99700"
PROC = {"pa": "#2e6fb7", "pb": "#e07a3e", "pc": "#3f9e5a"}

METADATA = {"Software": "switchyard"}   # fixed → no run timestamp in the PNG


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.facecolor": "white",
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.edgecolor": "#c3ccd6",
        "axes.linewidth": 1.0,
        "axes.labelcolor": MUTED,
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 1.0,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "lines.solid_capstyle": "round",
    })


def despine(ax, grid_axis: str = "y") -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=1.0)
    ax.grid(False, axis="x" if grid_axis == "y" else "y")
    ax.tick_params(length=0)


def new_fig(w: float = 8.6, h: float = 5.0, grid_axis: str = "y"):
    apply_style()
    fig, ax = plt.subplots(figsize=(w, h))
    despine(ax, grid_axis)
    return fig, ax


def titled(ax, title: str, subtitle: str | None = None) -> None:
    """A bold left-aligned title with an optional muted subtitle above the axes."""
    if subtitle:
        ax.text(0.0, 1.11, title, transform=ax.transAxes, fontsize=13.5,
                fontweight="bold", color=INK, va="bottom", ha="left")
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=10.5,
                color=MUTED, va="bottom", ha="left")
    else:
        ax.text(0.0, 1.02, title, transform=ax.transAxes, fontsize=13.5,
                fontweight="bold", color=INK, va="bottom", ha="left")


def thousands(ax, axis: str = "y") -> None:
    from matplotlib.ticker import FuncFormatter
    fmt = FuncFormatter(lambda v, _pos: f"{v:,.0f}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)
