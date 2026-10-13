"""Signature charts. Every chart carries its evidence footnote (grade + grade-C flags)."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from nsqueeze.provenance import Provenance
from nsqueeze.viz.theme import Mode, palette, themed


def evidence_footnote(
    fig: Figure, provenance: Provenance, *, source: str = "", mode: Mode = "light"
) -> None:
    p = palette(mode)
    flag = "▲ " if provenance.flags else ""  # ⚑ is missing from common UI fonts
    text = f"{flag}{provenance.footnote()}"
    if source:
        text += f"   Source: {source}"
    fig.text(
        0.01,
        -0.02,
        text,
        ha="left",
        va="top",
        fontsize=7.5,
        color=p.flag if provenance.flags else p.muted,
        wrap=True,
    )


def plot_cost_curve(
    curve: pd.DataFrame,
    *,
    price: float | None = None,
    price_label: str = "Import parity",
    mode: Mode = "light",
    ax: Axes | None = None,
    highlight: str | None = None,
) -> Axes:
    """Merit-order cost curve: capacity on x, marginal cash cost (κ = 1) on y.

    ``curve`` is the output of :func:`nsqueeze.cost_curve.cost_curve`. Bars are
    capacity-wide; a 2 px surface gap separates neighbours (no outlines).
    Plants above ``price`` are drawn in the muted tone: they are predicted to curtail.
    """
    with themed(mode) as p:
        if ax is None:
            _, ax = plt.subplots(figsize=(9, 4.6))
        gap = curve.capacity_kt.sum() * 0.0015
        for r in curve.itertuples():
            out_of_money = price is not None and r.cash_cost > price
            colour = p.axis if out_of_money else p.series[0]
            if highlight and r.plant_id == highlight:
                colour = p.series[1]
            ax.add_patch(
                Rectangle(
                    (r.x0_kt + gap / 2, 0),
                    r.capacity_kt - gap,
                    r.cash_cost,
                    facecolor=colour,
                    edgecolor="none",
                )
            )
        ax.set_xlim(0, curve.cum_capacity_kt.max())
        ax.set_ylim(0, curve.cash_cost.max() * 1.15)
        ax.set_xlabel("Cumulative EU ammonia capacity (kt NH₃ / yr)")
        ax.set_ylabel("Marginal cash cost, κ = 1 (€/t NH₃)")
        if price is not None:
            ax.axhline(price, color=p.text, lw=1.2)
            ax.annotate(
                f"{price_label}  €{price:,.0f}/t",
                (ax.get_xlim()[1], price),
                xytext=(-4, 6),
                textcoords="offset points",
                ha="right",
                fontsize=9,
                color=p.text,
            )
            running = curve.loc[curve.cash_cost <= price, "capacity_kt"].sum()
            share = running / curve.capacity_kt.sum()
            ax.set_title(f"EU ammonia cost curve — {share:.0%} of capacity in the money")
        else:
            ax.set_title("EU ammonia cost curve")
    return ax


def plot_cbam_ramp(
    ramp: pd.DataFrame, *, mode: Mode = "light", labels: Mapping[str, str] | None = None
) -> Figure:
    """CBAM €/t NH₃ by origin (columns) and year (index), direct-labelled at the line ends."""
    with themed(mode) as p:
        fig, ax = plt.subplots(figsize=(8.5, 4.4))
        handles = []
        for i, col in enumerate(ramp.columns):
            colour = p.series[i]
            (line,) = ax.plot(ramp.index, ramp[col], color=colour)
            handles.append(line)
            ax.plot(
                ramp.index[-1],
                ramp[col].iloc[-1],
                "o",
                color=colour,
                markeredgecolor=p.surface,
                markeredgewidth=2,
            )
            ax.annotate(
                f"{(labels or {}).get(col, col)}  €{ramp[col].iloc[-1]:,.0f}",
                (ramp.index[-1], ramp[col].iloc[-1]),
                xytext=(8, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
                color=p.text_2,
            )
        ax.set_xlim(ramp.index.min() - 0.3, ramp.index.max() + 2.2)
        ax.set_ylim(0, None)
        ax.set_ylabel("CBAM charge (€/t NH₃)")
        ax.set_xticks(list(ramp.index))
        ax.legend(
            handles,
            [(labels or {}).get(c, c) for c in ramp.columns],
            loc="upper left",
            ncols=len(ramp.columns),
        )
    return fig
