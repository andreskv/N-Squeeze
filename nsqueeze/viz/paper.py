"""Figure style for print: serif, quiet.

Conventions:
* serif text (STIX), 9 pt;
* black and grey carry most marks; colour is reserved for one or two identities
  and always paired with a line style, so figures survive greyscale printing;
* documented curtailment episodes are light grey vertical bands, labelled once;
* no titles inside figures: the caption is the title, notes go in the caption.
"""

from __future__ import annotations

from contextlib import contextmanager

import matplotlib as mpl
import pandas as pd

INK = "#111111"
GREY = "#6b6b6b"
LIGHT = "#d9d9d9"
BAND = "#ececec"
# Categorical colours, light mode, in a fixed order
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"

RC = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "axes.titlelocation": "left",
    "axes.titleweight": "normal",
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.1,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "svg.fonttype": "path",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}


def use() -> None:
    mpl.rcParams.update(RC)


@contextmanager
def style():
    with mpl.rc_context(RC):
        yield


def shade_episodes(ax, months: pd.Series, *, label: bool = True, width_months: int = 2):
    """Light grey bands from each documented onset month (one per episode cluster)."""
    for i, m in enumerate(sorted(set(months))):
        ax.axvspan(
            m,
            m + pd.DateOffset(months=width_months),
            color=BAND,
            lw=0,
            zorder=0,
            label="Documented curtailment onset" if (label and i == 0) else None,
        )


def panel_label(ax, text: str) -> None:
    ax.set_title(text, loc="left", fontsize=9, pad=4)


def cost_parity(panel: pd.DataFrame, band: pd.DataFrame, cost: dict, onset_months):
    """Headline figure: EU urea cash cost (gas + carbon, κ = 1) against the import-parity band."""
    import matplotlib.pyplot as plt

    with style():
        f, ax = plt.subplots(figsize=(6.4, 3.2))
        shade_episodes(ax, onset_months)
        ax.fill_between(
            band.index,
            band.low,
            band.high,
            color=BLUE,
            alpha=0.22,
            lw=0,
            label="Import parity (band)",
        )
        ax.plot(cost["eu_average"].total, color=INK, label="EU cost, average plant")
        ax.plot(
            cost["best_practice"].total, color=INK, ls="--", label="EU cost, best-practice plant"
        )
        ax.set_ylabel("€ per tonne of urea")
        ax.legend(loc="upper left")
        f.tight_layout()
    return f


def monte_carlo(e2: pd.DataFrame, e3n: pd.DataFrame, e3p: pd.DataFrame, e4: pd.DataFrame):
    """Simulation-study figure: nulls, selection (no skill / true skill), few events."""
    import matplotlib.pyplot as plt

    with style():
        f, axs = plt.subplots(2, 2, figsize=(6.4, 4.6))
        a = axs[0, 0]
        a.plot(e2.phi, e2.reject_naive, "o-", color=INK, ms=3, label="Permutation")
        a.plot(e2.phi, e2.reject_circular, "s--", color=BLUE, ms=3, label="Circular shift")
        a.set_xlabel("Persistence $\\phi$")
        a.set_ylabel("Rejection rate")
        panel_label(a, "(a) Nulls")
        for a, df, ylab, title in (
            (axs[0, 1], e3n, "False discoveries", "(b) Selection, no skill"),
            (axs[1, 0], e3p, "Detection rate", "(c) Selection, true Sharpe 1.0"),
        ):
            a.plot(
                df.n_trials, df.declared_psr, "o-", color=INK, ms=3, label="Probabilistic Sharpe"
            )
            a.plot(df.n_trials, df.declared_dsr, "s--", color=BLUE, ms=3, label="Deflated Sharpe")
            a.set_xscale("log")
            a.set_xlabel("Trials N")
            a.set_ylabel(ylab)
            panel_label(a, title)
        a = axs[1, 1]
        a.plot(e4.episodes, e4.reject_iid, "o-", color=INK, ms=3, label="Daily OLS")
        a.plot(
            e4.episodes, e4.reject_clustered, "s--", color=BLUE, ms=3, label="Clustered by episode"
        )
        a.annotate("undefined\nat G = 2", (2, 0.12), fontsize=7, color=GREY)
        a.set_xscale("log", base=2)
        a.set_xlabel("Independent episodes G")
        a.set_ylabel("Rejection rate")
        panel_label(a, "(d) Few events")
        for a in axs.flat:
            a.set_ylim(-0.02, 1.02)
            a.legend(loc="best")
        for a in (axs[0, 0], axs[0, 1], axs[1, 1]):
            a.axhline(0.05, color=GREY, lw=0.6, ls=":")
        f.tight_layout()
    return f


def framework_diagram():
    """Figure 1: physical state → model → prediction; measurement → surprise → four gates."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    with style():
        fig, ax = plt.subplots(figsize=(6.6, 2.55))
        ax.set_xlim(0, 13.4)
        ax.set_ylim(0, 4.3)
        ax.axis("off")

        def box(x, y, w, h, title, sub="", *, lw=0.7, fc="white"):
            ax.add_patch(
                FancyBboxPatch(
                    (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=INK, lw=lw
                )
            )
            ax.text(
                x + w / 2,
                y + h * (0.66 if sub else 0.5),
                title,
                ha="center",
                va="center",
                fontsize=7.6,
                color=INK,
            )
            if sub:
                ax.text(
                    x + w / 2, y + h * 0.29, sub, ha="center", va="center", fontsize=6.9, color=GREY
                )

        def arrow(p0, p1):
            ax.add_patch(
                FancyArrowPatch(
                    p0,
                    p1,
                    arrowstyle="-|>",
                    mutation_scale=7,
                    lw=0.7,
                    color=INK,
                    shrinkA=1,
                    shrinkB=1,
                )
            )

        h = 1.15
        box(0.05, 2.65, 2.55, h, "Physical state $X_t$", "flows, stocks, prices")
        box(3.1, 2.65, 2.55, h, "Structural model", r"$\hat{Q}_t = f(X_t;\,\theta)$")
        box(6.15, 2.65, 2.4, h, "Prediction", r"$\hat{Q}_t$, e.g. $U^{pred}_t$")
        box(6.15, 0.5, 2.4, h, "Measurement", r"$\tilde{Q}_t$: satellites, AIS")
        box(9.0, 1.57, 1.85, h, "Surprise", r"$S_t = \tilde{Q}_t - \hat{Q}_t$", lw=1.3)
        arrow((2.6, 3.22), (3.1, 3.22))
        arrow((5.65, 3.22), (6.15, 3.22))
        arrow((8.55, 3.0), (9.35, 2.72))
        arrow((8.55, 1.27), (9.35, 1.57))
        ax.text(
            1.33,
            2.38,
            "graded A/B/C, archived point-in-time",
            ha="center",
            fontsize=6.6,
            color=GREY,
            style="italic",
        )

        for i, g in enumerate(
            ["G1 measurement", "G2 information", "G3 price impact", "G4 tradability"]
        ):
            y = 3.3 - i * 0.92
            box(11.0, y, 2.35, 0.6, g, lw=0.6, fc="#f6f6f6")
            if i:
                arrow((12.175, y + 0.92), (12.175, y + 0.6))
        arrow((10.85, 2.15), (11.0, 3.55))
        ax.text(
            12.175,
            0.08,
            "failure at any gate is reported",
            ha="center",
            fontsize=6.6,
            color=GREY,
            style="italic",
        )
        return fig
