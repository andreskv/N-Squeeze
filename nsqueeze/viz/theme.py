"""N-Squeeze visual identity for matplotlib .

Concept — *from 1000 °C to −33 °C*: the chain runs from a hot reformer to cryogenic
ammonia. Heat (orange) marks gas and European cost; cryo (aqua/blue) marks nitrogen
and the US comparator. Everything else stays quiet ink.

Categorical order (validated with the dataviz palette validator, both modes):
    1 orange · 2 blue · 3 aqua · 4 yellow · 5 magenta · 6 green · 7 violet · 8 red
Light mode: aqua, yellow and magenta sit below 3:1 on the surface → always direct-label
or provide a table when those slots are used.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

import matplotlib as mpl
from cycler import cycler
from matplotlib import font_manager

Mode = Literal["light", "dark"]


@dataclass(frozen=True)
class Palette:
    page: str
    surface: str
    grid: str
    axis: str
    text: str
    text_2: str
    muted: str
    series: tuple[str, ...]
    heat: str  # brand accent — highlights, never a 9th series
    cryo: str
    flag: str  # grade-C / warning (always paired with ⚑ + label)


LIGHT = Palette(
    page="#f3f5f6",
    surface="#fbfcfc",
    grid="#e2e6e9",
    axis="#c5ccd2",
    text="#0b1218",
    text_2="#47535e",
    muted="#74808b",
    series=("#eb6834", "#2a78d6", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"),
    heat="#d9531e",
    cryo="#0e8fa8",
    flag="#b7791f",
)
DARK = Palette(
    page="#0a0e13",
    surface="#10161d",
    grid="#1e2833",
    axis="#2c3946",
    text="#e9eef2",
    text_2="#a9b5c1",
    muted="#74828f",
    series=("#d95926", "#3987e5", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"),
    heat="#ff7a33",
    cryo="#62d2e8",
    flag="#fab219",
)

# Semantic slots — colour follows the entity, never its rank.
SLOT = {"EU": 0, "TTF": 0, "US": 1, "HH": 1, "NH3": 2, "import": 2, "CBAM": 6, "freight": 3}

FONT_STACK = ["Archivo", "IBM Plex Sans", "Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"]
MONO_STACK = ["IBM Plex Mono", "Cascadia Mono", "Consolas", "DejaVu Sans Mono"]


def palette(mode: Mode = "light") -> Palette:
    return DARK if mode == "dark" else LIGHT


def _available(stack: list[str]) -> list[str]:
    have = {f.name for f in font_manager.fontManager.ttflist}
    return [f for f in stack if f in have] or ["DejaVu Sans"]


def rc(mode: Mode = "light") -> dict:
    p = palette(mode)
    return {
        "figure.facecolor": p.page,
        "axes.facecolor": p.surface,
        "savefig.facecolor": p.page,
        "axes.edgecolor": p.axis,
        "axes.labelcolor": p.text_2,
        "axes.titlecolor": p.text,
        "axes.titleweight": "semibold",
        "axes.titlesize": 13,
        "axes.titlelocation": "left",
        "axes.titlepad": 14,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "axes.prop_cycle": cycler(color=list(p.series)),
        "grid.color": p.grid,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",  # never dashed
        "xtick.color": p.muted,
        "ytick.color": p.muted,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "ytick.left": False,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "lines.markersize": 7,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": p.text_2,
        "text.color": p.text,
        "font.family": "sans-serif",
        "font.sans-serif": _available(FONT_STACK),
        "font.monospace": _available(MONO_STACK),
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }


def use(mode: Mode = "light") -> Palette:
    mpl.rcParams.update(rc(mode))
    return palette(mode)


@contextmanager
def themed(mode: Mode = "light") -> Iterator[Palette]:
    with mpl.rc_context(rc(mode)):
        yield palette(mode)
