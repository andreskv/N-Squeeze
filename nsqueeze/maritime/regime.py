"""Chokepoint regime indicator R(t).

Three regimes condition every strategy:
  NORMAL                  — no material stress
  EU_GAS_STRESS           — Europe-only gas shock (2021–22 template)
  GLOBAL_SUPPLY_SHOCK     — gas *and* global nitrogen supply shock (2026 template)

Sources disagree materially in 2026 (PortWatch ≈ 4/day vs other trackers ≈ 10/day
against an ≈ 85/day baseline). Fix ONE primary source in advance, then
report results under the alternatives. Thresholds are pre-registered, not tuned.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pandas as pd


class Regime(StrEnum):
    NORMAL = "normal"
    EU_GAS_STRESS = "eu_gas_stress"
    GLOBAL_SUPPLY_SHOCK = "global_supply_shock"


@dataclass(frozen=True)
class RegimeRule:
    """Pre-registered thresholds. The values are in ``config/parameters.yaml`` (``regime``)."""

    transit_ratio_shock: float  # transits / baseline below this → chokepoint shock
    gas_spread_stress: float  # TTF − HH (€/MWh) above this → EU gas stress
    smoothing_days: int = 7


def classify(transit_ratio: pd.Series, gas_spread: pd.Series, rule: RegimeRule) -> pd.Series:
    """Label each day. Chokepoint shock dominates gas stress."""
    tr = transit_ratio.rolling(rule.smoothing_days, min_periods=1).mean()
    gs = gas_spread.rolling(rule.smoothing_days, min_periods=1).mean()
    tr, gs = tr.align(gs, join="inner")
    out = pd.Series(Regime.NORMAL.value, index=tr.index, name="R")
    out[gs > rule.gas_spread_stress] = Regime.EU_GAS_STRESS.value
    out[tr < rule.transit_ratio_shock] = Regime.GLOBAL_SUPPLY_SHOCK.value
    return out
