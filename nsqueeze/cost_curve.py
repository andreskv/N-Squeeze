"""Layer 2

Plant cost curve and predicted utilisation.

    p_N(t)    ≈ min_o P_imp,o(t)                                 realised price
    M_i(t)    = p_N(t) − C_EU,i(t; κ = 1)                        marginal margin
    U_pred(t) = Σ cap_i · σ((M̄_i(t) − θ) / s) / Σ cap_i          smooth run rule

M̄_i is M_i averaged over a look-back window (start: 30 days) to mimic decision
lags. θ, s and O are fitted jointly on pre-2026 episodes, then frozen.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nsqueeze import cost_stack as cs


@dataclass(frozen=True)
class Plant:
    plant_id: str
    name: str
    country: str
    capacity_kt: float
    energy_gj_lhv: float = 35.7  # no plant-level evidence yet, so every plant gets the EU average
    integrated_downstream: bool = False  # values NH3 at downstream netback; flag only in Phase 1
    ets_scope: str = "EU"  # "UK" plants need UK gas & carbon prices — never mix into the EU curve
    lat: float | None = None
    lon: float | None = None


def load_plants(path) -> list[Plant]:
    df = pd.read_csv(path)
    return [
        Plant(
            plant_id=r.plant_id,
            name=r.name,
            country=r.country,
            capacity_kt=r.capacity_kt_nh3,
            energy_gj_lhv=r.energy_gj_lhv_per_t if pd.notna(r.energy_gj_lhv_per_t) else 35.7,
            integrated_downstream=bool(r.integrated_downstream),
            ets_scope=r.ets_scope,
            lat=r.lat,
            lon=r.lon,
        )
        for r in df.itertuples()
    ]


def cost_curve(
    plants: Sequence[Plant],
    *,
    ttf: float,
    eua: float,
    other: float = 0.0,
    hhv_over_lhv: float = 1.108,
    ef: float = 56.1,
) -> pd.DataFrame:
    """Merit-order snapshot: one row per EU-ETS plant, sorted by marginal cash cost (κ = 1)."""
    rows = []
    for p in plants:
        if p.ets_scope != "EU":
            continue
        g = cs.gas_intensity(p.energy_gj_lhv, hhv_over_lhv)
        e = cs.emission_intensity(p.energy_gj_lhv, ef)
        rows.append(
            {
                "plant_id": p.plant_id,
                "name": p.name,
                "country": p.country,
                "capacity_kt": p.capacity_kt,
                "gas_term": g * ttf,
                "carbon_term": e * eua,
                "other": other,
                "integrated_downstream": p.integrated_downstream,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["cash_cost"] = df.gas_term + df.carbon_term + df.other
    df = df.sort_values("cash_cost", kind="stable").reset_index(drop=True)
    df["cum_capacity_kt"] = df.capacity_kt.cumsum()
    df["x0_kt"] = df.cum_capacity_kt - df.capacity_kt
    return df


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def predicted_utilisation(
    margins: pd.DataFrame,
    capacity_kt: pd.Series,
    *,
    theta: float,
    softness: float,
    lookback_days: int = 30,
) -> pd.Series:
    """U_pred(t) from a (date × plant) frame of daily margins M_i(t), €/t.

    ``theta`` and ``softness`` are *fitted* parameters — the caller must pass
    frozen values (see ``config/freeze.yaml``); there are deliberately no defaults.
    """
    if softness <= 0:
        raise ValueError("softness s must be positive")
    m_bar = margins.rolling(f"{lookback_days}D", min_periods=1).mean()
    p_run = sigmoid((m_bar - theta) / softness)
    cap = capacity_kt.reindex(margins.columns).astype(float)
    if cap.isna().any():
        raise KeyError(f"Missing capacity for: {list(cap[cap.isna()].index)}")
    return (p_run * cap).sum(axis=1).div(cap.sum()).rename("U_pred")


def realised_price(landed_by_origin: pd.DataFrame) -> pd.Series:
    """p_N(t) ≈ min over origins actually shipping to Europe (Comext decides which columns)."""
    return landed_by_origin.min(axis=1).rename("p_N")
