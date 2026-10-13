"""EAUI aggregation and the HMM interface (Phase 3)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import pandas as pd


class State(IntEnum):
    RUNNING = 0
    REDUCED = 1
    OFF = 2


def eaui(p_running: pd.DataFrame, capacity_kt: pd.Series) -> pd.Series:
    """Capacity-weighted index from a (week × plant) frame of P(running)."""
    cap = capacity_kt.reindex(p_running.columns).astype(float)
    if cap.isna().any():
        raise KeyError(f"Missing capacity for: {list(cap[cap.isna()].index)}")
    # Plants with no estimate in a week drop out of numerator *and* denominator.
    w = p_running.notna().mul(cap, axis=1)
    return (p_running.fillna(0) * cap).sum(axis=1).div(w.sum(axis=1)).rename("EAUI")


@dataclass
class PlantHMM:
    """Per-plant HMM over weekly steps.

    · hidden states: running / reduced / off, sticky transition matrix
    · emissions: one likelihood per sensor (Bernoulli or Gaussian, see sensors.py)
    · missing data: cloudy/absent observations drop out of that week's likelihood
    · training: semi-supervised EM; documented episode weeks clamped to known states
    """

    n_states: int = 3
    stickiness_prior: float = 0.95
    transmat_: np.ndarray | None = None

    def fit(self, obs: pd.DataFrame, clamped: pd.Series | None = None) -> PlantHMM:
        raise NotImplementedError("Phase 3 — semi-supervised EM with masked emissions")

    def posterior(self, obs: pd.DataFrame) -> pd.DataFrame:
        """Forward–backward posteriors, columns = State names."""
        raise NotImplementedError("Phase 3")
