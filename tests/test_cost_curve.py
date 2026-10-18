import numpy as np
import pandas as pd
import pytest

from nsqueeze.cost_curve import Plant, cost_curve, predicted_utilisation, realised_price
from nsqueeze.params import ParameterSet

EF = ParameterSet.load().value("carbon.gas_emission_factor")

# Synthetic plants — test fixtures only, not a plant universe.
PLANTS = [
    Plant("A", "Alpha", "NL", 1000, energy_gj_lhv=28.5),
    Plant("B", "Beta", "DE", 800, energy_gj_lhv=35.7),
    Plant("C", "Gamma", "PL", 500, energy_gj_lhv=29.7),
    Plant("U", "Uk", "GB", 600, energy_gj_lhv=30.0, ets_scope="UK"),
]


def test_curve_is_sorted_and_excludes_uk():
    c = cost_curve(PLANTS, ttf=50, eua=80, ef=EF)
    assert list(c.plant_id) == ["A", "C", "B"]
    assert c.cash_cost.is_monotonic_increasing
    assert c.cum_capacity_kt.iloc[-1] == 2300


def test_u_pred_bounds_and_monotonicity():
    idx = pd.date_range("2022-01-01", periods=90, freq="D")
    cap = pd.Series({"A": 1000, "B": 800})
    low = pd.DataFrame({"A": np.full(90, -200.0), "B": np.full(90, -300.0)}, index=idx)
    high = low + 400
    u_low = predicted_utilisation(low, cap, theta=0, softness=25)
    u_high = predicted_utilisation(high, cap, theta=0, softness=25)
    assert ((u_low >= 0) & (u_low <= 1)).all()
    assert (u_high > u_low).all()


def test_u_pred_has_no_default_parameters():
    with pytest.raises(TypeError):
        predicted_utilisation(pd.DataFrame(), pd.Series(dtype=float))  # type: ignore[call-arg]


def test_realised_price_is_cheapest_origin():
    df = pd.DataFrame({"TT": [500, 480], "DZ": [490, 495]})
    assert realised_price(df).tolist() == [490, 480]
