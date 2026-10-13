"""Units and conversions.

Internal convention: gas in €/MWh (HHV), plant energy in GJ (LHV), products in €/t.
"""

from __future__ import annotations

from typing import TypeVar

import numpy as np
import numpy.typing as npt

# A scalar or anything numpy can broadcast (Series and arrays pass straight through).
Num = TypeVar("Num", float, npt.NDArray[np.float64])

# Exact by definition
GJ_PER_MWH = 3.6
GJ_PER_MMBTU = 1.05506
MMBTU_PER_MWH = GJ_PER_MWH / GJ_PER_MMBTU  # 3.4121…
T_PER_SHORT_TON = 0.90718474
GJ_PER_TJ = 1_000.0
KWH_PER_TWH = 1e9
GWH_PER_TWH = 1e3

# Molar masses, g/mol
M_N = 14.007
M_NH3 = 17.031
M_UREA = 60.06
M_CO2 = 44.01

# Stoichiometry
NH3_PER_T_UREA = 2 * M_NH3 / M_UREA  # 0.567 t NH3 per t urea
CO2_BOUND_PER_T_UREA = M_CO2 / M_UREA  # 0.733 t CO2 per t urea
NH3_PER_T_N = M_NH3 / M_N  # 1.216 t NH3-eq per t N
N_CONTENT_NH3 = M_N / M_NH3  # 0.822
N_CONTENT_UREA = 2 * M_N / M_UREA  # 0.466 (commercial grade 0.46)

# Physical range, graded B in config/parameters.yaml
HHV_OVER_LHV_GAS = 1.108


# Energy
def mwh_to_mmbtu(x: Num) -> Num:
    return x * MMBTU_PER_MWH


def mmbtu_to_mwh(x: Num) -> Num:
    return x / MMBTU_PER_MWH


def kwh_to_twh(x: Num) -> Num:
    """ENTSOG flows (kWh/d) → TWh/d, the unit of the AGSI+ storage balance."""
    return x / KWH_PER_TWH


def gwh_to_twh(x: Num) -> Num:
    """AGSI+ injection/withdrawal and ALSI+ send-out (GWh/d) → TWh/d."""
    return x / GWH_PER_TWH


def gj_lhv_to_mwh_hhv(e_gj_lhv: Num, hhv_over_lhv: float = HHV_OVER_LHV_GAS) -> Num:
    """Plant energy intensity (GJ LHV/t) → gas purchased (MWh HHV/t).

    g = E × (HHV/LHV) / 3.6. EU average 35.7 GJ/t → ≈ 11.0 MWh/t.
    """
    return e_gj_lhv * hhv_over_lhv / GJ_PER_MWH


def mwh_hhv_to_gj_lhv(g_mwh_hhv: Num, hhv_over_lhv: float = HHV_OVER_LHV_GAS) -> Num:
    return g_mwh_hhv * GJ_PER_MWH / hhv_over_lhv


# Prices
def eur_mwh_to_usd_mmbtu(p_eur_mwh: Num, usd_per_eur: Num) -> Num:
    """TTF quote (€/MWh) → $/MMBtu, comparable with Henry Hub."""
    return p_eur_mwh * usd_per_eur / MMBTU_PER_MWH


def usd_mmbtu_to_eur_mwh(p_usd_mmbtu: Num, usd_per_eur: Num) -> Num:
    """Henry Hub quote ($/MMBtu) → €/MWh, comparable with TTF."""
    return p_usd_mmbtu * MMBTU_PER_MWH / usd_per_eur


def per_short_ton_to_per_t(x: Num) -> Num:
    """$/short ton → $/metric tonne."""
    return x / T_PER_SHORT_TON


def per_t_to_per_short_ton(x: Num) -> Num:
    return x * T_PER_SHORT_TON


# Products
def nh3_equivalent(n_content: Num) -> Num:
    """t NH3-equivalent per t of product, from its nitrogen mass fraction."""
    return n_content * NH3_PER_T_N


def gas_use_from_sensitivity(usd_per_short_ton_per_usd_mmbtu: Num) -> Num:
    """Invert a 10-K gas sensitivity into implied gas use (MMBtu per metric tonne).

    CF: $32/short ton per $1/MMBtu  →  32 / 0.9072 ≈ 35.3 MMBtu/t NH3.
    """
    return per_short_ton_to_per_t(usd_per_short_ton_per_usd_mmbtu)
