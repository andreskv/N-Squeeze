"""Layer 2

Nitrogen cash-cost stack.

    C_EU,i(t) = g_i · P_TTF(t) + e_i · P_EUA(t) · κ + O_i
    C_US(t)   = h · P_HH(t) + O_US
    C_urea    = 0.567 · C_NH3 + g_conv · P_gas + O_conv
    ΔC(t)     = C_EU(t) − C_US(t) / FX(t)                     (€/t, FX = USD per EUR)

"""

from __future__ import annotations

from dataclasses import dataclass

from nsqueeze import units
from nsqueeze.params import ParameterSet
from nsqueeze.provenance import Estimate, Provenance

# Pure functions


def gas_intensity(energy_gj_lhv: float, hhv_over_lhv: float = units.HHV_OVER_LHV_GAS) -> float:
    """g_i, MWh HHV of gas purchased per t NH3."""
    return units.gj_lhv_to_mwh_hhv(energy_gj_lhv, hhv_over_lhv)


def emission_intensity(energy_gj_lhv: float, ef_t_per_tj: float = 56.1) -> float:
    """e_i ≈ E_i × 0.0561 — direct tCO2 per t NH3 from the gas burned (IPCC default EF)."""
    return energy_gj_lhv * ef_t_per_tj / units.GJ_PER_TJ


def free_allocation(
    benchmark: float, historical_activity: float, cbam_factor: float, cscf: float = 1.0
) -> float:
    """FA_i = BM_NH3 × HAL_i × (1 − CF_y) × CSCF_y  (allowances per year).

    Depends on *historical* activity, which is why a curtailing plant keeps most of
    its allowances — and why κ = 1 for the run/curtail decision.
    """
    return benchmark * historical_activity * (1.0 - cbam_factor) * cscf


def accounting_kappa(free_alloc: float, output_t: float, e: float) -> float:
    """κ for margin *reporting*: 1 − FA_i / (output_i · e_i).

    Never use it to predict curtailment — that decision prices carbon at κ = 1.
    """
    return 1.0 - free_alloc / (output_t * e)


def eu_ammonia_cash_cost(ttf_eur_mwh, eua_eur_t, *, g, e, other=0.0, kappa=1.0):
    """C_EU,i in €/t NH3. ``other`` is O_i (fitted; pass 0 to get the gas + carbon terms only)."""
    return g * ttf_eur_mwh + e * eua_eur_t * kappa + other


def us_ammonia_cash_cost(hh_usd_mmbtu, *, h, other=0.0, basis_usd_mmbtu=0.0):
    """C_US in $/t NH3. No carbon term: no federal US carbon price on ammonia."""
    return h * (hh_usd_mmbtu + basis_usd_mmbtu) + other


def urea_cash_cost(c_nh3, p_gas, *, g_conv, other_conv=0.0, nh3_per_urea=units.NH3_PER_T_UREA):
    """C_urea per t urea; ``p_gas`` and ``g_conv`` must share an energy unit."""
    return nh3_per_urea * c_nh3 + g_conv * p_gas + other_conv


def cost_gap(c_eu_eur_t, c_us_usd_t, usd_per_eur):
    """ΔC = C_EU − C_US / FX, €/t. Positive = EU disadvantage."""
    return c_eu_eur_t - c_us_usd_t / usd_per_eur


# Graded model wrapper

TIERS = {
    "best_practice": "ammonia_tech.energy_best_practice",
    "best_revamped": "ammonia_tech.energy_best_revamped",
    "eu_average": "ammonia_tech.energy_eu_average",
}


@dataclass(frozen=True)
class PlantTech:
    """Energy tier of a plant. Default every plant to ``eu_average``."""

    energy_gj_lhv: float
    provenance: Provenance

    @classmethod
    def tier(cls, params: ParameterSet, tier: str = "eu_average") -> PlantTech:
        p = params[TIERS[tier]]
        return cls(float(p.value), Provenance.of(p.ref))


class CostStack:
    """Reads graded parameters once; every method returns an :class:`Estimate`."""

    def __init__(self, params: ParameterSet | None = None) -> None:
        self.P = params or ParameterSet.load()

    def _p(self, key: str) -> tuple[float, Provenance]:
        p = self.P[key]
        return float(self.P.value(key)), Provenance.of(p.ref)

    def intensities(self, tech: PlantTech) -> Estimate[tuple[float, float]]:
        hl, prov_hl = self._p("units.hhv_over_lhv_gas")
        ef, prov_ef = self._p("carbon.gas_emission_factor")
        g = gas_intensity(tech.energy_gj_lhv, hl)
        e = emission_intensity(tech.energy_gj_lhv, ef)
        return Estimate((g, e), tech.provenance | prov_hl | prov_ef)

    def eu_ammonia(self, ttf, eua, tech: PlantTech, *, include_other: bool = True, kappa=1.0):
        """C_EU,i. With ``include_other=True`` this raises until O_i has been fitted."""
        est = self.intensities(tech)
        (g, e), prov = est.value, est.provenance
        other = 0.0
        if include_other:
            other, prov_o = self._p("fitted.other_cash_cost_eu")
            prov |= prov_o
        return Estimate(eu_ammonia_cash_cost(ttf, eua, g=g, e=e, other=other, kappa=kappa), prov)

    def us_ammonia(self, hh, *, include_other: bool = True, basis_usd_mmbtu: float = 0.0):
        h, prov = self._p("us_cost.h_cf_mmbtu_per_t")
        other = 0.0
        if include_other:
            other, prov_o = self._p("fitted.other_cash_cost_us")
            prov |= prov_o
        return Estimate(
            us_ammonia_cash_cost(hh, h=h, other=other, basis_usd_mmbtu=basis_usd_mmbtu), prov
        )
