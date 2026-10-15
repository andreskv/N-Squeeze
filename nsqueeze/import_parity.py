"""Layer 2


Landed import parity into NW Europe.

    P_imp,o(t) = (FOB_o + F_o)(1 + τ_o) + T_o + CBAM_o
    CBAM_o    ≈ [E_emb,o − BM_CBAM · (1 − CF_y)] × P_CBAM,q

Key insight carried by this module: in 2026 an importer already pays CBAM on embedded emissions
above the free-allocation benchmark.

Freight: it is always an explicit input
results report the policy wedge (τ + T + CBAM) separately so conclusions never hinge on F_o.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from nsqueeze.params import ParameterSet
from nsqueeze.policy import PolicySchedule
from nsqueeze.provenance import Estimate, Provenance

# Pure functions


def cbam_charge(
    embedded: float,
    *,
    benchmark: float,
    cbam_factor: float,
    price: float,
    markup: float = 0.0,
    cscf: float = 1.0,
) -> float:
    """CBAM €/t product. ``markup`` applies only to default (not verified actual) values.

    The free-allocation term is benchmark × (1 − phase-out share) × CSCF.
    """
    certificates = embedded * (1.0 + markup) - benchmark * (1.0 - cbam_factor) * cscf
    return max(certificates, 0.0) * price


def landed_price(fob: float, freight: float, *, ad_valorem: float, specific: float, cbam: float):
    """P_imp in the currency of the inputs (convert FOB/freight to € first)."""
    return (fob + freight) * (1.0 + ad_valorem) + specific + cbam


@dataclass(frozen=True)
class Wedge:
    """Landed price decomposed so the policy wedge is visible apart from freight."""

    fob: float
    freight: float
    ad_valorem_duty: float
    specific_duty: float
    cbam: float

    @property
    def policy_wedge(self) -> float:
        return self.ad_valorem_duty + self.specific_duty + self.cbam

    @property
    def landed(self) -> float:
        return self.fob + self.freight + self.policy_wedge


# ── Origins ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Origin:
    code: str  # ISO-2, or "generic"
    duty_group: str  # MFN | PREF | RU_BY  (matches policy_schedule scope prefix)


ORIGINS = {
    "DZ": Origin("DZ", "PREF"),
    "TT": Origin("TT", "PREF"),
    "EG": Origin("EG", "PREF"),
    "RU": Origin("RU", "RU_BY"),
    "SA": Origin("SA", "MFN"),
    "US": Origin("US", "MFN"),
}


class ImportParity:
    def __init__(self, params: ParameterSet | None = None, policy: PolicySchedule | None = None):
        self.P = params or ParameterSet.load()
        self.policy = policy or PolicySchedule.load()

    def cbam(
        self,
        origin: str,
        on: dt.date | str,
        *,
        product_cn: str = "2814",
        cbam_price: float | None = None,
        cscf: float | None = None,
        embedded: float | None = None,
    ) -> Estimate[float]:
        """CBAM €/t product for an ``origin`` cargo of ammonia (2814) or urea (3102.10).

        Default values (plus mark-up) unless verified ``embedded`` emissions are given.
        ``cbam_price`` and ``cscf`` default to the rules in force on that date.
        """
        urea = product_cn.startswith("3102")
        cf = self.policy.value("cbam_factor", "EU", on)
        bm_p = self.P["cbam.benchmark_urea" if urea else "cbam.benchmark_ammonia"]
        prov = cf.provenance | Provenance.of(bm_p.ref)
        if cscf is None:
            c = self.policy.value("cscf", "EU", on)
            cscf, prov = c.value, prov | c.provenance
        if embedded is None:
            # An origin without its own row takes "Other countries and territories".
            table = "cbam.default_embedded_emissions" + ("_urea" if urea else "")
            key = f"{table}.{origin}"
            if key not in self.P:
                key = f"{table}.generic"
            emb_p, mk_p = self.P[key], self.P["cbam.default_markup_fertilisers"]
            embedded, markup = float(emb_p.value), float(mk_p.value)
            prov |= Provenance.of(emb_p.ref, mk_p.ref)
        else:
            markup = 0.0
        if cbam_price is None:
            price = self.policy.value("cbam_price", "EU", on)
            cbam_price, prov = price.value, prov | price.provenance
        value = cbam_charge(
            embedded,
            benchmark=float(bm_p.value),
            cbam_factor=cf.value,
            price=cbam_price,
            markup=markup,
            cscf=cscf,
        )
        return Estimate(value, prov)

    def wedge(
        self,
        origin: str,
        on: dt.date | str,
        *,
        fob_eur: float,
        freight_eur: float,
        product_cn: str = "2814",
        cbam_price: float | None = None,
    ) -> Estimate[Wedge]:
        """Full landed-price decomposition for an ammonia (CN 2814) or urea (3102.10) cargo."""
        group = ORIGINS[origin].duty_group
        tau = self.policy.value("duty_ad_valorem", f"{group}:{product_cn}", on)
        prov = tau.provenance
        specific = 0.0
        if group == "RU_BY":
            hs4 = product_cn.split(".")[0][:4]
            t_spec = self.policy.value("duty_specific", f"RU_BY:{hs4}", on)
            specific, prov = t_spec.value, prov | t_spec.provenance
        cb = self.cbam(origin, on, product_cn=product_cn, cbam_price=cbam_price)
        w = Wedge(
            fob=fob_eur,
            freight=freight_eur,
            ad_valorem_duty=(fob_eur + freight_eur) * tau.value,
            specific_duty=specific,
            cbam=cb.value,
        )
        return Estimate(w, prov | cb.provenance)
