"""Registry of archivable sources. Names match ``config/sources.yaml``."""

from __future__ import annotations

from nsqueeze.io.sources.base import NotYetImplemented, Source
from nsqueeze.io.sources.ecb import ECBFx
from nsqueeze.io.sources.eex import EUAAuctions
from nsqueeze.io.sources.eia import HenryHub
from nsqueeze.io.sources.entsoe import ENTSOEGasGeneration
from nsqueeze.io.sources.entsog import ENTSOGFlows
from nsqueeze.io.sources.eurostat import Comext, EurostatHDD, EurostatPopulation
from nsqueeze.io.sources.gie import AGSI, ALSI
from nsqueeze.io.sources.openmeteo import OpenMeteoWeather
from nsqueeze.io.sources.portwatch import PortWatch
from nsqueeze.io.sources.worldbank import PinkSheetMonthly


def _stub(source_name: str, phase: int) -> type[Source]:
    return type(f"Stub_{source_name}", (NotYetImplemented,), {"name": source_name, "phase": phase})


REGISTRY: dict[str, type[Source]] = {
    s.name: s
    for s in (
        ECBFx,
        HenryHub,
        AGSI,
        ALSI,
        ENTSOGFlows,
        ENTSOEGasGeneration,
        PinkSheetMonthly,
        EUAAuctions,
        Comext,
        EurostatPopulation,
        EurostatHDD,
        OpenMeteoWeather,
        PortWatch,
        _stub("ttf_daily_candidate", 0),
        _stub("wto_hormuz", 0),
        _stub("ets_registry", 1),
    )
}


def get(name: str) -> Source:
    try:
        return REGISTRY[name]()
    except KeyError:
        raise KeyError(f"Unknown source {name!r}. Known: {', '.join(sorted(REGISTRY))}") from None
