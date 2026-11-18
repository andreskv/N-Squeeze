"""Registry of archivable sources. Names match ``config/sources.yaml``."""

from __future__ import annotations

from nsqueeze.io.sources.base import NotYetImplemented, Source
from nsqueeze.io.sources.company_news import BASFNews, OCINews
from nsqueeze.io.sources.ecb import ECBFx, ECBFxNOK, ECBFxPLN
from nsqueeze.io.sources.eex import EUAAuctions
from nsqueeze.io.sources.eia import HenryHub
from nsqueeze.io.sources.emodnet import EmodnetVesselDensity
from nsqueeze.io.sources.entsoe import ENTSOEGasGeneration
from nsqueeze.io.sources.entsog import ENTSOGFlows
from nsqueeze.io.sources.ets import ETSVerifiedEmissions
from nsqueeze.io.sources.eurostat import (
    Comext,
    ComextReporter,
    EurostatFertiliserIndex,
    EurostatGasTrade,
    EurostatHDD,
    EurostatHDDNuts2,
    EurostatPopulation,
    EurostatPopulationNuts2,
)
from nsqueeze.io.sources.firms import FirmsVIIRS
from nsqueeze.io.sources.gfw import GFWPresence
from nsqueeze.io.sources.gie import AGSI, ALSI
from nsqueeze.io.sources.grains import GrainFutures
from nsqueeze.io.sources.gus import GUSProduction
from nsqueeze.io.sources.landsat import LandsatThermal
from nsqueeze.io.sources.lseg import LSEGVessels
from nsqueeze.io.sources.lseg_ports import LSEGPortCalls
from nsqueeze.io.sources.lseg_series import LSEGSeries
from nsqueeze.io.sources.newsweb import NewswebFilings
from nsqueeze.io.sources.nightfire import NightfireVIIRS
from nsqueeze.io.sources.openmeteo import OpenMeteoPlantWeather, OpenMeteoWeather
from nsqueeze.io.sources.portwatch import PortWatch
from nsqueeze.io.sources.prodcom import EurostatProdcom
from nsqueeze.io.sources.sentinel2 import Sentinel2Plume, Sentinel2SWIR
from nsqueeze.io.sources.sentinel5p import Sentinel5PNO2
from nsqueeze.io.sources.the import THEConsumption
from nsqueeze.io.sources.worldbank import PinkSheetMonthly
from nsqueeze.io.sources.yahoo import YahooEquities, YahooTTF


def _stub(source_name: str, phase: int) -> type[Source]:
    return type(f"Stub_{source_name}", (NotYetImplemented,), {"name": source_name, "phase": phase})


REGISTRY: dict[str, type[Source]] = {
    s.name: s
    for s in (
        ECBFx,
        ECBFxNOK,
        ECBFxPLN,
        HenryHub,
        AGSI,
        ALSI,
        ENTSOGFlows,
        ENTSOEGasGeneration,
        PinkSheetMonthly,
        EUAAuctions,
        Comext,
        ComextReporter,
        EurostatPopulation,
        EurostatHDD,
        EurostatHDDNuts2,
        EurostatPopulationNuts2,
        EurostatGasTrade,
        EurostatFertiliserIndex,
        EurostatProdcom,
        OpenMeteoWeather,
        PortWatch,
        Sentinel2SWIR,
        LandsatThermal,
        Sentinel2Plume,
        FirmsVIIRS,
        NightfireVIIRS,
        Sentinel5PNO2,
        GFWPresence,
        EmodnetVesselDensity,
        OpenMeteoPlantWeather,
        YahooTTF,
        YahooEquities,
        GrainFutures,
        LSEGVessels,
        LSEGPortCalls,
        LSEGSeries,
        NewswebFilings,
        OCINews,
        BASFNews,
        GUSProduction,
        THEConsumption,
        _stub("wto_hormuz", 0),
        ETSVerifiedEmissions,
    )
}


def get(name: str) -> Source:
    try:
        return REGISTRY[name]()
    except KeyError:
        raise KeyError(f"Unknown source {name!r}. Known: {', '.join(sorted(REGISTRY))}") from None
