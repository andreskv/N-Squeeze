"""Sensor catalogue. Resolutions and revisits checked against mission documentation
(ESA, EUMETSAT, NASA/USGS) on 2026-10-15. ``prior`` is a judgement; measuring it is the point.

Learning order: Nightfire first (point CSVs, no rasters) → Sentinel-2 SWIR clipped to
small boxes around each plant → fusion model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Emission = Literal["bernoulli", "gaussian"]


@dataclass(frozen=True)
class Sensor:
    key: str
    name: str
    sees: str
    resolution: str
    prior: str
    emission: Emission
    order: int  # learning order; lower = earlier


SENSORS: tuple[Sensor, ...] = (
    Sensor(
        "viirs_nightfire",
        "VIIRS Nightfire",
        "Night-time high-temperature sources",
        "≈ 750 m",
        "medium",
        "bernoulli",
        1,
    ),
    Sensor(
        "s2_swir",
        "Sentinel-2 SWIR (B11/B12)",
        "Daytime hot spots",
        "20 m, ≈ 5-day revisit",
        "medium",
        "gaussian",
        2,
    ),
    Sensor(
        "s2_plume",
        "Sentinel-2 visible",
        "Condensation plumes (cold weather)",
        "10 m",
        "low–medium",
        "bernoulli",
        3,
    ),
    Sensor(
        "s5p_no2",
        "Sentinel-5P NO₂",
        "Combustion emissions",
        "3.5 × 5.5 km (along-track 7 km before 2019-08-06)",
        "low (single plants)",
        "gaussian",
        4,
    ),
    Sensor("iasi_nh3", "IASI NH₃", "Ammonia point sources", "≈ 12 km", "low–medium", "gaussian", 5),
    Sensor(
        "landsat_tirs",
        "Landsat 8/9 thermal",
        "Surface heat",
        "100 m (delivered at 30 m), 8-day combined revisit",
        "low–medium",
        "gaussian",
        6,
    ),
)
