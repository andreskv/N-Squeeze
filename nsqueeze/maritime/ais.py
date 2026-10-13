"""Live AIS archiving areas. The websocket client is ``archivers/ais_stream.py``.

Bounding boxes are deliberately coarse [[lat_min, lon_min], [lat_max, lon_max]] and
must be refined to terminal polygons once the GEM / GIE terminal lists are loaded.
Vessels en route with declared destinations and ETAs are the only real source of
multi-day LNG lead time, and that history cannot be recovered later.
"""

from __future__ import annotations

BOXES: dict[str, list[list[float]]] = {
    # Approximate regions — refine
    "hormuz_approaches": [[24.0, 54.0], [27.5, 58.5]],
    "nw_europe_lng": [[49.0, -6.0], [54.5, 9.5]],  # Channel, Benelux, German Bight
    "iberia_lng": [[36.0, -10.0], [44.0, 3.5]],
    "italy_lng": [[38.0, 8.0], [46.0, 14.0]],
    "baltic_lng": [[53.5, 12.0], [60.0, 25.0]],
}

# AIS ship-type codes of interest (ITU-R M.1371): 80–89 tankers. Gas carriers are a
# subset; filter further by dimensions/name lists, as type codes are often unreliable.
TANKER_TYPES = range(80, 90)
