"""The EMODnet vessel-density fetcher: pure functions and row assembly (no network)."""

import datetime as dt
import json
from pathlib import Path

import httpx
import numpy as np

from nsqueeze.io.sources import emodnet

SAMPLE = Path(__file__).parent / "fixtures" / "emodnet_getfeatureinfo.json"


def test_point_in_polygon_square():
    ring = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    points = np.array([[0.5, 0.5], [1.5, 0.5], [0.5, 1.5], [-0.1, 0.5]])
    assert emodnet.point_in_polygon(points, ring).tolist() == [True, False, False, False]


def test_point_in_polygon_concave_ring():
    # L-shape: the notch at (2, 2) is outside although it sits inside the bounding box.
    ring = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 1.0], [1.0, 1.0], [1.0, 3.0], [0.0, 3.0]])
    points = np.array([[0.5, 2.5], [2.0, 2.0], [2.5, 0.5], [3.5, 0.5]])
    assert emodnet.point_in_polygon(points, ring).tolist() == [True, False, True, False]


def test_point_in_polygon_closed_ring():
    ring = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
    points = np.array([[0.5, 0.5], [2.0, 2.0]])
    assert emodnet.point_in_polygon(points, ring).tolist() == [True, False]


def test_cell_centres_are_lattice_midpoints():
    ring = np.array([[0.034, 0.024], [0.046, 0.024], [0.046, 0.036], [0.034, 0.036]])
    centres = emodnet.cell_centres(ring)
    assert centres.shape == (4, 2)  # two lattice cells across and up
    assert np.allclose(centres % emodnet.GRID_DEG, emodnet.HALF_DEG)
    assert emodnet.point_in_polygon(centres, ring).all()


def test_parse_value_and_no_data():
    assert emodnet.parse_value(SAMPLE.read_text()) == 158.33782958984375
    assert emodnet.parse_value('{"features":[{"properties":{"GRAY_INDEX":-9999.0}}]}') is None
    assert emodnet.parse_value('{"features":[{"properties":{"GRAY_INDEX":-3.4028235e38}}]}') is None
    assert emodnet.parse_value('{"features":[]}') is None


def test_fetch_one_row_per_zone_ship_type_and_month(monkeypatch):
    zone = {
        "type": "Feature",
        "properties": {"plant_id": "XX-TEST"},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[0.034, 0.024], [0.046, 0.024], [0.046, 0.036], [0.034, 0.036]]],
        },
    }
    monkeypatch.setattr(emodnet, "zones", lambda: [zone])
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        # the all-ships map marks missing cells with the lowest float32, the tanker map with -9999
        value = 2.5 if "vesseldensity_10" in str(request.url) else -3.4028235e38
        body = {"features": [{"type": "Feature", "properties": {"GRAY_INDEX": value}}]}
        return httpx.Response(200, json=body)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    frame = emodnet.EmodnetVesselDensity(client=client).fetch(
        dt.date(2022, 6, 1), dt.date(2022, 7, 31)
    )
    assert sorted(frame.series_id.unique()) == ["EMODNET.XX-TEST.all", "EMODNET.XX-TEST.tanker"]
    assert frame.observed_at.astype(str).tolist() == ["2022-06-01", "2022-07-01"] * 2
    assert (frame.value[frame.series_id.str.endswith("tanker")] == 2.5).all()
    assert frame.value[frame.series_id.str.endswith("all")].isna().all()  # no data stays missing
    assert json.loads(frame.meta[0])["cells"] == 4
    assert len(calls) == 16  # four lattice cells in the zone x two ship types x two months
