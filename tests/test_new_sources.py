"""Parsers for the data-archive fetchers added on 2026-10-12 (no network)."""

import json

from nsqueeze.io.sources import REGISTRY, eurostat, gie, openmeteo, portwatch
from nsqueeze.io.sources import entsog as entsog_src


def test_jsonstat_row_major_decoding():
    body = {
        "id": ["geo", "time"],
        "size": [2, 3],
        "dimension": {
            "geo": {"category": {"index": {"DE": 0, "FR": 1}}},
            "time": {"category": {"index": {"2024-01": 0, "2024-02": 1, "2024-03": 2}}},
        },
        "value": {"0": 10, "4": 21},  # sparse: DE-2024-01 and FR-2024-02
    }
    df = eurostat.parse_jsonstat(body).set_index(["geo", "time"]).value
    assert df[("DE", "2024-01")] == 10 and df[("FR", "2024-02")] == 21 and len(df) == 2


def test_alsi_flattens_nested_inventory():
    row = {
        "gasDayStart": "2025-01-01",
        "status": "C",
        "updatedAt": "x",
        "inventory": {"lng": "123.4", "gwh": "800"},
        "sendOut": "55.1",
        "dtmi": {"lng": "-", "gwh": None},
        "dtrs": "",
    }
    out = gie.rows_to_canonical([row], "ALSI", "be", gie.ALSI_FIELDS)
    ids = {r["series_id"]: r["value"] for r in out}
    assert ids == {
        "ALSI.BE.inventory.lng": "123.4",
        "ALSI.BE.inventory.gwh": "800",
        "ALSI.BE.sendOut": "55.1",
    }


def test_entsog_gas_day_and_series_id():
    rows = [
        {
            "operatorKey": "FI-TSO-0003",
            "pointKey": "LNG-00072",
            "directionKey": "entry",
            "periodFrom": "2026-10-01T06:00:00+02:00",
            "value": 0,
            "unit": "kWh/d",
            "pointType": "LNG Entry point",
        },
        {"operatorKey": "X", "pointKey": None, "directionKey": "entry", "value": 1},
    ]
    df = entsog_src.to_canonical(rows)
    assert len(df) == 1
    assert df.series_id[0] == "ENTSOG.FI-TSO-0003.LNG-00072.entry"
    assert df.observed_at[0] == "2026-10-01"
    assert json.loads(df.meta[0])["pointType"] == "LNG Entry point"


def test_entsog_keeps_the_confirmed_value_of_a_day():
    day = {
        "operatorKey": "UK-TSO-0001",
        "pointKey": "ITP-00022",
        "directionKey": "entry",
        "periodFrom": "2021-05-28T06:00:00+02:00",
        "unit": "kWh/d",
    }
    rows = [
        day | {"value": 1, "flowStatus": "Provisional"},
        day | {"value": 2, "flowStatus": "Confirmed"},
        day | {"value": 2, "flowStatus": "Confirmed"},
    ]
    df = entsog_src.to_canonical(rows)
    assert df.value.tolist() == [2]


def test_portwatch_and_openmeteo_canonical():
    feats = [
        {
            "attributes": {
                "date": "2019-01-01",
                "portid": "chokepoint6",
                "portname": "Strait of Hormuz",
                "n_total": 52,
                "n_tanker": 33,
                "capacity": 2795480,
            }
        }
    ]
    pw = portwatch.to_canonical(feats).set_index("series_id")
    assert pw.loc["PW.chokepoint6.n_tanker", "value"] == 33
    assert pw.loc["PW.chokepoint6.capacity", "unit"] == "dwt"

    payload = [
        {
            "daily": {
                "time": ["1990-01-01"],
                "temperature_2m_max": [1.0],
                "temperature_2m_min": [-3.0],
                "temperature_2m_mean": [None],
            }
        }
    ]
    om = openmeteo.to_canonical(payload, ["AT"])
    assert set(om.series_id) == {"OM.AT.temperature_2m_max", "OM.AT.temperature_2m_min"}


def test_weather_points_cover_eu27():
    pts = openmeteo.weather_points()
    assert set(pts.geo) == set(eurostat.EU27_GEO) and pts.grade.eq("C").all()


def test_registry_has_no_stub_for_key_free_phase0_sources():
    from nsqueeze.io.sources.base import NotYetImplemented

    for name in ("entsog", "portwatch", "comext", "weather", "alsi"):
        assert not issubclass(REGISTRY[name], NotYetImplemented), name


def test_entsoe_gas_generation_fills_a03_gaps_and_skips_consumption():
    from nsqueeze.io.sources import entsoe

    ns = "urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0"

    def series(domain, psr, points):
        pts = "".join(
            f"<Point><position>{p}</position><quantity>{q}</quantity></Point>" for p, q in points
        )
        return (
            f"<TimeSeries><{domain} codingScheme='A01'>10Y1001A1001A83F</{domain}>"
            "<quantity_Measure_Unit.name>MAW</quantity_Measure_Unit.name>"
            f"<MktPSRType><psrType>{psr}</psrType></MktPSRType><Period><timeInterval>"
            "<start>2023-08-15T22:00Z</start><end>2023-08-15T23:00Z</end></timeInterval>"
            f"<resolution>PT15M</resolution>{pts}</Period></TimeSeries>"
        )

    xml = (
        f"<GL_MarketDocument xmlns='{ns}'>"
        + series("inBiddingZone_Domain.mRID", "B04", [(1, 100), (3, 130)])
        + series("outBiddingZone_Domain.mRID", "B04", [(1, 5)])
        + series("inBiddingZone_Domain.mRID", "B01", [(1, 9)])
        + "</GL_MarketDocument>"
    ).encode()
    df = entsoe.to_canonical(xml, "DE")
    assert list(df.value) == [100, 100, 130, 130]
    assert df.observed_at[0] == __import__("pandas").Timestamp("2023-08-15 22:00")
    assert set(df.series_id) == {"ENTSOE.DE.B04"} and set(df.unit) == {"MW"}


def test_entsoe_acknowledgement_is_empty():
    from nsqueeze.io.sources import entsoe

    xml = (
        b"<Acknowledgement_MarketDocument><Reason><code>999</code>"
        b"<text>No matching data found</text></Reason></Acknowledgement_MarketDocument>"
    )
    assert entsoe.to_canonical(xml, "MT").empty
    assert len(entsoe.areas()) == 27


def test_entsog_archive_value_decoding():
    assert entsog_src.parse_archive_value("93.698.000.000.000") == 93_698_000.0
    assert entsog_src.parse_archive_value("106.978.050.258") == 106_978.050258
    assert entsog_src.parse_archive_value("0.000000") == 0.0
    assert entsog_src.parse_archive_value("695445.00") == 695_445.0
    assert entsog_src.parse_archive_value("") is None


def test_entsog_archive_rows(tmp_path):
    head = "Operator Name;Operator EIC;Point name;Point EIC;Direction;Indicator;"
    head += "Period From;Period To;Value;Unit\n"
    day, nxt, hour = "2016-03-01 05:00:00.000", "2016-03-02 05:00:00.000", "2016-03-01 06:00:00.000"
    rows = [
        f"Op;OPEIC;Mapped;21ZA;entry;PhysicalFlow;{day};{nxt};1.000.000.000;kWh/d",
        f"Op;OPEIC;Mapped;21ZA;entry;PhysicalFlow;{day};{hour};9.000000;kWh/d",  # hourly
        f"Op;OPEIC;Greifswald;?;entry;PhysicalFlow;{day};{nxt};2.000000;kWh/d",
        f"Op;OPEIC;Unknown;?;exit;PhysicalFlow;{day};{nxt};3.000000;kWh/d",
        f"Op;OPEIC;Twice;21ZB;exit;PhysicalFlow;{day};{nxt};4.000000;kWh/d",
        "Op;OPEIC;Twice;21ZB;exit;PhysicalFlow;"
        "2016-03-01 07:00:00.000;2016-03-02 07:00:00.000;5.000000;kWh/d",
    ]
    f = tmp_path / "AllTSOs_CalendarYear2016_PhysicalFlow.csv"
    f.write_text(head + "\n".join(rows), encoding="utf-8")
    api = {
        "operationaldatas": [
            {"tsoEicCode": "OPEIC", "tsoItemIdentifier": "21ZA", "directionKey": "entry",
             "operatorKey": "DE-TSO-1", "pointKey": "ITP-1", "pointLabel": "Mapped"},
            {"tsoEicCode": "OPEIC", "tsoItemIdentifier": "21ZG", "directionKey": "entry",
             "operatorKey": "DE-TSO-1", "pointKey": "ITP-2", "pointLabel": "Greifswald"},
        ]
    }  # fmt: skip
    df = entsog_src.archive_to_canonical(f, entsog_src.api_keymap([api]))
    got = dict(zip(df.series_id, df.value, strict=True))
    assert got == {
        "ENTSOG.DE-TSO-1.ITP-1.entry": 1000.0,
        "ENTSOG.DE-TSO-1.ITP-2.entry": 2.0,
        "ENTSOG.NAME.OPEIC.Unknown.exit": 3.0,
    }  # the two conflicting "Twice" rows are dropped
    assert set(df.observed_at) == {"2016-03-01"}


def test_ets_verified_emissions_parse():
    import io

    import pandas as pd

    from nsqueeze.io.sources import ets

    head = [
        "REGISTRY_CODE",
        "IDENTIFIER_IN_REG",
        "INSTALLATION_NAME",
        "INSTALLATION_IDENTIFIER",
        "PERMIT_IDENTIFIER",
        "MAIN_ACTIVITY_TYPE_CODE",
        "ALLOCATION_2025",
        "VERIFIED_EMISSIONS_2025",
        "VERIFIED_EMISSIONS_2024",
        "ACCOUNT_TYPE",
    ]
    rows = [
        ["AT", "x", "Ammoniakanlage", 205363, "P1", 41, 650000, 700000, "Excluded",
         "OPERATOR HOLDING ACCOUNT"],
        ["DE", "y", "Kraftwerk", 7, "P2", 20, None, None, 1200, "OPERATOR HOLDING ACCOUNT"],
        ["FR", "z", "Airline", 9, "", 10, 0, 50, 60, "AIRCRAFT OPERATOR HOLDING ACCOUNT"],
    ]  # fmt: skip
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as w:
        pd.DataFrame([["Date of Extraction", pd.Timestamp("2026-04-01")]]).to_excel(
            w, sheet_name="Read Me", header=False, index=False
        )
        pd.DataFrame([["Please see Read Me"], [None], head, *rows]).to_excel(
            w, sheet_name="data", header=False, index=False
        )
    df = ets.to_canonical(buf.getvalue())
    got = {(s, d.year): v for s, d, v in zip(df.series_id, df.observed_at, df.value, strict=True)}
    assert got == {  # no aircraft; a blank allocation is no row
        ("ETS.AT.205363", 2025): 700000,
        ("ETS.DE.7", 2024): 1200,
        ("ETSALLOC.AT.205363", 2025): 650000,
    }
    assert set(df[df.series_id.str.startswith("ETSALLOC")].unit) == {"EUA"}
    assert json.loads(df.meta[0]) == {
        "name": "Ammoniakanlage", "activity": 41, "permit": "P1", "extracted": "2026-04-01"
    }  # fmt: skip
    html = (
        '<a href="/document/download/a_en?filename=verified_emissions_2024_en.xlsx">'
        '<a href="/document/download/b_en?filename=verified_emissions_2025_en.xlsx">'
    )
    assert ets.latest_link(html) == (
        "/document/download/b_en?filename=verified_emissions_2025_en.xlsx",
        2025,
    )


def test_yahoo_ttf_dates_in_exchange_time_zone():
    from nsqueeze.io.sources.yahoo import chart_to_canonical

    body = {
        "chart": {
            "result": [
                {
                    "meta": {"currency": "EUR", "exchangeName": "NYM", "gmtoffset": -18000},
                    # 2025-12-01 05:00 and 2025-12-02 05:00 UTC = 00:00 New York; one null close
                    "timestamp": [1764565200, 1764651600, 1764738000],
                    "indicators": {"quote": [{"close": [28.2, None, 28.0]}]},
                }
            ]
        }
    }
    df = chart_to_canonical(body)
    assert df.observed_at.dt.strftime("%Y-%m-%d").tolist() == ["2025-12-01", "2025-12-03"]
    assert df.unit.unique().tolist() == ["EUR/MWh"] and df.series_id.iloc[0] == "YF.TTF=F"


def test_yahoo_equity_uses_adjusted_close():
    from nsqueeze.io.sources.yahoo import adjclose_to_canonical

    body = {
        "chart": {
            "result": [
                {
                    "meta": {"currency": "NOK", "exchangeName": "OSL", "gmtoffset": 3600},
                    "timestamp": [1764576000, 1764662400],
                    "indicators": {
                        "quote": [{"close": [100.0, 101.0]}],
                        "adjclose": [{"adjclose": [95.0, None]}],
                    },
                }
            ]
        }
    }
    df = adjclose_to_canonical(body, "YAR.OL")
    assert df.value.tolist() == [95.0] and df.series_id.iloc[0] == "YF.YAR.OL.adjclose"
    assert df.unit.iloc[0] == "NOK"


def test_ecb_nok_requests_the_nok_series():
    import datetime as dt

    import httpx

    from nsqueeze.io.sources.ecb import ECBFxNOK

    seen = []

    def handler(req):
        seen.append(str(req.url))
        return httpx.Response(200, text="TIME_PERIOD,OBS_VALUE\n2026-10-16,11.62\n")

    src = ECBFxNOK(client=httpx.Client(transport=httpx.MockTransport(handler)))
    src.keep_raw = lambda *a: None
    df = src.fetch(dt.date(2026, 10, 16), dt.date(2026, 10, 16))
    assert "/EXR/D.NOK.EUR.SP00.A" in seen[0]
    assert df.series_id.iloc[0] == "EXR.D.NOK.EUR.SP00.A" and df.unit.iloc[0] == "NOK/EUR"


def test_sentinel2_scene_features_to_rows():
    from nsqueeze.io.sources.sentinel2 import STATS, to_canonical

    props = {s: 0.5 for s in STATS} | {
        "time": 1659524220339,
        "product_id": "S2B_X",
        "b12_p99": None,
    }
    df = to_canonical([{"properties": props}], "NL-SLUISK")
    assert len(df) == len(STATS) - 1  # a missing statistic (cloud-free box empty) is left out
    assert df.series_id.str.startswith("S2.NL-SLUISK.").all()
    assert str(df.observed_at.iloc[0]) == "2022-08-03 10:57:00.339000"


def test_sentinel2_overlapping_tiles_keep_the_fuller_one():
    from nsqueeze.io.sources.sentinel2 import STATS, to_canonical

    base = {s: 0.5 for s in STATS} | {"time": 1659524220339}
    edge = base | {"product_id": "EDGE", "n_px": 40.0}
    full = base | {"product_id": "FULL", "n_px": 700.0}
    df = to_canonical([{"properties": full}, {"properties": edge}], "NL-SLUISK")
    assert len(df) == len(STATS) and not df.duplicated(["series_id", "observed_at"]).any()
    assert df.meta.str.contains("FULL").all()


def test_landsat_scene_features_to_rows():
    from nsqueeze.io.sources.landsat import STATS, to_canonical

    props = {s: 300.0 for s in STATS} | {
        "time": 1659524220339,
        "scene_id": "LC09_X",
        "st_core_mean": None,
    }
    df = to_canonical([{"properties": props}], "NL-SLUISK")
    assert len(df) == len(STATS) - 1  # no clear pixel in the disc: the mean is left out
    assert df.series_id.str.startswith("LS.NL-SLUISK.").all()
    assert set(df.unit) == {"K", "share"}


def test_plume_scene_features_to_rows():
    from nsqueeze.io.sources.sentinel2 import PLUME_STATS, plume_to_canonical

    props = {s: 0.2 for s in PLUME_STATS} | {"time": 1659524220339, "core_px": None}
    df = plume_to_canonical([{"properties": props}], "NL-SLUISK")
    assert len(df) == len(PLUME_STATS) - 1
    assert df.series_id.str.startswith("S2P.NL-SLUISK.").all()


def test_plant_weather_points_are_the_plants():
    from nsqueeze.io.sources.openmeteo import OpenMeteoPlantWeather

    pts = OpenMeteoPlantWeather().points()
    assert list(pts.columns) == ["geo", "lat", "lon"] and pts.geo.str.contains("-").all()


def test_firms_keeps_detections_near_plants_only():
    import pandas as pd

    from nsqueeze.io.sources.firms import near_plants

    plants = pd.DataFrame({"plant_id": ["P"], "lat": [50.0], "lon": [10.0]})
    det = pd.DataFrame(
        {
            "latitude": [50.0045, 51.0],  # about 0.5 km and 111 km north
            "longitude": [10.0, 10.0],
            "acq_date": ["2022-01-10", "2022-01-10"],
            "acq_time": [104, 104],
            "daynight": ["N", "N"],
            "type": [2, 2],
            "confidence": ["n", "n"],
            "frp": [1.2, 3.4],
        }
    )
    out = near_plants(det, plants, "VIIRS_SNPP_SP")
    assert len(out) == 1 and abs(out.value.iloc[0] - 0.5) < 0.01
    assert str(out.observed_at.iloc[0]) == "2022-01-10 01:04:00"


def test_firms_one_row_per_overpass_keeps_nearest():
    import json

    import pandas as pd

    from nsqueeze.io.sources.firms import near_plants

    plants = pd.DataFrame({"plant_id": ["P"], "lat": [50.0], "lon": [10.0]})
    det = pd.DataFrame(
        {
            "latitude": [50.027, 50.0045],  # about 3 km and 0.5 km, same overpass
            "longitude": [10.0, 10.0],
            "acq_date": ["2022-01-10"] * 2,
            "acq_time": [104, 104],
            "daynight": ["N", "N"],
            "frp": [2.0, 1.0],
        }
    )
    out = near_plants(det, plants, "VIIRS_SNPP_SP")
    assert len(out) == 1 and abs(out.value.iloc[0] - 0.5) < 0.01
    meta = json.loads(out.meta.iloc[0])
    assert (meta["n_5km"], meta["n_1km"], meta["daynight"]) == (2, 1, "N")


def test_plume_keeps_the_best_covering_tile():
    from nsqueeze.io.sources.sentinel2 import plume_to_canonical

    base = {"time": 1659524220339, "core_cloud_frac": 0.0, "ring_cloud_frac": 0.0}
    partial = base | {"core_px": 100.0, "ring_px": 900.0, "core_cloud_frac": 0.9}
    full = base | {"core_px": 1942.0, "ring_px": 52446.0, "core_cloud_frac": 0.1}
    df = plume_to_canonical([{"properties": partial}, {"properties": full}], "P")
    core = df[df.series_id == "S2P.P.core_cloud_frac"]
    assert len(core) == 1 and core.value.iloc[0] == 0.1


def test_s5p_week_rows_one_per_plant_and_area():
    import pandas as pd

    from nsqueeze.io.sources.sentinel5p import week_to_canonical

    feats = [
        {"properties": {"plant_id": "RO-TARGUM", "area": "disc", "no2": 6e-5, "n_obs": 4.3}},
        {"properties": {"plant_id": "RO-TARGUM", "area": "ring", "no2": 3e-5, "n_obs": 4.1}},
        {"properties": {"plant_id": "HR-KUTINA", "area": "disc", "no2": None, "n_obs": 0}},
    ]
    df = week_to_canonical(feats, pd.Timestamp("2022-01-03"))
    assert df.series_id.tolist() == ["S5P.RO-TARGUM.disc", "S5P.RO-TARGUM.ring"]
    assert (df.observed_at == pd.Timestamp("2022-01-03")).all()
    assert json.loads(df.meta[0])["n_obs"] == 4.3


def test_gfw_presence_rows_one_per_vessel_and_month():
    import pandas as pd

    from nsqueeze.io.sources.gfw import to_canonical, zones

    rows = [
        {
            "vesselId": "v1",
            "date": "2022-04",
            "hours": 31,
            "imo": "9734836",
            "shipName": "YARA KARA",
            "geartype": "TANKER",
            "speed": 0.08,
        },
        {"vesselId": "v1", "date": "2022-05", "hours": 12, "imo": "9734836"},
        {"vesselId": None, "date": "2022-05", "hours": 3},
    ]
    df = to_canonical(rows, "NL-SLUISK")
    assert df.series_id.tolist() == ["GFWP.NL-SLUISK.v1"] * 2
    assert df.observed_at.tolist() == [pd.Timestamp("2022-04-01"), pd.Timestamp("2022-05-01")]
    assert json.loads(df.meta[0])["shipName"] == "YARA KARA"
    assert {z["properties"]["plant_id"] for z in zones()} >= {"NL-SLUISK", "PL-POLICE"}


def test_gfw_presence_fetch_starts_on_the_first_of_the_month(monkeypatch):
    import datetime as dt

    from nsqueeze.io.sources.gfw import GFWPresence

    seen = []
    src = GFWPresence()
    monkeypatch.setattr(src, "_report", lambda f, start, end: seen.append(start) or [])
    src.fetch(dt.date(2023, 1, 4), dt.date(2024, 1, 4))
    assert set(seen) == {dt.date(2023, 1, 1)}


def test_vnf_night_parsing_and_near_plants():
    import gzip

    import pandas as pd

    from nsqueeze.io.sources import nightfire

    csv = (
        "id,Lat_GMTCO,Lon_GMTCO,Date_Mscan,Temp_BB,unused\n"
        "1,51.312,3.830,2026/10/01 00:41:35.597,1459,9\n"  # plant A, one overpass
        "2,51.313,3.831,2026/10/01 00:41:35.900,999999,9\n"  # plant A, same overpass
        "3,52.671,4.759,2026/10/01 01:10:00.000,999999,9\n"  # plant B, later overpass
        "4,40.000,-3.000,2026/10/01 00:41:35.000,999999,9\n"  # far from every plant
    )
    det = nightfire.parse_night(gzip.compress(csv.encode()))
    assert det.temp_k.iloc[0] == 1459.0 and det.temp_k.isna().sum() == 3

    plants = pd.DataFrame({"plant_id": ["A", "B"], "lat": [51.310, 52.670], "lon": [3.830, 4.760]})
    out = nightfire.near_plants(det, plants, "v30")
    assert out.series_id.tolist() == ["VNF.A", "VNF.B"]  # one row per plant and overpass
    assert (out.unit == "km").all() and (out.value < 1.0).all()
    meta_a = json.loads(out.meta.iloc[0])
    assert meta_a == {"n_5km": 2, "n_1km": 2, "max_temp_k": 1459.0, "product": "v30"}
    meta_b = json.loads(out.meta.iloc[1])
    assert meta_b["n_5km"] == 1 and meta_b["max_temp_k"] is None
    assert out.observed_at.tolist() == [
        pd.Timestamp("2026-10-01 00:41"),
        pd.Timestamp("2026-10-01 01:10"),
    ]


def test_vnf_login_accepts_a_live_keycloak_session(monkeypatch):
    import httpx
    import pytest

    from nsqueeze.io.sources.nightfire import NightfireVIIRS

    monkeypatch.setenv("EOG_USER", "u")
    monkeypatch.setenv("EOG_PASSWORD", "p")
    no_form = httpx.MockTransport(lambda req: httpx.Response(200, text="<html>VNF</html>"))
    src = NightfireVIIRS(httpx.Client(transport=no_form))
    with pytest.raises(ValueError, match="no Keycloak login form"):
        src._login()
    src.client.cookies.set("KEYCLOAK_IDENTITY", "x")
    src._login()  # second login of a back-fill: the session redirects past the form


def test_vnf_night_urls_by_product_era():
    import datetime as dt

    from nsqueeze.io.sources.nightfire import NightfireVIIRS

    src = NightfireVIIRS()
    v21 = src._night_urls(dt.date(2017, 11, 30))
    assert v21 == [
        (
            "https://eogdata.mines.edu/wwwdata/viirs_products/vnf/v21/VNF_npp_d20171130_noaa_v21.csv.gz",
            "v21",
        )
    ]
    v30 = src._night_urls(dt.date(2017, 12, 1))
    assert v30[0][1] == "v30" and v30[0][0].endswith(
        "/v30/rearrange/2017/12/npp/VNF_npp_d20171201_noaa_v30.csv.gz"
    )
    assert v30[1][1] == "v30-nrp" and v30[1][0].endswith(
        "/v30_nrp/VNF_npp_d20171201_eog_nrp_v30.csv.gz"
    )
