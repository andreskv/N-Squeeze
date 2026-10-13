"""Parsers for the Phase 0 fetchers added on 2026-10-12 (no network)."""

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
