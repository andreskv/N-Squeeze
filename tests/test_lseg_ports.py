"""LSEG port calls: workbook layouts, hold-out masking and duplicate calls.

All vessels, places and events here are invented. The licensed export stays out of tests.
"""

import json

import pandas as pd
from openpyxl import Workbook

from nsqueeze.io import PITStore
from nsqueeze.io.sources.lseg import RETRIEVED_AT
from nsqueeze.io.sources.lseg_ports import LSEGPortCalls, read_port

DEP_COLUMNS = [
    "Vessel",
    "Type",
    "Sub-type",
    "Sub-sub-type",
    "IMO",
    "Last Seen",
    "Country/Region",
    "Polygon Type",
    "Polygon Name",
    "Departure Date (GMT)",
    "Draft (m)",
    "Berth Type",
    "Next Country/Region",
    "Next Location",
    "Next Location Arrival Date",
    "Next Location Draft (m) (Entering port)",
    "Next Location Draft (m) (Exiting port)",
    "Berth Type",
    "DWT",
    "Max Draft (m)",
    "Cubic Capacity (m3)",
    "Clean / Dirty (tankers only)",
]
ARR_COLUMNS = [
    "Vessel",
    "Type",
    "Sub-type",
    "Sub-sub-type",
    "IMO",
    "Last Seen",
    "Country/Region",
    "Port",
    "Arrival Date (GMT) (Entering port)",
    "Draft (m) (Entering port)",
    "Berth Type",
    "Previous Country/Region",
    "Previous Location",
    "Previous Location Departure Date (GMT)",
    "Previous Location Draft (m) (Entering Location)",
    "Previous Location Draft (m) (Exiting Location)",
    "Berth Type",
    "DWT",
    "Max Draft (m)",
    "Cubic Capacity (m3)",
    "Clean / Dirty (tankers only)",
]
STAY_COLUMNS = [
    "Vessel",
    "Type",
    "Sub-type",
    "Sub-sub-type",
    "IMO",
    "Last Seen",
    "Arrival Date (GMT) (Entering port)",
    "Departure Date (GMT)",
    "Destination (current)",
    "ETA (GMT)",
    "DWT",
    "Length (m)",
    "Beam (m)",
    "Max Draft (m)",
    "Clean / Dirty (tankers only)",
]


def test_departure_masks_2026_and_deduplicates_in_file_order(tmp_path):
    future_next = _record(
        IMO=9000001,
        **{
            "Polygon Name": "Sluiskil",
            "Departure Date (GMT)": "31/12/2025 10:00:00",
            "Draft (m)": 8.2,
            "Next Location Arrival Date": "03/01/2026 08:00:00",
        },
    )
    future_event = _record(
        IMO=9000002,
        **{
            "Polygon Name": "Terneuzen",
            "Departure Date (GMT)": "01/01/2026 10:00:00",
            "Draft (m)": 9.0,
        },
    )
    open_call = _record(
        IMO=9000003,
        **{
            "Polygon Name": "Terneuzen",
            "Departure Date (GMT)": "30/12/2025 10:00:00",
            "Draft (m)": 8.5,
            "Next Location Arrival Date": "31/12/2025 12:00:00",
        },
    )
    no_place = _record(
        IMO=9000004,
        **{
            "Polygon Name": " - ",
            "Departure Date (GMT)": "29/12/2025 10:00:00",
            "Draft (m)": 8.0,
        },
    )
    first_path = tmp_path / "p00_dep_test_2019-2025.xlsx"
    _workbook(
        first_path,
        DEP_COLUMNS,
        "Sluiskil, Terneuzen",
        [
            future_next,
            future_event,
            open_call,
            no_place,
        ],
        header_at=8,
    )
    _workbook(tmp_path / "p11_dep_test_2019-2025.xlsx", DEP_COLUMNS, "Sluiskil", [future_next])

    frame = LSEGPortCalls(folder=tmp_path).fetch(None, None)

    assert len(frame) == 2
    _, counts = read_port(first_path)
    assert counts["dropped_no_place"] == 1
    assert frame.series_id[0] == "IMO9000001.dep.sluiskil"
    first = json.loads(frame.meta[0])
    assert first["place"] == "Sluiskil" and first["file"] == "p00_dep_test_2019-2025"
    assert not any(key.startswith("next_") for key in first)
    assert json.loads(frame.meta[1])["next_location"] == "Next Port"
    forbidden = {"last_seen", "destination_current", "eta_gmt", "next_location_arrival_date"}
    assert all(forbidden.isdisjoint(json.loads(meta)) for meta in frame.meta)
    assert all("2026" not in meta for meta in frame.meta)
    assert not frame.duplicated(["series_id", "observed_at"]).any()


def test_arrival_blanks_future_and_inconsistent_previous_location(tmp_path):
    records = [
        _record(
            IMO=9000010,
            Port="Invented Port",
            **{
                "Arrival Date (GMT) (Entering port)": "20/12/2025 10:00:00",
                "Previous Location Departure Date (GMT)": "03/01/2026 08:00:00",
            },
        ),
        _record(
            IMO=9000011,
            Port="Invented Port",
            **{
                "Arrival Date (GMT) (Entering port)": "01/01/2025 10:00:00",
                "Previous Location Departure Date (GMT)": "02/01/2025 08:00:00",
            },
        ),
    ]
    path = tmp_path / "p10_arr_test_2019-2025.xlsx"
    _workbook(path, ARR_COLUMNS, "Invented Port", records)

    frame, counts = read_port(path)

    assert counts["previous_blanked"] == 2
    assert len(frame) == 2
    for meta in frame.meta:
        fields = json.loads(meta)
        assert not any(key.startswith("previous_") for key in fields)
        assert "2026" not in meta


def test_stay_uses_filter_place_hours_and_keeps_first_duplicate(tmp_path):
    complete = _record(
        IMO=9000020,
        **{
            "Arrival Date (GMT) (Entering port)": "01/06/2025 00:00:00",
            "Departure Date (GMT)": "01/06/2025 06:00:00",
            "Length (m)": 160.0,
        },
    )
    missing_departure = _record(
        IMO=9000021,
        **{
            "Arrival Date (GMT) (Entering port)": "02/06/2025 00:00:00",
            "Departure Date (GMT)": None,
        },
    )
    _workbook(
        tmp_path / "p03_stay_test_2019-2025.xlsx",
        STAY_COLUMNS,
        "Invented Ammonia Berth (North)",
        [complete, missing_departure],
    )
    _workbook(
        tmp_path / "p04_stay_test_2019-2025.xlsx",
        STAY_COLUMNS,
        "Invented Ammonia Berth (North)",
        [complete],
    )

    frame = LSEGPortCalls(folder=tmp_path).fetch(None, None)

    assert len(frame) == 2
    assert frame.series_id[0] == "IMO9000020.stay.invented_ammonia_berth_north"
    assert frame.value[0] == 6.0 and pd.isna(frame.value[1])
    assert frame.unit.eq("h").all()
    assert json.loads(frame.meta[0])["place"] == "Invented Ammonia Berth (North)"
    assert json.loads(frame.meta[0])["file"] == "p03_stay_test_2019-2025"
    forbidden = {"last_seen", "destination_current", "eta_gmt"}
    assert all(forbidden.isdisjoint(json.loads(meta)) for meta in frame.meta)
    assert all("2026" not in meta for meta in frame.meta)


def test_archive_stamps_the_session_as_the_vintage(tmp_path, monkeypatch):
    src = LSEGPortCalls(folder=tmp_path)
    canonical = pd.DataFrame(
        {
            "series_id": ["IMO9000030.dep.invented_port"],
            "observed_at": [pd.Timestamp("2025-01-01")],
            "value": [8.0],
            "unit": ["m"],
            "meta": ["{}"],
        }
    )
    monkeypatch.setattr(src, "fetch", lambda start, end: canonical)
    store = PITStore(tmp_path / "pit")

    assert src.archive(store, None, None) == 1
    vintage = store.vintages("lseg_portcalls")
    assert vintage.retrieved_at[0] == pd.Timestamp(RETRIEVED_AT).tz_localize(None)


def _record(**changes):
    record = {
        "Vessel": "Invented Vessel",
        "Type": "Tankers",
        "Sub-type": "LPG Tankers",
        "Sub-sub-type": "MGC",
        "Country/Region": "Testland",
        "Polygon Type": "Port",
        "Next Country/Region": "Nextland",
        "Next Location": "Next Port",
        "Next Location Draft (m) (Entering port)": 8.0,
        "Next Location Draft (m) (Exiting port)": 8.1,
        "Previous Country/Region": "Previousland",
        "Previous Location": "Previous Port",
        "Previous Location Draft (m) (Entering Location)": 7.8,
        "Previous Location Draft (m) (Exiting Location)": 7.9,
        "Last Seen": "31/12/2025 23:00:00",
        "Destination (current)": "Masked Destination",
        "ETA (GMT)": "03/01/2026 08:00:00",
        "Berth Type": "Berth",
        "Berth Type.1": "Anchorage",
        "DWT": 30000,
        "Max Draft (m)": 12.0,
        "Cubic Capacity (m3)": 35000,
        "Clean / Dirty (tankers only)": "Clean",
        "Draft (m) (Entering port)": 7.5,
        "Beam (m)": 25.0,
    }
    record.update(changes)
    return record


def _workbook(path, columns, place, records, header_at=6):
    book = Workbook()
    sheet = book.active
    sheet.cell(1, 1, "Invented Historical Vessel Data")
    sheet.cell(
        5,
        1,
        f"Port, Anchorage, Berth or Location: [{place}]; Search by: [Departed]; "
        "Date from: [2019-01-01]; Date to: [2025-12-31]; Vessel type: [Tankers]",
    )
    for column, name in enumerate(columns, start=1):
        sheet.cell(header_at + 1, column, name)
    for row, record in enumerate(records, start=header_at + 2):
        second_berth = False
        for column, name in enumerate(columns, start=1):
            key = "Berth Type.1" if name == "Berth Type" and second_berth else name
            sheet.cell(row, column, record.get(key))
            second_berth |= name == "Berth Type"
    sheet.cell(header_at + len(records) + 3, 1, "Invented export footer")
    book.save(path)
