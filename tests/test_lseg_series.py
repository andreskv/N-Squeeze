"""LSEG daily files: filename parsing, hold-out removal and archive vintage.

Every instrument and observation in this file is invented. The licensed export stays out
of the repository.
"""

import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from nsqueeze.io import PITStore
from nsqueeze.io.sources.lseg import RETRIEVED_AT
from nsqueeze.io.sources.lseg_series import (
    LSEGSeries,
    filename_to_ric,
    read_session,
    root_of,
)


def _csv(rows: list[dict]) -> bytes:
    return pd.DataFrame(rows).to_csv(index=False).encode()


def _session(tmp_path: Path, members: dict[str, list[dict]], loose: dict[str, list[dict]]):
    with zipfile.ZipFile(tmp_path / "06_lseg_export_final_gas_freight_iv_equity.zip", "w") as z:
        for name, rows in members.items():
            z.writestr(name, _csv(rows))
        z.writestr("x_EMAc1_last20days.csv", "Date,SETTLE\n2025-01-02,999\n")
        z.writestr("search2.csv", "query,RIC,DocumentTitle\nignored,IGNORED,Ignored\n")
    folder = tmp_path / "lseg_export_folder"
    folder.mkdir()
    for name, rows in loose.items():
        (folder / name).write_bytes(_csv(rows))
    (folder / "grain_report.csv").write_text("not,read\n1,2\n")
    (folder / "search_results.csv").write_text("not,read\n1,2\n")


@pytest.mark.parametrize(
    ("name", "ric"),
    [
        ("CN0_1.csv", "CN0^1"),
        ("extra_UFEZ24_2.csv", "UFEZ24^2"),
        ("extra_SAMMFX6.csv", "SAMMFX6"),
        ("x_.BLPG.csv", ".BLPG"),
        ("x_PNX-TTFEOD-DA.csv", "PNX-TTFEOD-DA"),
        ("x_YAR.OL.csv", "YAR.OL"),
    ],
)
def test_filename_to_ric(name, ric):
    assert filename_to_ric(name) == ric


def test_root_distinguishes_delivery_month_continuation_and_full_index():
    assert root_of("CN24^2") == "C"
    assert root_of("Cc1") == "C"
    assert root_of("VTPYc1") == "VTPY"
    assert root_of("VTPMc1") == "VTPM"
    assert root_of("CATMIVMZ") == "CATMIVM"
    assert root_of("PNX-TTFEOD-DA") == "PNX-TTFEOD-DA"
    with pytest.raises(ValueError, match="no ROOTS entry"):
        root_of("INVENTEDc1")


def test_selected_fields_units_zero_and_metadata(tmp_path):
    rows = [
        {
            "Date": "2025-01-02",
            "SETTLE": "11.5",
            "TRDPRC_1": "12",
            "OFF_CLOSE": "13",
            "FRGT_RATE1": "14",
            "MID_PRICE": "15",
            "FAIR_VALUE": "16",
            "ACVOL_UNS": "0",
            "OPINT_1": "18",
            "OPEN_PRC": "999",
        },
        {
            "Date": "2025-01-03",
            "SETTLE": "0",
            "TRDPRC_1": "",
            "ACVOL_UNS": "",
            "OPINT_1": "0",
        },
    ]
    _session(tmp_path, {"extra_UFEZ24_2.csv": rows}, {})
    frame = read_session(tmp_path)
    assert set(frame.series_id.str.rsplit(".", n=1).str[1]) == {
        "SETTLE",
        "TRDPRC_1",
        "OFF_CLOSE",
        "FRGT_RATE1",
        "MID_PRICE",
        "FAIR_VALUE",
        "ACVOL_UNS",
        "OPINT_1",
    }
    assert "OPEN_PRC" not in frame.series_id.str.rsplit(".", n=1).str[1].tolist()
    assert frame.query("series_id == 'UFEZ24^2.SETTLE'").value.tolist() == [11.5, 0.0]
    assert frame.query("series_id == 'UFEZ24^2.ACVOL_UNS'").value.tolist() == [0.0]
    assert frame.query("series_id == 'UFEZ24^2.ACVOL_UNS'").unit.eq("contracts").all()
    assert frame.query("series_id == 'UFEZ24^2.OPINT_1'").unit.eq("contracts").all()
    assert frame.query("series_id == 'UFEZ24^2.SETTLE'").unit.eq("USD/t").all()
    assert set(frame.observed_at) == {pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-03")}
    for sid, meta in zip(frame.series_id, frame.meta.map(json.loads), strict=True):
        assert meta["file"] == "extra_UFEZ24_2.csv"
        if sid.rsplit(".", 1)[1] not in ("ACVOL_UNS", "OPINT_1"):
            assert meta["name"] == "CBOT urea FOB Egypt"
    volume_meta = json.loads(frame.query("series_id == 'UFEZ24^2.ACVOL_UNS'").iloc[0].meta)
    assert "name" not in volume_meta


def test_zip_loose_cutoff_blank_values_and_counts(tmp_path):
    zip_rows = [
        {"Date": "2025-01-02", "TRDPRC_1": "10"},
        {"Date": "2025-01-03", "TRDPRC_1": ""},
        {"Date": "2026-01-01", "TRDPRC_1": "999"},
        {"Date": "2026-11-16", "TRDPRC_1": "999"},
    ]
    loose_rows = [
        {"Date": "2025-02-03", "SETTLE": "0", "ACVOL_UNS": "2", "OPINT_1": "3"},
        {"Date": "2027-01-04", "SETTLE": "999"},
    ]
    _session(tmp_path, {"x_.BLPG.csv": zip_rows}, {"CN1_2.csv": loose_rows})
    frame = read_session(tmp_path)
    assert frame.series_id.unique().tolist() == [
        ".BLPG.TRDPRC_1",
        "CN1^2.SETTLE",
        "CN1^2.ACVOL_UNS",
        "CN1^2.OPINT_1",
    ]
    assert frame.value.tolist() == [10.0, 0.0, 2.0, 3.0]
    assert frame.unit.tolist() == ["unverified", "USD/bu cents", "contracts", "contracts"]
    assert frame.observed_at.max() == pd.Timestamp("2025-02-03")
    assert frame.attrs["dropped_2026"] == 3
    assert frame.attrs["files_read"] == ["x_.BLPG.csv", "CN1_2.csv"]
    assert frame.attrs["rics"] == [".BLPG", "CN1^2"]


def test_duplicate_rics_raise_before_reading_values(tmp_path):
    _session(tmp_path, {"extra_UFEZ24_2.csv": [], "x_UFEZ24_2.csv": []}, {})
    with pytest.raises(ValueError, match=r"UFEZ24\^2 comes from duplicate files"):
        read_session(tmp_path)


def test_archive_stamps_the_session_as_the_vintage(tmp_path):
    _session(
        tmp_path,
        {"x_.BLPG.csv": [{"Date": "2025-01-02", "TRDPRC_1": "10"}]},
        {},
    )
    source = LSEGSeries(folder=tmp_path)
    store = PITStore(tmp_path / "pit")
    assert source.archive(store, None, None) == 1
    vintage = store.vintages("lseg_series")
    assert vintage.retrieved_at[0] == pd.Timestamp(RETRIEVED_AT).tz_localize(None)


def test_implied_volatility_keeps_only_the_volatility_field(tmp_path):
    # On these RICs LSEG puts the underlying price in TRDPRC_1 and the expiry date in ACVOL_UNS.
    rows = [{"Date": "2025-01-02", "SETTLE": "23.5", "TRDPRC_1": "450", "ACVOL_UNS": "23062017"}]
    _session(tmp_path, {"x_CATMIVMZ.csv": rows}, {})
    frame = read_session(tmp_path)
    assert frame.series_id.tolist() == ["CATMIVMZ.SETTLE"]
    assert frame.unit.tolist() == ["%"] and frame.value.tolist() == [23.5]
