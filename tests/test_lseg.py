"""LSEG ship registry: the join of the two ship files and the archive vintage.

The ships here are invented. The real export stays out of the repository.
"""

import json

import pandas as pd
import pytest

from nsqueeze.io import PITStore
from nsqueeze.io.sources.lseg import RETRIEVED_AT, LSEGVessels, to_canonical

DETAILS = pd.DataFrame(
    {
        "Instrument": ["R1", "R2"],
        "Name": ["SHIP A", "SHIP B"],
        "IMO": [9000002, 9000001],
        "Sub-Type": ["LPG Tankers", "Oil Tankers"],
        "Sub-Sub-Type": ["SGC", None],
        "Dead weight tonnage": [23000, 50000],
        "Cubic Capacity": [20000.0, None],
        "Flag": ["Norway", "Malta"],
        "Overall Length": [160.0, 183.0],
        "Beam": [26.0, 32.0],
    }
)
DETAILS2 = pd.DataFrame(
    {
        "Instrument": ["R2", "R1"],
        "IMO": [9000001, 9000002],
        "Name": ["SHIP B", "SHIP A"],
        "Ex Name": [None, "OLD A"],
        "Status": ["BROKEN UP", "IN SERVICE/COMMISSION"],
        "Built": ["2001-03-01T00:00:00Z", "2016-05-01T00:00:00Z"],
        "Registered Owner": ["O2", "O1"],
        "Manager": ["M2", "M1"],
        "Hull Type": ["Double Hull", "Double Hull"],
        "Coated": [None, "Y"],
        "Class": ["C2", "C1"],
        "ICE Class": ["No", "No"],
        "Built Demolition": ["2001 / 2024", None],
    }
)


def test_join_gives_one_row_per_ship_with_volume_and_meta():
    df = to_canonical(DETAILS, DETAILS2, "2026-11-16")
    assert df.series_id.tolist() == ["IMO9000001", "IMO9000002"]
    assert df.unit.eq("m3").all() and (df.observed_at == pd.Timestamp("2026-11-16")).all()
    assert pd.isna(df.value[0]) and df.value[1] == 20000.0
    a = json.loads(df.meta[1])
    assert a["name"] == "SHIP A" and a["sub_type"] == "LPG Tankers" and a["size_class"] == "SGC"
    assert a["built"] == "2016-05-01" and a["ex_name"] == "OLD A" and a["owner"] == "O1"
    assert "size_class" not in json.loads(df.meta[0])  # blank fields are left out


def test_a_ship_in_one_file_only_is_an_error():
    with pytest.raises(ValueError, match="one ship file only"):
        to_canonical(DETAILS, DETAILS2.iloc[:1], "2026-11-16")


def test_archive_stamps_the_session_as_the_vintage(tmp_path, monkeypatch):
    src = LSEGVessels(folder=tmp_path)
    monkeypatch.setattr(src, "fetch", lambda s, e: to_canonical(DETAILS, DETAILS2, "2026-11-16"))
    store = PITStore(tmp_path / "pit")
    assert src.archive(store, None, None) == 2
    v = store.vintages("lseg_vessels")
    assert v.retrieved_at[0] == pd.Timestamp(RETRIEVED_AT).tz_localize(None)
