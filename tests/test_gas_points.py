"""The plant-to-ENTSOG-point table and its reader, on the real config and on broken rows."""

import pandas as pd
import pytest

from nsqueeze.config import config_dir
from nsqueeze.plant_activity.gas_points import MATCHES, balance, load_points

COLUMNS = [
    "plant_id",
    "match",
    "operatorKey",
    "pointKey",
    "directionKey",
    "pointLabel",
    "evidence",
    "source_url",
    "grade",
    "checked",
]
ROW = [
    "BE-ANTWER",
    "industrial_aggregate",
    "BE-TSO-0001",
    "FNC-00033",
    "exit",
    "Industrial clients H-zone",
    "aggregate holds the site",
    "https://example.org",
    "C",
    "2026-10-27",
]


def write(tmp_path, rows):
    p = tmp_path / "points.csv"
    pd.DataFrame(rows, columns=COLUMNS).to_csv(p, index=False)
    return p


def test_config_table_loads_and_covers_every_plant():
    df = load_points()
    plants = pd.read_csv(config_dir() / "plants.csv")
    assert set(df.plant_id) == set(plants.plant_id)
    assert set(df.match) <= set(MATCHES)
    assert set(df.grade) <= {"A", "B", "C"}


def test_none_row_without_point_key_loads(tmp_path):
    none_row = [
        "AT-LINZ",
        "none",
        "",
        "",
        "",
        "",
        "no Austrian exit point",
        "https://example.org",
        "C",
        "2026-10-27",
    ]
    df = load_points(write(tmp_path, [none_row]))
    assert df.match[0] == "none" and df.pointKey[0] == ""


def test_rejects_unknown_plant(tmp_path):
    row = list(ROW)
    row[0] = "XX-NOWHERE"
    with pytest.raises(ValueError, match="plant_id"):
        load_points(write(tmp_path, [row]))


def test_rejects_bad_match_and_grade(tmp_path):
    bad_match = list(ROW)
    bad_match[1] = "probably"
    with pytest.raises(ValueError, match="match"):
        load_points(write(tmp_path, [bad_match]))
    bad_grade = list(ROW)
    bad_grade[8] = "D"
    with pytest.raises(ValueError, match="grade"):
        load_points(write(tmp_path, [bad_grade]))


def test_rejects_none_with_point_and_missing_point(tmp_path):
    none_with_key = [
        "AT-LINZ",
        "none",
        "AT-TSO-0001",
        "FNC-00001",
        "exit",
        "x",
        "y",
        "https://example.org",
        "C",
        "2026-10-27",
    ]
    with pytest.raises(ValueError, match="pointKey"):
        load_points(write(tmp_path, [none_with_key]))
    missing = list(ROW)
    missing[3] = ""
    with pytest.raises(ValueError, match="pointKey"):
        load_points(write(tmp_path, [missing]))


def test_rejects_duplicate_plant_point_direction(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        load_points(write(tmp_path, [list(ROW), list(ROW)]))


def test_balance_is_entries_less_exits_and_skips_incomplete_days():
    rows = pd.DataFrame(
        {
            "operatorKey": ["LT-TSO-0001"] * 3,
            "pointKey": ["ITP-1", "ITP-2", "ITP-3"],
            "directionKey": ["entry", "exit", "entry"],
        }
    )
    days = pd.to_datetime(["2022-01-01", "2022-01-02", "2022-01-03"])

    def flow(key, direction, values):
        sid = f"ENTSOG.LT-TSO-0001.{key}.{direction}"
        return pd.DataFrame({"series_id": sid, "observed_at": days, "value": values})

    flows = pd.concat(
        [
            flow("ITP-1", "entry", [100.0, 100.0, None]),  # missing on day 3
            flow("ITP-2", "exit", [30.0, 30.0, 30.0]),
            flow("ITP-3", "entry", [None, 5.0, 5.0]),  # a new point from day 2
            flow("ITP-9", "entry", [999.0, 999.0, 999.0]),  # not in the rows
        ]
    ).dropna()
    out = balance(flows, rows)
    assert out.to_dict() == {days[0]: 70.0, days[1]: 75.0}
