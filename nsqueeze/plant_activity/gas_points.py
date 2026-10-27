"""Map each ammonia plant to the ENTSOG point whose flow contains its gas offtake.

The table ``config/entsog_plant_points.csv`` holds one row per plant and point: the
match class, the point keys of the ENTSOG series, a short evidence sentence and its
grade. A row with match ``none`` carries no point keys: ENTSOG publishes no flow that
contains the plant's offtake. ``load_points`` reads the table and checks it before any
model uses it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nsqueeze.config import config_dir

MATCHES = ("direct", "industrial_aggregate", "distribution", "none")
GRADES = ("A", "B", "C")


def load_points(path: Path | str | None = None) -> pd.DataFrame:
    """Read the plant-to-point table and check it. Raise ValueError on a broken row.

    I check that every plant_id is in ``config/plants.csv``, that every match is one of
    ``MATCHES``, that every grade is A, B or C, that a non-none row carries a pointKey
    and a none row does not, and that no (plant_id, pointKey, directionKey) appears
    twice.
    """
    cfg = config_dir()
    df = pd.read_csv(path or cfg / "entsog_plant_points.csv", dtype=str, keep_default_na=False)
    plants = pd.read_csv(cfg / "plants.csv", dtype=str, keep_default_na=False)

    problems = []
    unknown = sorted(set(df.plant_id) - set(plants.plant_id))
    if unknown:
        problems.append(f"plant_id not in config/plants.csv: {unknown}")
    bad_match = sorted(set(df.loc[~df.match.isin(MATCHES), "match"]))
    if bad_match:
        problems.append(f"match must be one of {MATCHES}: {bad_match}")
    bad_grade = sorted(set(df.loc[~df.grade.isin(GRADES), "grade"]))
    if bad_grade:
        problems.append(f"grade must be one of {GRADES}: {bad_grade}")

    has_key = df.pointKey != ""
    broken = df[df.match.eq("none") & has_key]
    if len(broken):
        problems.append(f"match none must not carry a pointKey: {sorted(set(broken.plant_id))}")
    broken = df[df.match.ne("none") & ~has_key]
    if len(broken):
        problems.append(f"match {sorted(set(broken.match))} needs a pointKey")

    dup = df.duplicated(["plant_id", "pointKey", "directionKey"], keep=False)
    if dup.any():
        row = df[dup].iloc[0]
        problems.append(
            f"duplicate (plant_id, pointKey, directionKey): "
            f"{row.plant_id} {row.pointKey} {row.directionKey}"
        )

    if problems:
        raise ValueError("entsog_plant_points.csv: " + "; ".join(problems))
    return df
