"""Which plants of ``config/plants.csv`` make ammonia in a given year.

``config/plant_ammonia.csv`` lists only the exceptions: a plant whose site does not make
ammonia in every year. ``ammonia_until`` is the last year with ammonia production, or
``never`` for a site that buys all its ammonia. Every other plant counts as a producer in
every year. The plant list itself stays unchanged, because other files refer to it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nsqueeze.config import config_dir


def load_status(path: Path | str | None = None) -> pd.DataFrame:
    """Read the exceptions table and check it. Raise ValueError on a broken row."""
    cfg = config_dir()
    df = pd.read_csv(path or cfg / "plant_ammonia.csv", dtype=str, keep_default_na=False)
    plants = set(pd.read_csv(cfg / "plants.csv", dtype=str).plant_id)
    problems = []
    unknown = sorted(set(df.plant_id) - plants)
    if unknown:
        problems.append(f"plant_id not in config/plants.csv: {unknown}")
    bad = df[~(df.ammonia_until.eq("never") | df.ammonia_until.str.fullmatch(r"(19|20)\d\d"))]
    if len(bad):
        problems.append(f"ammonia_until must be a year or 'never': {sorted(bad.plant_id)}")
    if df.plant_id.duplicated().any():
        problems.append("a plant_id appears twice")
    if problems:
        raise ValueError("plant_ammonia.csv: " + "; ".join(problems))
    return df


def producers(year: int, path: Path | str | None = None) -> list[str]:
    """Plant IDs that make ammonia in ``year``."""
    status = load_status(path).set_index("plant_id").ammonia_until
    plants = pd.read_csv(config_dir() / "plants.csv", dtype=str).plant_id
    out = []
    for pid in plants:
        until = status.get(pid)
        if until is None or (until != "never" and year <= int(until)):
            out.append(pid)
    return out
