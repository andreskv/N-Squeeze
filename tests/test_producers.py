"""Ammonia producers by year: the exceptions table and its checks."""

import pandas as pd
import pytest

from nsqueeze.plant_activity.producers import load_status, producers


def test_config_table_loads_and_counts_producers():
    load_status()
    assert len(producers(2018)) == 33  # Ravenna never
    assert len(producers(2020)) == 32  # and Devnya stopped in 2019
    assert "BG-DEVNYA" in producers(2019) and "IT-RAVENN" not in producers(2016)


@pytest.mark.parametrize(
    "row, message",
    [
        (["XX-NONE", "2019"], "not in config/plants.csv"),
        (["BG-DEVNYA", "soon"], "a year or 'never'"),
    ],
)
def test_rejects_broken_rows(tmp_path, row, message):
    path = tmp_path / "status.csv"
    pd.DataFrame([row], columns=["plant_id", "ammonia_until"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match=message):
        load_status(path)
