import datetime as dt

import pandas as pd
import pytest

from nsqueeze.io.pit import PITStore


def _frame(value: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_id": ["AGSI.EU.full"] * 2,
            "observed_at": ["2026-10-01", "2026-10-02"],
            "value": [value, value + 0.1],
            "unit": "%",
        }
    )


def test_as_of_returns_the_vintage_known_at_the_time(tmp_path):
    store = PITStore(tmp_path)
    t1 = dt.datetime(2026, 10, 3, 6, 0, tzinfo=dt.UTC)
    t2 = dt.datetime(2026, 10, 4, 6, 0, tzinfo=dt.UTC)
    store.write("agsi", _frame(80.0), retrieved_at=t1)
    store.write("agsi", _frame(81.0), retrieved_at=t2)  # revised data

    early = store.read("agsi", as_of=dt.datetime(2026, 10, 3, 12, tzinfo=dt.UTC))
    late = store.read("agsi", as_of=t2)
    assert early.value.tolist() == pytest.approx([80.0, 80.1])
    assert late.value.tolist() == pytest.approx([81.0, 81.1])
    assert len(store.vintages("agsi")) == 2


def test_nothing_is_known_before_first_retrieval(tmp_path):
    store = PITStore(tmp_path)
    store.write("agsi", _frame(80.0), retrieved_at=dt.datetime(2026, 10, 3, tzinfo=dt.UTC))
    assert store.read("agsi", as_of="2026-10-02").empty


def test_writes_never_overwrite(tmp_path):
    store = PITStore(tmp_path)
    t = dt.datetime(2026, 10, 3, 6, 0, tzinfo=dt.UTC)
    p1 = store.write("agsi", _frame(80.0), retrieved_at=t)
    p2 = store.write("agsi", _frame(80.0), retrieved_at=t)
    assert p1 != p2 and p1.exists() and p2.exists()


def test_rejects_non_canonical_frames(tmp_path):
    with pytest.raises(ValueError, match="canonical"):
        PITStore(tmp_path).write("agsi", pd.DataFrame({"x": [1]}))
