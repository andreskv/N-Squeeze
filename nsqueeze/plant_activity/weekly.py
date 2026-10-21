"""Weekly sensor anomalies and labelled weeks for the satellite detection test.

A plant-week is an ISO week, keyed by its Monday. Choices fixed before any score was computed:

* Labels: a week takes the label of a period that contains the whole week (Monday to Sunday).
  For "off" periods the first and last two weeks are dropped. A week with both labels is dropped.
* Weekly value: the mean of the week's scene statistics; weeks without a scene have no value.
* Anomaly: the weekly value minus the median, for the same plant and calendar month (of the
  Monday), over the running weeks before ``split``. No baseline means no anomaly.
* AUC: the probability that an "off" week scores higher than a "running" week, ties count half.
"""

from __future__ import annotations

import pandas as pd

TRIM = pd.Timedelta(weeks=2)


def label_weeks(labels: pd.DataFrame, weeks: pd.DatetimeIndex) -> pd.DataFrame:
    """Rows (plant_id, week, label) for every whole week inside a labelled period."""
    rows = []
    for r in labels.itertuples():
        start, end = pd.Timestamp(r.start), pd.Timestamp(r.end)
        if r.label == "off":
            start, end = start + TRIM, end - TRIM
        inside = weeks[(weeks >= start) & (weeks + pd.Timedelta(days=6) <= end)]
        rows += [(r.plant_id, w, r.label) for w in inside]
    out = pd.DataFrame(rows, columns=["plant_id", "week", "label"]).drop_duplicates()
    both = out.duplicated(["plant_id", "week"], keep=False)
    return out[~both].reset_index(drop=True)


def anomaly(weekly: pd.DataFrame, labelled: pd.DataFrame, split: pd.Timestamp) -> pd.Series:
    """Weekly value minus the plant x calendar-month median of running weeks before ``split``."""
    w = weekly.merge(labelled, on=["plant_id", "week"], how="left")
    w["month"] = w.week.dt.month
    cal = w[(w.label == "running") & (w.week < split)]
    base = cal.groupby(["plant_id", "month"]).value.median()
    med = w.set_index(["plant_id", "month"]).index.map(base.to_dict())
    return pd.Series(w.value.to_numpy() - med.to_numpy(dtype=float), index=w.index, name="anomaly")


def auc(score: pd.Series, off: pd.Series) -> float:
    """Rank AUC of ``score`` for off (True) against running (False)."""
    r = score.rank()
    n1, n0 = int(off.sum()), int((~off).sum())
    return float((r[off].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
