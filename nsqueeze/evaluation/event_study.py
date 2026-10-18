"""Price impact. Event studies with event-clustered inference.

Few independent events exist (two or three regimes, a handful of shocks). Never
present daily-return t-statistics as if days were independent events: cluster by
event, report effect sizes with honest intervals.

Planned events. Hormuz event study by capacity location: Hormuz closure (28 Feb 2026),
first partial and full reopening, June agreement, ceasefire collapse. Unexpected
curtailment: announcements with or without a prior plant-activity signal. Event dates
are fixed in the pre-registration before estimation.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class EventWindow:
    pre: int = -5  # trading days relative to event
    post: int = 10
    estimation: int = 120  # days before `pre` used for the normal-return model


def abnormal_returns(
    returns: pd.DataFrame, market: pd.Series, events: pd.DataFrame, window: EventWindow
) -> pd.DataFrame:
    """Market-model abnormal returns, stacked (event_id, ticker, rel_day).

    ``events`` columns: event_id, ticker, date. Not built yet (Hormuz event study).
    """
    raise NotImplementedError("Hormuz event study: not built yet")


def clustered_car(abnormal: pd.DataFrame, *, cluster: str = "event_id") -> pd.DataFrame:
    """Cumulative abnormal returns with standard errors clustered by ``cluster``."""
    raise NotImplementedError("Hormuz event study: not built yet")
