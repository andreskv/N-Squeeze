"""Price impact. Event studies with event-clustered inference.

Few independent events exist (two or three regimes, a handful of shocks). Never
present daily-return t-statistics as if days were independent events: cluster by
event, report effect sizes with honest intervals.

Planned events (S2): Hormuz closure (28 Feb 2026), June agreement, ceasefire
collapse, reopening steps; (S3): curtailment announcements with/without a prior
EAUI signal. Dates are pre-registered in PREREGISTRATION.md before estimation.
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

    ``events`` columns: event_id, ticker, date. To be implemented in Phase 2 (S2).
    """
    raise NotImplementedError("S2 event study: not built yet")


def clustered_car(abnormal: pd.DataFrame, *, cluster: str = "event_id") -> pd.DataFrame:
    """Cumulative abnormal returns with standard errors clustered by ``cluster``."""
    raise NotImplementedError("S2 event study: not built yet")
