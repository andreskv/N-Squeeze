"""Dated regulatory parameters (config/policy_schedule.csv)
regulatory parameters are stored as dated rows with an effective-from date, so a historical run uses
only the rules in force at that time.
Looking up a rule on a date it was not in force raises rather than falling back to today's value.
"""

from __future__ import annotations

import contextlib
import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from nsqueeze.config import config_dir
from nsqueeze.provenance import Estimate, Grade, Provenance, Ref

DateLike = str | dt.date | pd.Timestamp


class NoRuleInForce(LookupError):
    pass


@dataclass(frozen=True)
class Rule:
    parameter: str
    scope: str
    value: float | None
    unit: str
    effective_from: pd.Timestamp
    effective_to: pd.Timestamp | None
    grade: Grade
    source: str
    notes: str

    @property
    def ref(self) -> Ref:
        start = self.effective_from.date().isoformat()
        oi = next((tok for tok in self.notes.split() if tok.startswith("OI-")), None)
        return Ref(f"policy:{self.parameter}[{self.scope}]@{start}", self.grade, oi)


class PolicySchedule:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    @classmethod
    def load(cls, path: Path | str | None = None) -> PolicySchedule:
        path = Path(path) if path else config_dir() / "policy_schedule.csv"
        df = pd.read_csv(path, dtype={"scope": str}, keep_default_na=False, na_values=[""])
        df["effective_from"] = pd.to_datetime(df["effective_from"])
        df["effective_to"] = pd.to_datetime(df["effective_to"])
        df["notes"] = df["notes"].fillna("")
        return cls(df)

    def rule(self, parameter: str, scope: str, on: DateLike) -> Rule:
        """The single rule in force for ``parameter``/``scope`` on date ``on``.

        Where a temporary rule overlaps a standing one (e.g. the 2026–27 MFN duty
        suspension over the normal MFN rate), the most recently started rule wins.
        """
        t = pd.Timestamp(on)
        df = self.frame
        m = (
            (df.parameter == parameter)
            & (df.scope == scope)
            & (df.effective_from <= t)
            & (df.effective_to.isna() | (df.effective_to >= t))
        )
        hits = df[m].sort_values("effective_from")
        if hits.empty:
            raise NoRuleInForce(f"No '{parameter}' rule for scope '{scope}' in force on {t.date()}")
        r = hits.iloc[-1]
        return Rule(
            parameter=r.parameter,
            scope=r.scope,
            value=None if pd.isna(r.value) else float(r.value),
            unit=r.unit,
            effective_from=r.effective_from,
            effective_to=None if pd.isna(r.effective_to) else r.effective_to,
            grade=Grade(r.grade),
            source=r.source,
            notes=r.notes,
        )

    def value(self, parameter: str, scope: str, on: DateLike) -> Estimate[float]:
        r = self.rule(parameter, scope, on)
        if r.value is None:
            raise NoRuleInForce(f"'{parameter}' [{scope}] has no value yet — open item: {r.notes}")
        return Estimate(r.value, Provenance.of(r.ref))

    def series(self, parameter: str, scope: str, dates: pd.DatetimeIndex) -> pd.Series:
        """Vectorised lookup; NaN where no rule is in force."""
        out = pd.Series(index=dates, dtype=float, name=f"{parameter}[{scope}]")
        for d in dates:
            with contextlib.suppress(NoRuleInForce):
                out[d] = self.rule(parameter, scope, d).value
        return out
