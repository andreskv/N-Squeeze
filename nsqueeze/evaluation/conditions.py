"""The four conditions for a signal, as data, so reports can render pass/fail uniformly.

A signal must pass, in order: measurement → information → price impact → tradability.
A failed condition is reported as a result, not hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Condition(StrEnum):
    MEASUREMENT = "measurement"
    INFORMATION = "information"
    PRICE_IMPACT = "price_impact"
    TRADABILITY = "tradability"


ORDER = (
    Condition.MEASUREMENT,
    Condition.INFORMATION,
    Condition.PRICE_IMPACT,
    Condition.TRADABILITY,
)


@dataclass(frozen=True)
class ConditionResult:
    condition: Condition
    passed: bool
    metric: str
    value: float
    threshold: float
    note: str = ""


@dataclass
class SignalReport:
    signal: str
    results: list[ConditionResult] = field(default_factory=list)

    def record(self, result: ConditionResult) -> None:
        expected = ORDER[len(self.results)] if len(self.results) < len(ORDER) else None
        if result.condition is not expected:
            raise ValueError(
                f"Conditions are checked in order; expected {expected}, got {result.condition}"
            )
        if self.results and not self.results[-1].passed:
            raise ValueError(f"{self.signal}: stopped at {self.results[-1].condition} — report it")
        self.results.append(result)

    @property
    def stopped_at(self) -> Condition | None:
        return next((r.condition for r in self.results if not r.passed), None)
