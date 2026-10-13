"""The four rigour gates as data, so reports can render pass/fail uniformly.

A signal must pass, in order: measurement → information → price impact → tradability.
A failure at any gate is reported as a result, not hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Gate(StrEnum):
    MEASUREMENT = "measurement"
    INFORMATION = "information"
    PRICE_IMPACT = "price_impact"
    TRADABILITY = "tradability"


ORDER = (Gate.MEASUREMENT, Gate.INFORMATION, Gate.PRICE_IMPACT, Gate.TRADABILITY)


@dataclass(frozen=True)
class GateResult:
    gate: Gate
    passed: bool
    metric: str
    value: float
    threshold: float
    note: str = ""


@dataclass
class SignalReport:
    signal: str
    results: list[GateResult] = field(default_factory=list)

    def record(self, result: GateResult) -> None:
        expected = ORDER[len(self.results)] if len(self.results) < len(ORDER) else None
        if result.gate is not expected:
            raise ValueError(f"Gates run in order; expected {expected}, got {result.gate}")
        if self.results and not self.results[-1].passed:
            raise ValueError(f"{self.signal}: stopped at {self.results[-1].gate} — report it")
        self.results.append(result)

    @property
    def stopped_at(self) -> Gate | None:
        return next((r.gate for r in self.results if not r.passed), None)
