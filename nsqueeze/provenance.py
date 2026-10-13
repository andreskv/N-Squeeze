from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Generic, TypeVar

T = TypeVar("T")


class Grade(StrEnum):
    """A = primary/audited · B = reputable secondary · C = derived, placeholder or fitted."""

    A = "A"
    B = "B"
    C = "C"

    @property
    def rank(self) -> int:
        return {"A": 0, "B": 1, "C": 2}[self.value]

    @classmethod
    def worst(cls, grades: Iterable[Grade]) -> Grade:
        return max(grades, key=lambda g: g.rank, default=cls.A)


@dataclass(frozen=True, order=True)
class Ref:
    """A pointer to one graded input: a parameter key or a dated policy rule."""

    key: str
    grade: Grade
    open_item: str | None = None


@dataclass(frozen=True)
class Provenance:
    refs: frozenset[Ref] = field(default_factory=frozenset)

    @classmethod
    def of(cls, *refs: Ref) -> Provenance:
        return cls(frozenset(refs))

    def __or__(self, other: Provenance) -> Provenance:
        return Provenance(self.refs | other.refs)

    @property
    def grade(self) -> Grade:
        return Grade.worst(r.grade for r in self.refs)

    @property
    def flags(self) -> tuple[str, ...]:
        """Keys of grade-C inputs, i.e. what a reader must be warned about."""
        return tuple(sorted(r.key for r in self.refs if r.grade is Grade.C))

    @property
    def open_items(self) -> tuple[str, ...]:
        return tuple(sorted({r.open_item for r in self.refs if r.open_item}))

    def footnote(self) -> str:
        """One-line caption for a figure or table."""
        if not self.flags:
            return f"Evidence grade {self.grade}."
        return f"Evidence grade {self.grade} — depends on grade-C inputs: {', '.join(self.flags)}."


@dataclass(frozen=True)
class Estimate(Generic[T]):
    """A model output and the evidence behind it."""

    value: T
    provenance: Provenance = field(default_factory=Provenance)

    @property
    def grade(self) -> Grade:
        return self.provenance.grade

    @property
    def flagged(self) -> bool:
        return bool(self.provenance.flags)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        flag = " ⚑" if self.flagged else ""
        return f"Estimate({self.value!r}, grade={self.grade}{flag})"
