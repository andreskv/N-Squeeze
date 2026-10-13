"""Typed access to ``config/parameters.yaml``.

>>> P = ParameterSet.load()
>>> P.value("ammonia_tech.energy_eu_average")
35.7
>>> P["cbam.benchmark_ammonia"].grade
<Grade.C: 'C'>
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from nsqueeze.config import config_dir
from nsqueeze.provenance import Grade, Ref

_LEAF_KEYS = {"value", "grade", "range"}


class ParameterNotFitted(RuntimeError):
    """Raised when code asks for a fitted parameter before calibration."""


@dataclass(frozen=True)
class Param:
    key: str
    value: Any
    grade: Grade
    unit: str | None = None
    source: str | None = None
    url: str | None = None
    basis: str | None = None
    open_item: str | None = None
    status: str | None = None
    range: tuple[float, float] | None = None

    @property
    def ref(self) -> Ref:
        return Ref(self.key, self.grade, self.open_item)

    @property
    def is_fitted(self) -> bool:
        return bool(self.status and self.status.startswith("fitted"))


class ParameterSet(Mapping[str, Param]):
    def __init__(self, params: dict[str, Param], path: Path | None = None) -> None:
        self._params = params
        self.path = path

    # construction
    @classmethod
    def load(cls, path: Path | str | None = None) -> ParameterSet:
        path = Path(path) if path else config_dir() / "parameters.yaml"
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        raw.pop("meta", None)
        return cls(dict(_walk(raw)), path)

    # Mapping protocol
    def __getitem__(self, key: str) -> Param:
        try:
            return self._params[key]
        except KeyError:
            close = [k for k in self._params if key.split(".")[-1] in k]
            hint = f" Did you mean: {', '.join(close[:3])}?" if close else ""
            raise KeyError(f"Unknown parameter '{key}'.{hint}") from None

    def __iter__(self) -> Iterator[str]:
        return iter(self._params)

    def __len__(self) -> int:
        return len(self._params)

    # convenience
    def value(self, key: str) -> Any:
        p = self[key]
        if p.value is None and p.is_fitted:
            raise ParameterNotFitted(
                f"'{key}' is a fitted parameter and has not been calibrated yet. "
                "Fit it on pre-2026 episodes and freeze it before use."
            )
        return p.value

    def by_grade(self, grade: Grade) -> list[Param]:
        return [p for p in self._params.values() if p.grade is grade]

    def open_items(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for p in self._params.values():
            if p.open_item:
                out.setdefault(p.open_item, []).append(p.key)
        return dict(sorted(out.items()))

    def unfitted(self) -> list[Param]:
        return [p for p in self._params.values() if p.is_fitted and p.value is None]


def _walk(node: dict[str, Any], prefix: str = "", inherited: dict[str, Any] | None = None):
    """Yield (dotted_key, Param) for every leaf. ``_source``/``_open_item`` are inherited."""
    inherited = {**(inherited or {}), **{k[1:]: v for k, v in node.items() if k.startswith("_")}}
    for k, v in node.items():
        if k.startswith("_"):
            continue
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict) and _LEAF_KEYS & v.keys():
            yield key, _to_param(key, v, inherited)
        elif isinstance(v, dict):
            yield from _walk(v, key, inherited)


def _to_param(key: str, v: dict[str, Any], inherited: dict[str, Any]) -> Param:
    rng = v.get("range")
    return Param(
        key=key,
        value=v.get("value"),
        grade=Grade(v.get("grade", "C")),
        unit=v.get("unit"),
        source=v.get("source", inherited.get("source")),
        url=v.get("url"),
        basis=v.get("basis") or v.get("derivation"),
        open_item=v.get("open_item", inherited.get("open_item")),
        status=v.get("status"),
        range=tuple(rng) if rng else None,
    )
