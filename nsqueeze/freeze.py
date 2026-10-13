"""Pre-registration guard. Reads ``config/freeze.yaml``.

Until parameters are frozen, 2026 hold-out episodes cannot be loaded. This turns
"don't peek" from a promise into a code path that raises.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from nsqueeze.config import config_dir


class HoldoutLocked(PermissionError):
    pass


@dataclass(frozen=True)
class FreezeState:
    frozen: bool
    preregistration_commit: str | None
    frozen_at: str | None

    @classmethod
    def load(cls, path: Path | None = None) -> FreezeState:
        raw = yaml.safe_load((path or config_dir() / "freeze.yaml").read_text(encoding="utf-8"))
        return cls(bool(raw["frozen"]), raw.get("preregistration_commit"), raw.get("frozen_at"))


def sealed_after(state: FreezeState | None = None) -> pd.Timestamp | None:
    """Last date any analysis may use while unfrozen (end of the calibration window).

    Returns ``None`` once frozen. Data after this date may be *archived* (archivers
    never stop), but analysis code must drop it, so no 2026 result can leak into a
    figure before the pre-registration commit exists.
    """
    s = state or FreezeState.load()
    if s.frozen and s.preregistration_commit:
        return None
    raw = yaml.safe_load((config_dir() / "freeze.yaml").read_text(encoding="utf-8"))
    return pd.Timestamp(raw["calibration_window"][1])


def require_frozen(state: FreezeState | None = None) -> FreezeState:
    s = state or FreezeState.load()
    if not (s.frozen and s.preregistration_commit):
        raise HoldoutLocked(
            "Parameters are not frozen. Commit PREREGISTRATION.md, record the hash in "
            "config/freeze.yaml and set frozen: true before touching 2026 hold-out data."
        )
    return s


def load_episodes(*, include_holdout: bool = False) -> pd.DataFrame:
    """Calibration episodes; 2026 hold-out episodes only after the freeze."""
    df = pd.read_csv(config_dir() / "episodes.csv", parse_dates=["date"])
    holdout_path = config_dir() / "episodes_holdout_2026.csv"
    if include_holdout:
        require_frozen()
        if holdout_path.exists():
            df = pd.concat([df, pd.read_csv(holdout_path, parse_dates=["date"])])
    return df.reset_index(drop=True)
