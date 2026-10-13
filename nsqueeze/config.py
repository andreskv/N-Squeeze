"""
repository's config directory.

Resolution order: ``$NSQ_CONFIG_DIR`` → ``<repo>/config`` next to this package.
"""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent


def config_dir() -> Path:
    override = os.environ.get("NSQ_CONFIG_DIR")
    path = Path(override) if override else REPO_ROOT / "config"
    if not path.is_dir():
        raise FileNotFoundError(f"Config directory not found: {path} (set NSQ_CONFIG_DIR)")
    return path


def data_dir() -> Path:
    return Path(os.environ.get("NSQ_DATA_DIR", REPO_ROOT / "data"))
