"""Small helpers for CLI scripts."""

import os
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_path(path_value: str | None, root: Path) -> Path | None:
    if not path_value:
        return None
    expanded = os.path.expandvars(os.path.expanduser(path_value))
    path = Path(expanded)
    if path.is_absolute():
        return path
    return (root / path).resolve()
