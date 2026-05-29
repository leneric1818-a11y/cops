"""Simple dataset loaders."""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Iterator, Sequence


def load_jsonl(path: str | Path) -> Iterator[dict]:
    """Yield records from a JSONL file."""
    jsonl_path = _expand_path(path)
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_records(path: str | Path) -> list[dict]:
    """Load records from JSONL, JSON, or pickle."""
    record_path = _expand_path(path)
    if not record_path.exists():
        raise FileNotFoundError(f"Dataset not found: {record_path}")

    suffix = record_path.suffix.lower()
    if suffix == ".jsonl":
        return list(load_jsonl(record_path))
    if suffix == ".json":
        return json.loads(record_path.read_text(encoding="utf-8"))
    if suffix in {".pkl", ".pickle"}:
        with record_path.open("rb") as handle:
            return pickle.load(handle)

    raise ValueError(f"Unsupported dataset extension: {suffix}")


def _expand_path(path_value: str | Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(str(path_value)))
    return Path(expanded)
