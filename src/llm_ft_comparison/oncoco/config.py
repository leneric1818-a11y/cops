"""Configuration defaults for the OnCoCo classifier."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_ONCOCO_MODEL_PATH = os.environ.get(
    "ONCOCO_MODEL_PATH",
    str(PROJECT_ROOT / "models" / "xlm-roberta-large-OnCoCo-DE-EN"),
)

DEFAULT_ONCOCO_MAPPING_PATH = os.environ.get(
    "ONCOCO_MAPPING_PATH",
    str(PROJECT_ROOT / "data" / "oncoco" / "OnCoCo_Categories_DE_EN.csv"),
)

FILE_ENCODINGS = ["utf-8", "latin1", "cp1252", "iso-8859-1"]
