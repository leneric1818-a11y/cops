"""Label normalization helpers."""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable, Optional, Tuple


def normalize_label(label: Optional[str]) -> Optional[str]:
    """Normalize a label string to a canonical form."""
    if label is None:
        return None
    try:
        text = str(label)
    except Exception:
        return label

    text = unicodedata.normalize("NFC", text).strip()
    text = re.sub(r"\s+", " ", text)

    if "|" in text:
        left, right = text.split("|", 1)
        text = f"{left.strip()} | {right.strip()}"

    return text


def extract_label_code(label: Optional[str]) -> Optional[str]:
    """Extract only the label code part (before the pipe)."""
    if label is None:
        return None
    label_str = str(label).strip()
    return label_str.split("|", 1)[0].strip() if "|" in label_str else label_str


def normalize_to_code(label: Optional[str]) -> Optional[str]:
    """Normalize a label to its code part."""
    return extract_label_code(label)


def labels_equal(a: Optional[str], b: Optional[str]) -> bool:
    """Compare labels by code only."""
    return extract_label_code(a) == extract_label_code(b)


def topk_contains(topk: Optional[Iterable[Tuple[str, Any]]], target: Optional[str]) -> bool:
    """Return True if a target label is present in a top-k list."""
    if not topk or target is None:
        return False
    target_code = extract_label_code(target)
    for label, _score in topk:
        if extract_label_code(label) == target_code:
            return True
    return False
