"""Dialogue preprocessing helpers."""

from __future__ import annotations


def normalize_dialogue(text: str) -> str:
    """Normalize whitespace and strip leading/trailing spaces."""
    return " ".join(text.split()).strip()
