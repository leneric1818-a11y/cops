"""Helpers for extracting dialogue fields and labels."""

from __future__ import annotations

import re
from typing import Iterable, Sequence


HISTORY_FIELDS = ("conversation_history", "history", "dialogue_history", "text")
LABEL_FIELDS = ("label", "target_category", "next_category")
CURRENT_FIELDS = ("current_category", "category")


def first_present(item: dict, fields: Sequence[str]) -> str:
    for field in fields:
        value = item.get(field)
        if value:
            return value
    return ""


def extract_history(item: dict, fields: Sequence[str] = HISTORY_FIELDS) -> str:
    return first_present(item, fields)


def extract_current_category(item: dict, fields: Sequence[str] = CURRENT_FIELDS) -> str:
    return first_present(item, fields)


def extract_target_category(item: dict, fields: Sequence[str] = LABEL_FIELDS) -> str:
    return first_present(item, fields)


def extract_prev_category_from_history(history: str) -> str:
    """Extract the last labeled category from a history line.

    Expects lines like: "K (K-... | ...): text".
    """
    if not history:
        return ""
    last_line = history.strip().split("\n")[-1]
    match = re.search(r"\(([^)]+)\)\s*:", last_line)
    return match.group(1).strip() if match else ""


def extract_last_utterance(history: str) -> str:
    if not history:
        return ""
    last_line = history.strip().split("\n")[-1]
    if ":" in last_line:
        return last_line.split(":", 1)[-1].strip()
    return last_line.strip()


def iter_histories(items: Iterable[dict]) -> Iterable[str]:
    for item in items:
        history = extract_history(item)
        if history:
            yield history
