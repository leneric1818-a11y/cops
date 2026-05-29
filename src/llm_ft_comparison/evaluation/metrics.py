"""Minimal evaluation helpers."""

from __future__ import annotations


def compute_accuracy(predictions: list[str], labels: list[str]) -> float:
    if not labels:
        return 0.0
    correct = sum(1 for pred, label in zip(predictions, labels) if pred == label)
    return correct / len(labels)
