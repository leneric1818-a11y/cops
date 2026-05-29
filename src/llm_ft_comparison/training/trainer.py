"""Base trainer stub."""

from __future__ import annotations


class BaseTrainer:
    def train(self) -> None:
        raise NotImplementedError("Training logic not implemented.")
