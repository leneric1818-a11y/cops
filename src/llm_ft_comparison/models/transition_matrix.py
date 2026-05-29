"""Transition matrix utilities."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import pandas as pd


def _pairwise(sequence: Iterable[str]) -> Iterable[tuple[str, str]]:
    iterator = iter(sequence)
    try:
        prev = next(iterator)
    except StopIteration:
        return
    for item in iterator:
        yield prev, item
        prev = item


class TransitionMatrix:
    """Estimate next-category probabilities from sequences."""

    def __init__(self, smoothing: float = 1.0) -> None:
        self.smoothing = smoothing
        self.labels: list[str] = []
        self.probabilities: dict[str, dict[str, float]] = {}

    def fit(self, sequences: list[list[str]]) -> None:
        counts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        label_set = set()

        for sequence in sequences:
            label_set.update(sequence)
            for src, dst in _pairwise(sequence):
                counts[src][dst] += 1.0

        self.labels = sorted(label_set)
        for src in self.labels:
            total = 0.0
            self.probabilities[src] = {}
            for dst in self.labels:
                value = counts[src][dst] + self.smoothing
                self.probabilities[src][dst] = value
                total += value
            for dst in self.labels:
                self.probabilities[src][dst] /= total

    def next_distribution(self, current_label: str) -> dict[str, float]:
        if current_label not in self.probabilities:
            raise KeyError(f"Unknown label: {current_label}")
        return dict(self.probabilities[current_label])

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.probabilities).T

    def save_csv(self, path: str) -> None:
        self.to_frame().to_csv(path)

    @classmethod
    def from_csv(cls, path: str) -> "TransitionMatrix":
        df = pd.read_csv(path, index_col=0)
        instance = cls(smoothing=0.0)
        instance.labels = list(df.index)
        instance.probabilities = df.to_dict(orient="index")
        return instance


class TransitionMatrixFilter:
    """Filter and rerank predictions using a transition matrix."""

    def __init__(
        self,
        transition_matrix_path: str,
        threshold: float = 0.01,
        rerank_threshold: float = 0.05,
        use_code_matching: bool = True,
    ) -> None:
        self.threshold = threshold
        self.rerank_threshold = rerank_threshold
        self.use_code_matching = use_code_matching

        self.transition_df = pd.read_csv(transition_matrix_path, index_col=0)
        self._build_transition_maps()

    def _extract_code(self, category: str) -> str:
        if self.use_code_matching and "|" in category:
            return category.split("|")[0].strip()
        return category

    def _build_transition_maps(self) -> None:
        self.allowed_transitions: dict[str, set[str]] = {}
        self.transition_probs: dict[str, dict[str, float]] = {}

        for from_cat in self.transition_df.index:
            from_code = self._extract_code(from_cat)
            row = self.transition_df.loc[from_cat]
            allowed = row[row >= self.threshold]

            self.allowed_transitions[from_code] = set()
            self.transition_probs[from_code] = {}

            for to_cat, prob in allowed.items():
                to_code = self._extract_code(to_cat)
                self.allowed_transitions[from_code].add(to_code)
                self.transition_probs[from_code][to_code] = float(prob)

    def filter_predictions(self, predictions: list[dict], current_category: str | None = None) -> list[dict]:
        if not current_category:
            return predictions

        current_code = self._extract_code(current_category)
        allowed = self.allowed_transitions.get(current_code)
        if not allowed:
            return predictions

        filtered = []
        for pred in predictions:
            pred_code = self._extract_code(pred["category"])
            if pred_code in allowed:
                filtered.append(pred)
        return filtered

    def rerank_predictions(self, predictions: list[dict], current_category: str | None = None) -> list[dict]:
        if not current_category:
            return predictions

        current_code = self._extract_code(current_category)
        trans_probs = self.transition_probs.get(current_code)
        if not trans_probs:
            return predictions

        reranked = []
        for pred in predictions:
            pred_code = self._extract_code(pred["category"])
            trans_prob = trans_probs.get(pred_code, 0.0)
            model_prob = pred["probability"]
            combined_score = (model_prob ** 0.7) * (max(trans_prob, 1e-6) ** 0.3)

            reranked.append(
                {
                    "category": pred["category"],
                    "probability": pred["probability"],
                    "transition_probability": trans_prob,
                    "combined_score": combined_score,
                }
            )

        reranked.sort(key=lambda item: item["combined_score"], reverse=True)
        return reranked
