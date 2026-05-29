"""Lightweight model package exports.

Keep registry utilities importable without pulling in heavyweight training
dependencies. The predictor classes remain available via lazy imports.
"""

from __future__ import annotations

from llm_ft_comparison.models.model_registry import (
    infer_model_family,
    load_model_registry,
    model_supports_reft,
    render_chat_prompt,
    resolve_model_registry_entry,
    slugify_model_name,
)

__all__ = [
    "CategoryPredictor",
    "TransitionMatrix",
    "TransitionMatrixFilter",
    "infer_model_family",
    "load_model_registry",
    "model_supports_reft",
    "render_chat_prompt",
    "resolve_model_registry_entry",
    "slugify_model_name",
]


def __getattr__(name: str):
    if name == "CategoryPredictor":
        from llm_ft_comparison.models.category_predictor import CategoryPredictor

        return CategoryPredictor
    if name in {"TransitionMatrix", "TransitionMatrixFilter"}:
        from llm_ft_comparison.models.transition_matrix import TransitionMatrix, TransitionMatrixFilter

        return {
            "TransitionMatrix": TransitionMatrix,
            "TransitionMatrixFilter": TransitionMatrixFilter,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
