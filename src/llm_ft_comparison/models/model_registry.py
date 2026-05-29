from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_REGISTRY_PATH = ROOT / "configs" / "models" / "persona_model_matrix_v1.json"


def slugify_model_name(model_name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", model_name.strip()).strip("_").lower()
    return re.sub(r"_+", "_", slug)


def infer_model_family(model_name_or_path: str | None) -> str:
    lowered = (model_name_or_path or "").strip().lower()
    if "qwen" in lowered:
        return "qwen"
    if "gemma" in lowered:
        return "gemma"
    return "generic"


@lru_cache(maxsize=1)
def load_model_registry(path: str | Path | None = None) -> dict:
    registry_path = Path(path) if path else DEFAULT_MODEL_REGISTRY_PATH
    if not registry_path.exists():
        return {"models": []}
    return json.loads(registry_path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def model_registry_by_path(path: str | Path | None = None) -> dict[str, dict]:
    registry = load_model_registry(path)
    indexed = {}
    for entry in registry.get("models", []):
        model_path = entry.get("model_path")
        if isinstance(model_path, str) and model_path.strip():
            indexed[model_path] = entry
    return indexed


def resolve_model_registry_entry(model_name_or_path: str | None, path: str | Path | None = None) -> dict | None:
    if not model_name_or_path:
        return None
    return model_registry_by_path(path).get(model_name_or_path)


def model_supports_reft(model_name_or_path: str | None, path: str | Path | None = None) -> bool:
    entry = resolve_model_registry_entry(model_name_or_path, path)
    if entry is not None:
        return bool(entry.get("supports_reft", False))
    return infer_model_family(model_name_or_path) == "qwen"


def render_chat_prompt(tokenizer, messages: list[dict[str, str]], *, model_name_or_path: str | None = None) -> str:
    family = infer_model_family(model_name_or_path or getattr(tokenizer, "name_or_path", None))

    common_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if family == "qwen":
        try:
            return tokenizer.apply_chat_template(
                messages,
                enable_thinking=False,
                **common_kwargs,
            )
        except TypeError:
            patched_messages = [dict(message) for message in messages]
            patched_messages[-1]["content"] = f"/no_think\n\n{patched_messages[-1]['content']}"
            return tokenizer.apply_chat_template(
                patched_messages,
                **common_kwargs,
            )
    return tokenizer.apply_chat_template(messages, **common_kwargs)
