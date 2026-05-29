from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
CLAUSE_SPLIT_RE = re.compile(r"(?<=[,;:])\s+")


def require_pyreft():
    try:
        import pyreft  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise SystemExit(
            "pyreft is required for the ReFT benchmark family. "
            "Install it with `pip install git+https://github.com/stanfordnlp/pyreft.git`."
        ) from exc
    return pyreft


def transform_target_text(
    text: str,
    target_mode: str,
    max_target_chars: int | None,
) -> str:
    cleaned = " ".join(text.strip().split())
    if target_mode == "first_sentence":
        cleaned = SENTENCE_SPLIT_RE.split(cleaned, maxsplit=1)[0].strip()
    elif target_mode == "first_clause":
        cleaned = CLAUSE_SPLIT_RE.split(cleaned, maxsplit=1)[0].strip()
        cleaned = SENTENCE_SPLIT_RE.split(cleaned, maxsplit=1)[0].strip()

    if max_target_chars and max_target_chars > 0 and len(cleaned) > max_target_chars:
        truncated = cleaned[:max_target_chars].rstrip()
        last_space = truncated.rfind(" ")
        if last_space >= max(8, max_target_chars // 2):
            truncated = truncated[:last_space].rstrip()
        cleaned = truncated.rstrip(" ,;:")
    return cleaned.strip()


def build_reft_examples(
    rows: list[dict],
    target_field: str,
    build_prompt,
    neutral_instruction: str,
    target_mode: str,
    max_target_chars: int | None,
    tokenizer=None,
    model_path: str | None = None,
    prompt_format: str = "raw",
):
    prompts = []
    targets = []
    for row in rows:
        target = row.get(target_field)
        if not isinstance(target, str) or not target.strip():
            continue
        target_text = transform_target_text(
            target,
            target_mode=target_mode,
            max_target_chars=max_target_chars,
        )
        if not target_text:
            continue
        prompts.append(
            build_prompt(
                row["context"],
                neutral_instruction,
                tokenizer=tokenizer,
                model_path=model_path,
                prompt_format=prompt_format,
            )
        )
        targets.append(target_text)
    return prompts, targets


def resolve_reft_hidden_size(model) -> int | None:
    config = getattr(model, "config", None)
    if config is None:
        return None

    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is not None:
        return int(hidden_size)

    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        hidden_size = getattr(text_config, "hidden_size", None)
        if hidden_size is not None:
            return int(hidden_size)
    return None


def build_reft_config(pyreft, model, layers: list[int], rank: int, component: str):
    hidden_size = resolve_reft_hidden_size(model)
    if hidden_size is None:
        raise SystemExit("The base model config does not expose hidden_size for LoReFT.")
    representations = [
        {
            "layer": layer_idx,
            "component": component,
            "low_rank_dimension": rank,
            "intervention": pyreft.LoreftIntervention(
                embed_dim=hidden_size,
                low_rank_dimension=rank,
            ),
        }
        for layer_idx in layers
    ]
    return pyreft.ReftConfig(representations=representations)


def build_unit_locations(num_layers: int, prompt_length: int) -> dict:
    last_prompt_index = max(int(prompt_length) - 1, 0)
    return {
        "sources->base": (
            None,
            [[[last_prompt_index]] for _ in range(num_layers)],
        )
    }


def maybe_save_reft_artifacts(reft_model, save_dir: Path | None) -> None:
    if save_dir is None:
        return
    save_dir.mkdir(parents=True, exist_ok=True)
    reft_model.set_device("cpu")
    reft_model.save(save_directory=str(save_dir))


def maybe_patch_qwen3_modelcard(model) -> bool:
    """Register Qwen3 classes against the existing Qwen2 pyvene mappings.

    pyvene 0.1.8 ships Qwen2 mappings but not Qwen3 ones. Qwen3-4B uses the
    same top-level `model.layers[*]` layout, so the Qwen2 causal-LM mapping is
    the most reasonable first compatibility shim.
    """
    model_name = type(model).__name__
    if not model_name.startswith("Qwen3"):
        return False

    try:
        from pyvene.models.intervenable_modelcard import (
            type_to_dimension_mapping,
            type_to_module_mapping,
        )
        from pyvene.models.qwen2.modelings_intervenable_qwen2 import (
            qwen2_lm_type_to_dimension_mapping,
            qwen2_lm_type_to_module_mapping,
            qwen2_type_to_dimension_mapping,
            qwen2_type_to_module_mapping,
        )
    except ImportError:  # pragma: no cover - runtime dependency
        return False

    model_type = type(model)
    type_to_module_mapping.setdefault(model_type, qwen2_lm_type_to_module_mapping)
    type_to_dimension_mapping.setdefault(model_type, qwen2_lm_type_to_dimension_mapping)

    base_model = getattr(model, "model", None)
    if base_model is not None:
        type_to_module_mapping.setdefault(type(base_model), qwen2_type_to_module_mapping)
        type_to_dimension_mapping.setdefault(type(base_model), qwen2_type_to_dimension_mapping)
    return True


def _rewrite_module_mapping_paths(module_mapping: dict, replacements: list[tuple[str, str]]) -> dict:
    rewritten = {}
    for key, value in module_mapping.items():
        if not isinstance(value, tuple) or not value:
            rewritten[key] = deepcopy(value)
            continue
        path = value[0]
        if isinstance(path, str):
            for old, new in replacements:
                path = path.replace(old, new)
            rewritten[key] = (path, *deepcopy(value[1:]))
        else:
            rewritten[key] = deepcopy(value)
    return rewritten


def _promote_text_config_attrs(config, attr_names: list[str]) -> None:
    text_config = getattr(config, "text_config", None)
    if text_config is None:
        return
    for name in attr_names:
        if getattr(config, name, None) is None and getattr(text_config, name, None) is not None:
            setattr(config, name, getattr(text_config, name))


def maybe_patch_gemma4_modelcard(model) -> bool:
    """Register Gemma4 classes against the existing Gemma2 pyvene mappings.

    Gemma4 chat models currently load as `Gemma4ForConditionalGeneration`, where
    the text stack sits under `model.language_model.layers[*]`. pyvene 0.1.8
    ships Gemma/Gemma2 mappings but not Gemma4 ones, so we patch the wrapper and
    text-model classes onto the closest existing mapping layout.
    """
    model_name = type(model).__name__
    if not model_name.startswith("Gemma4"):
        return False

    try:
        from pyvene.models.intervenable_modelcard import (
            type_to_dimension_mapping,
            type_to_module_mapping,
        )
        from pyvene.models.gemma2.modelings_intervenable_gemma2 import (
            gemma2_lm_type_to_dimension_mapping,
            gemma2_lm_type_to_module_mapping,
            gemma2_type_to_dimension_mapping,
            gemma2_type_to_module_mapping,
        )
    except ImportError:  # pragma: no cover - runtime dependency
        return False

    gemma4_lm_type_to_module_mapping = _rewrite_module_mapping_paths(
        gemma2_lm_type_to_module_mapping,
        [
            ("model.layers[%s]", "model.language_model.layers[%s]"),
            ("model.embed_tokens", "model.language_model.embed_tokens"),
            ("model.norm", "model.language_model.norm"),
            ("model.rotary_emb", "model.language_model.rotary_emb"),
        ],
    )
    gemma4_wrapper_type_to_module_mapping = _rewrite_module_mapping_paths(
        gemma2_type_to_module_mapping,
        [
            ("layers[%s]", "language_model.layers[%s]"),
            ("embed_tokens", "language_model.embed_tokens"),
            ("norm", "language_model.norm"),
            ("rotary_emb", "language_model.rotary_emb"),
        ],
    )

    model_type = type(model)
    _promote_text_config_attrs(
        model.config,
        [
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "intermediate_size",
        ],
    )
    type_to_module_mapping.setdefault(model_type, gemma4_lm_type_to_module_mapping)
    type_to_dimension_mapping.setdefault(model_type, gemma2_lm_type_to_dimension_mapping)

    wrapper_model = getattr(model, "model", None)
    if wrapper_model is not None:
        type_to_module_mapping.setdefault(type(wrapper_model), gemma4_wrapper_type_to_module_mapping)
        type_to_dimension_mapping.setdefault(type(wrapper_model), gemma2_type_to_dimension_mapping)

        text_model = getattr(wrapper_model, "language_model", None)
        if text_model is not None:
            type_to_module_mapping.setdefault(type(text_model), gemma2_type_to_module_mapping)
            type_to_dimension_mapping.setdefault(type(text_model), gemma2_type_to_dimension_mapping)
    return True
