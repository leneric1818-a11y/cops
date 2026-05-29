"""Shared helpers for Unsloth-based training."""

from __future__ import annotations

from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model


DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def resolve_dtype(dtype: str | None) -> torch.dtype | None:
    if not dtype or dtype == "auto":
        return None
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def load_unsloth_model(
    base_model: str,
    max_seq_len: int,
    load_in_4bit: bool,
    dtype: str | None,
) -> tuple[Any, Any]:
    try:
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise ImportError("Unsloth is required for this pipeline.") from exc

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_len,
        dtype=resolve_dtype(dtype),
        load_in_4bit=load_in_4bit,
    )
    return model, tokenizer


def apply_lora(
    model: Any,
    lora_config: dict,
) -> Any:
    from unsloth import FastLanguageModel

    target_modules = lora_config.get("target_modules") or DEFAULT_TARGET_MODULES
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_config.get("r", 16),
        target_modules=target_modules,
        lora_alpha=lora_config.get("alpha", 32),
        lora_dropout=lora_config.get("dropout", 0.05),
        bias=lora_config.get("bias", "none"),
        use_gradient_checkpointing=lora_config.get("use_gradient_checkpointing", "unsloth"),
        random_state=lora_config.get("random_state", 3407),
        use_rslora=lora_config.get("use_rslora", False),
        loftq_config=lora_config.get("loftq_config"),
    )
    if isinstance(model, PeftModel):
        return model

    peft_config = LoraConfig(
        r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("alpha", 32),
        target_modules=target_modules,
        lora_dropout=lora_config.get("dropout", 0.05),
        bias=lora_config.get("bias", "none"),
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, peft_config)
