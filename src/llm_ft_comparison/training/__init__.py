"""Training helpers and trainer stubs."""

from llm_ft_comparison.training.trainer import BaseTrainer


def train_dpo(*args, **kwargs):
    from llm_ft_comparison.training.unsloth_training import train_dpo as _train_dpo

    return _train_dpo(*args, **kwargs)


def train_lora_sft(*args, **kwargs):
    from llm_ft_comparison.training.unsloth_training import train_lora_sft as _train_lora_sft

    return _train_lora_sft(*args, **kwargs)


__all__ = ["BaseTrainer", "train_dpo", "train_lora_sft"]
