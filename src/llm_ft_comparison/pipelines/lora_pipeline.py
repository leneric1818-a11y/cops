"""LoRA fine-tuning pipeline."""

from __future__ import annotations

from llm_ft_comparison.training.unsloth_training import train_lora_sft


class LoRAPipeline:
    def run(self, config: dict) -> str:
        return train_lora_sft(config)
