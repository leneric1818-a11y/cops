"""DPO pipeline."""

from __future__ import annotations

from llm_ft_comparison.training.unsloth_training import train_dpo


class DpoPipeline:
    def run(self, config: dict) -> str:
        return train_dpo(config)
