"""LoRA + DPO pipeline."""

from __future__ import annotations

from llm_ft_comparison.training.unsloth_training import train_dpo, train_lora_sft


class LoRADpoPipeline:
    def run(self, config: dict) -> str:
        stage_cfg = config.get("lora_dpo", {})
        adapter_path = stage_cfg.get("adapter_path")

        if stage_cfg.get("run_sft_first", False):
            sft_config = stage_cfg.get("sft_config") or config
            adapter_path = train_lora_sft(sft_config)

        return train_dpo(config, adapter_path=adapter_path)
