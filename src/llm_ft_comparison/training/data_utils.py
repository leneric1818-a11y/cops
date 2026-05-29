"""Dataset helpers for SFT and DPO pipelines."""

from __future__ import annotations

import os
from pathlib import Path

from datasets import Dataset, load_dataset

from llm_ft_comparison.data import load_records


def load_dataset_any(path: str | Path) -> Dataset:
    data_path = _expand_path(path)
    if data_path.suffix in {".json", ".jsonl"}:
        return load_dataset("json", data_files=str(data_path), split="train")
    records = load_records(data_path)
    return Dataset.from_list(records)


def build_sft_dataset(
    path: str | Path,
    text_field: str = "text",
    prompt_field: str | None = None,
    response_field: str | None = None,
    separator: str = "\n",
    prompt_template: str | None = None,
    eos_token: str | None = None,
) -> Dataset:
    dataset = load_dataset_any(path)

    if text_field in dataset.column_names:
        return dataset

    if prompt_field and response_field:
        def _format(example: dict) -> dict:
            prompt = example.get(prompt_field, "")
            response = example.get(response_field, "")
            if prompt_template:
                text = prompt_template.format(prompt=prompt, response=response)
            else:
                text = f"{prompt}{separator}{response}".strip()
            if eos_token:
                text = f"{text}{eos_token}"
            return {text_field: text}

        return dataset.map(_format, remove_columns=dataset.column_names)

    raise ValueError(
        "SFT dataset missing text field. Provide text_field or prompt_field/response_field."
    )


def build_dpo_dataset(
    path: str | Path,
    prompt_field: str = "prompt",
    chosen_field: str = "chosen",
    rejected_field: str = "rejected",
    prompt_template: str | None = None,
    prompt_template_first: str | None = None,
) -> Dataset:
    dataset = load_dataset_any(path)

    required = {prompt_field, chosen_field, rejected_field}
    if not required.issubset(set(dataset.column_names)):
        raise ValueError(
            "DPO dataset missing required fields. Expected: "
            f"{prompt_field}, {chosen_field}, {rejected_field}"
        )

    def _format(example: dict) -> dict:
        prompt_value = format_dpo_prompt(
            example,
            prompt_field=prompt_field,
            prompt_template=prompt_template,
            prompt_template_first=prompt_template_first,
        )
        return {
            "prompt": prompt_value,
            "chosen": example[chosen_field],
            "rejected": example[rejected_field],
        }

    return dataset.map(_format, remove_columns=dataset.column_names)


def _expand_path(path_value: str | Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(str(path_value)))
    return Path(expanded)


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return ""


def format_dpo_prompt(
    example: dict,
    prompt_field: str = "prompt",
    prompt_template: str | None = None,
    prompt_template_first: str | None = None,
) -> str:
    prompt_value = example.get(prompt_field, "")
    if not (prompt_template or prompt_template_first):
        return prompt_value

    template_vars = _build_template_vars(example, prompt_field, prompt_value)
    template = _select_prompt_template(
        template_vars=template_vars,
        prompt_template=prompt_template,
        prompt_template_first=prompt_template_first,
    )
    if not template:
        return prompt_value
    return template.format_map(_SafeDict(template_vars))


def _build_template_vars(example: dict, prompt_field: str, prompt_value: str) -> dict:
    template_vars = dict(example)

    metadata = example.get("metadata")
    if isinstance(metadata, dict):
        for key, value in metadata.items():
            template_vars.setdefault(key, value)
        template_vars.setdefault("name", metadata.get("persona", ""))

    template_vars.setdefault("prompt", prompt_value)
    template_vars.setdefault("personality_condition", prompt_value)

    chat_history = example.get("conversation_context") or example.get("chat_history") or ""
    template_vars.setdefault("conversation_context", chat_history)
    template_vars.setdefault("chat_history", chat_history)

    return template_vars


def _select_prompt_template(
    template_vars: dict,
    prompt_template: str | None,
    prompt_template_first: str | None,
) -> str | None:
    if not prompt_template_first:
        return prompt_template

    chat_history = template_vars.get("chat_history", "")
    turn = template_vars.get("turn")
    is_first_turn = not str(chat_history).strip() or str(turn) == "0"
    if is_first_turn:
        return prompt_template_first
    return prompt_template
