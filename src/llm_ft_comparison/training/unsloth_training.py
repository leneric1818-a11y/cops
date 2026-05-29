"""Unsloth-based training utilities for SFT and DPO."""

from __future__ import annotations

import inspect
from pathlib import Path

import unsloth  # noqa: F401
from peft import PeftModel

from transformers import TrainingArguments
from trl import DPOTrainer, SFTTrainer
try:
    from trl import DPOConfig
except ImportError:  # pragma: no cover - older TRL versions
    DPOConfig = None

from llm_ft_comparison.training.data_utils import build_dpo_dataset, build_sft_dataset
from llm_ft_comparison.training.unsloth_helpers import apply_lora, load_unsloth_model


def train_lora_sft(config: dict) -> str:
    dataset_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})

    output_dir = Path(config.get("output_dir", "outputs/checkpoints/lora_sft"))
    output_dir.mkdir(parents=True, exist_ok=True)

    max_seq_len = training_cfg.get("max_seq_len", 2048)
    model, tokenizer = load_unsloth_model(
        base_model=model_cfg["base_model"],
        max_seq_len=max_seq_len,
        load_in_4bit=model_cfg.get("load_in_4bit", True),
        dtype=model_cfg.get("dtype"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_cfg = model_cfg.get("lora")
    if lora_cfg:
        model = apply_lora(model, lora_cfg)

    eos_token = tokenizer.eos_token if tokenizer.eos_token else None
    dataset = build_sft_dataset(
        path=dataset_cfg["path"],
        text_field=dataset_cfg.get("text_field", "text"),
        prompt_field=dataset_cfg.get("prompt_field"),
        response_field=dataset_cfg.get("response_field"),
        separator=dataset_cfg.get("separator", "\n"),
        prompt_template=dataset_cfg.get("prompt_template"),
        eos_token=eos_token,
    )

    split_ratio = training_cfg.get("validation_split", 0.0)
    if split_ratio:
        split = dataset.train_test_split(test_size=split_ratio, seed=training_cfg.get("seed", 42))
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        train_dataset = dataset
        eval_dataset = None

    eval_steps = training_cfg.get("eval_steps", 200) if eval_dataset is not None else None
    run_name = training_cfg.get("run_name") or config.get("experiment_name")
    logging_dir = training_cfg.get("logging_dir")
    args = TrainingArguments(
        **_build_training_args(
            output_dir=output_dir,
            training_cfg=training_cfg,
            eval_dataset_present=eval_dataset is not None,
            eval_steps=eval_steps,
            run_name=run_name,
            logging_dir=logging_dir,
        )
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field=dataset_cfg.get("text_field", "text"),
        max_seq_length=max_seq_len,
        packing=training_cfg.get("packing", False),
        args=args,
    )
    trainer.train()

    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    return str(output_dir)


def train_dpo(config: dict, adapter_path: str | None = None) -> str:
    dataset_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    dpo_cfg = config.get("dpo", {})

    output_dir = Path(config.get("output_dir", "outputs/checkpoints/dpo"))
    output_dir.mkdir(parents=True, exist_ok=True)

    max_seq_len = training_cfg.get("max_seq_len", 2048)
    model, tokenizer = load_unsloth_model(
        base_model=model_cfg["base_model"],
        max_seq_len=max_seq_len,
        load_in_4bit=model_cfg.get("load_in_4bit", True),
        dtype=model_cfg.get("dtype"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_cfg = model_cfg.get("lora")
    if lora_cfg and adapter_path is None:
        model = apply_lora(model, lora_cfg)
    elif model_cfg.get("load_in_4bit", True) and adapter_path is None:
        raise ValueError(
            "DPO on 4-bit models requires LoRA adapters. "
            "Set model.lora or provide an adapter_path."
        )

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    if model_cfg.get("load_in_4bit", True) and not isinstance(model, PeftModel):
        raise ValueError(
            "DPO on 4-bit models requires LoRA adapters. "
            "Set model.lora or provide an adapter_path."
        )

    dataset = build_dpo_dataset(
        path=dataset_cfg["path"],
        prompt_field=dataset_cfg.get("prompt_field", "prompt"),
        chosen_field=dataset_cfg.get("chosen_field", "chosen"),
        rejected_field=dataset_cfg.get("rejected_field", "rejected"),
        prompt_template=dataset_cfg.get("prompt_template"),
        prompt_template_first=dataset_cfg.get("prompt_template_first"),
    )

    split_ratio = training_cfg.get("validation_split", 0.0)
    if split_ratio:
        split = dataset.train_test_split(test_size=split_ratio, seed=training_cfg.get("seed", 42))
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        train_dataset = dataset
        eval_dataset = None

    eval_steps = training_cfg.get("eval_steps", 200) if eval_dataset is not None else None
    run_name = training_cfg.get("run_name") or config.get("experiment_name")
    logging_dir = training_cfg.get("logging_dir")
    base_args = _build_training_args(
        output_dir=output_dir,
        training_cfg=training_cfg,
        eval_dataset_present=eval_dataset is not None,
        eval_steps=eval_steps,
        run_name=run_name,
        logging_dir=logging_dir,
    )
    dpo_args = {**base_args, **_build_dpo_config_args(dpo_cfg, max_seq_len)}
    if DPOConfig is not None:
        args = DPOConfig(**dpo_args)
    else:
        args = TrainingArguments(**base_args)
    _ensure_args_attrs(args)

    trainer = DPOTrainer(
        **_build_dpo_trainer_kwargs(
            model=model,
            args=args,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dpo_cfg=dpo_cfg,
            max_seq_len=max_seq_len,
        )
    )
    trainer.train()

    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    return str(output_dir)


def _build_training_args(
    output_dir: Path,
    training_cfg: dict,
    eval_dataset_present: bool,
    eval_steps: int | None,
    run_name: str | None,
    logging_dir: str | None,
) -> dict:
    eval_strategy = "steps" if eval_dataset_present else "no"
    max_steps = training_cfg.get("max_steps")
    kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": _coerce_float(training_cfg.get("epochs", 1)),
        "max_steps": _coerce_int(max_steps) if max_steps is not None else -1,
        "per_device_train_batch_size": _coerce_int(training_cfg.get("batch_size", 1)),
        "per_device_eval_batch_size": _coerce_int(training_cfg.get("batch_size", 1)),
        "gradient_accumulation_steps": _coerce_int(training_cfg.get("gradient_accumulation_steps", 1)),
        "learning_rate": _coerce_float(training_cfg.get("lr", 1e-5)),
        "warmup_steps": _coerce_int(training_cfg.get("warmup_steps", 0)),
        "logging_steps": _coerce_int(training_cfg.get("logging_steps", 50)),
        "save_steps": _coerce_int(training_cfg.get("save_steps", 200)),
        "eval_steps": _coerce_int(eval_steps) if eval_steps is not None else None,
        "save_total_limit": _coerce_int(training_cfg.get("save_total_limit", 2)),
        "fp16": training_cfg.get("fp16", False),
        "bf16": training_cfg.get("bf16", False),
        "optim": training_cfg.get("optim", "adamw_8bit"),
        "report_to": training_cfg.get("report_to", []),
        "run_name": run_name,
        "logging_dir": logging_dir,
        "seed": training_cfg.get("seed", 42),
    }

    signature = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in signature.parameters:
        kwargs["evaluation_strategy"] = eval_strategy
    else:
        kwargs["eval_strategy"] = eval_strategy

    return kwargs


def _build_dpo_config_args(dpo_cfg: dict, max_seq_len: int) -> dict:
    max_prompt_length = _coerce_int(dpo_cfg.get("max_prompt_length", max_seq_len // 2))
    max_completion_length = dpo_cfg.get("max_completion_length")
    if max_completion_length is None:
        max_completion_length = dpo_cfg.get("max_target_length", max_seq_len // 2)

    return {
        "beta": _coerce_float(dpo_cfg.get("beta", 0.1)),
        "max_length": _coerce_int(dpo_cfg.get("max_length", max_seq_len)),
        "max_prompt_length": max_prompt_length,
        "max_completion_length": _coerce_int(max_completion_length),
    }


def _build_dpo_trainer_kwargs(
    model,
    args: TrainingArguments,
    tokenizer,
    train_dataset,
    eval_dataset,
    dpo_cfg: dict,
    max_seq_len: int,
) -> dict:
    length_kwargs = _build_dpo_trainer_lengths(dpo_cfg, max_seq_len)
    kwargs = {
        "model": model,
        "ref_model": None,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "tokenizer": tokenizer,
        "processing_class": tokenizer,
        **length_kwargs,
    }
    return _filter_kwargs(DPOTrainer.__init__, kwargs)


def _build_dpo_trainer_lengths(dpo_cfg: dict, max_seq_len: int) -> dict:
    max_prompt_length = _coerce_int(dpo_cfg.get("max_prompt_length", max_seq_len // 2))
    max_completion_length = dpo_cfg.get("max_completion_length")
    if max_completion_length is None:
        max_completion_length = dpo_cfg.get("max_target_length", max_seq_len // 2)

    return {
        "max_length": _coerce_int(dpo_cfg.get("max_length", max_seq_len)),
        "max_prompt_length": max_prompt_length,
        "max_completion_length": _coerce_int(max_completion_length),
        "max_target_length": _coerce_int(dpo_cfg.get("max_target_length"))
        if dpo_cfg.get("max_target_length") is not None
        else None,
    }


def _filter_kwargs(callable_obj, kwargs: dict) -> dict:
    signature = inspect.signature(callable_obj)
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def _ensure_args_attrs(args: TrainingArguments) -> None:
    if not hasattr(args, "model_init_kwargs"):
        setattr(args, "model_init_kwargs", None)


def _coerce_int(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return value


def _coerce_float(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return value
