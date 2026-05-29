#!/usr/bin/env python3
"""Build and test simple contrastive steering vectors on a local causal LM.

This script implements a minimal activation-steering workflow:
1. Build a contrastive steering vector from prompt pairs that differ only in
   a positive vs. negative instruction.
2. Inject that vector into one transformer layer during generation.
3. Sweep layer indices and steering strengths, and write generations to JSONL.

The goal is to make the first experiment easy to run on the local repo model.
It is intentionally simple and keeps the steering source fixed to prompt
differences instead of SAE features or learned controllers.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_ft_comparison.data import extract_history, load_records
from llm_ft_comparison.models import render_chat_prompt
try:
    from peft import PeftModel
except ImportError:  # pragma: no cover - optional dependency for adapter runs
    PeftModel = None


TURN_WITH_LABEL_RE = re.compile(r"^\s*([A-Za-zÄÖÜäöüß]+)\s*\([^)]*\)\s*:\s*(.+?)\s*$")
TURN_WITHOUT_LABEL_RE = re.compile(r"^\s*([A-Za-zÄÖÜäöüß]+)\s*:\s*(.+?)\s*$")
SPEAKER_MAP = {
    "K": "Klient",
    "B": "Berater",
}
PROMPT_PREAMBLE = (
    "Du bist der Klient in einem Beratungsgespräch.\n"
    "Antworte nur mit der nächsten Äußerung des Klienten.\n"
    "Bleibe eng beim gegebenen Gesprächsverlauf und verändere keine bekannten Fakten, Zeitangaben oder Ereignisse.\n"
    "Schreibe natürliches gesprochenes Deutsch in ein bis drei Sätzen.\n"
    "Keine Erklärungen, keine Rollenbeschreibung, keine Labels, keine Regieanweisungen."
)
DEFAULT_STOP_MARKERS = [
    "Berater:",
    "Therapeut:",
    "Klient:",
    "Client:",
    "Assistant:",
    "\n\n",
    "**",
]
LEADING_ROLE_RE = re.compile(r"^\s*(Klient|Client|Assistant)\s*:\s*", re.IGNORECASE)
LEADING_THINK_RE = re.compile(r"^\s*<think>\s*.*?\s*</think>\s*", re.IGNORECASE | re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and evaluate a simple prompt-difference steering vector."
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="JSON/JSONL input with counseling contexts or conversation histories.",
    )
    parser.add_argument(
        "--model-path",
        default="models/language_models/Qwen3-4B",
        help="Base model name or local path.",
    )
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="Optional LoRA adapter path to load on top of the base model.",
    )
    parser.add_argument(
        "--context-field",
        default=None,
        help="Optional field name to use instead of auto-detecting a history/text field.",
    )
    parser.add_argument(
        "--eval-data-path",
        default=None,
        help="Optional separate dataset or JSONL of contexts used only for evaluation.",
    )
    parser.add_argument(
        "--eval-context-field",
        default=None,
        help="Optional context field for --eval-data-path. Defaults to --context-field.",
    )
    parser.add_argument(
        "--eval-manifest-path",
        default=None,
        help="Optional shared eval manifest JSONL with at least a context field.",
    )
    parser.add_argument(
        "--train-group-field",
        default=None,
        help="Optional field used to exclude held-out eval ids from the training pool.",
    )
    parser.add_argument(
        "--strip-history-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove annotation labels like 'K (K-... | ...):' from the prompt history.",
    )
    parser.add_argument(
        "--require-last-speaker",
        choices=("Klient", "Berater", "none"),
        default="Berater",
        help="Keep only contexts whose last line belongs to this speaker.",
    )
    parser.add_argument(
        "--min-turns",
        type=int,
        default=2,
        help="Minimum number of non-empty dialogue lines required in a context.",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=1200,
        help="Discard contexts longer than this many characters after cleaning. Use 0 to disable.",
    )
    parser.add_argument(
        "--positive-instruction",
        required=True,
        help="Instruction used to create the positive steering prompts.",
    )
    parser.add_argument(
        "--negative-instruction",
        required=True,
        help="Instruction used to create the negative steering prompts.",
    )
    parser.add_argument(
        "--neutral-instruction",
        default="Continue as the client in this counseling conversation.",
        help="Instruction used for evaluation prompts.",
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=32,
        help="Number of records used to estimate the steering vector.",
    )
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=32,
        help="Number of records used for generation and comparison.",
    )
    parser.add_argument(
        "--eval-offset",
        type=int,
        default=None,
        help="Start offset for evaluation records. Defaults to train-limit.",
    )
    parser.add_argument(
        "--layers",
        default="12,18,24,30",
        help="Comma-separated layer indices to test.",
    )
    parser.add_argument(
        "--steering-mode",
        choices=("single", "gaussian"),
        default="single",
        help="Apply steering at one layer at a time or across multiple layers with Gaussian weights.",
    )
    parser.add_argument(
        "--gaussian-center",
        type=float,
        default=None,
        help="Center layer for Gaussian multi-layer steering. Defaults to the midpoint of --layers.",
    )
    parser.add_argument(
        "--gaussian-sigma",
        type=float,
        default=3.0,
        help="Standard deviation for Gaussian multi-layer steering.",
    )
    parser.add_argument(
        "--gaussian-min-weight",
        type=float,
        default=0.0,
        help="Drop layers whose Gaussian weight falls below this threshold.",
    )
    parser.add_argument(
        "--alphas",
        default="0.5,1.0,1.5,2.0",
        help="Comma-separated steering strengths to test.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=120,
        help="Generation length per sample.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling threshold.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling threshold.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.05,
        help="Repetition penalty during generation.",
    )
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use sampling instead of greedy decoding.",
    )
    parser.add_argument(
        "--normalize-vector",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="L2-normalize each steering vector before use.",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
        help="Requested model dtype.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass trust_remote_code to transformers model/tokenizer loading.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base RNG seed used for reproducible generation sweeps.",
    )
    parser.add_argument(
        "--save-vector-dir",
        default=None,
        help="Optional directory for per-layer vector .pt files.",
    )
    parser.add_argument(
        "--output-path",
        default="outputs/metrics/steering_vector_experiment.jsonl",
        help="JSONL output path for generations.",
    )
    parser.add_argument(
        "--trim-to-first-utterance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trim generations to the first client utterance before saving.",
    )
    parser.add_argument(
        "--prompt-format",
        choices=("raw", "chat"),
        default="chat",
        help="Use the legacy raw prompt string or the model's chat template.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    data_path = resolve_required_path(args.data_path, root)
    eval_data_path = resolve_path(args.eval_data_path, root) if args.eval_data_path else None
    eval_manifest_path = (
        resolve_path(args.eval_manifest_path, root) if args.eval_manifest_path else None
    )
    model_path = args.model_path
    adapter_path = resolve_path(args.adapter_path, root) if args.adapter_path else None
    output_path = resolve_required_output_path(args.output_path, root)
    save_vector_dir = resolve_path(args.save_vector_dir, root) if args.save_vector_dir else None

    records = load_records(data_path)
    context_rows = extract_context_rows(
        records=records,
        context_field=args.context_field,
        strip_history_labels=args.strip_history_labels,
        require_last_speaker=None
        if args.require_last_speaker == "none"
        else args.require_last_speaker,
        min_turns=args.min_turns,
        max_context_chars=args.max_context_chars,
        id_field=args.train_group_field,
    )
    print(f"Usable contexts after filtering: {len(context_rows)} / {len(records)}")

    if eval_manifest_path is not None:
        eval_rows = load_eval_manifest_rows(eval_manifest_path)
        if args.eval_limit:
            eval_rows = eval_rows[: args.eval_limit]
    elif eval_data_path is not None:
        eval_records = load_records(eval_data_path)
        eval_rows = extract_context_rows(
            records=eval_records,
            context_field=args.eval_context_field or args.context_field,
            strip_history_labels=args.strip_history_labels,
            require_last_speaker=None
            if args.require_last_speaker == "none"
            else args.require_last_speaker,
            min_turns=args.min_turns,
            max_context_chars=args.max_context_chars,
            id_field=args.train_group_field,
        )
        if args.eval_limit:
            eval_rows = eval_rows[: args.eval_limit]
    else:
        eval_offset = args.eval_offset if args.eval_offset is not None else args.train_limit
        if eval_offset >= len(context_rows):
            raise SystemExit(
                f"eval_offset={eval_offset} is outside the usable context range ({len(context_rows)})."
            )
        eval_rows = context_rows[eval_offset : eval_offset + args.eval_limit]

    if not eval_rows:
        raise SystemExit("No evaluation contexts selected. Adjust eval settings.")

    held_out_ids = {
        normalize_eval_id(
            row.get(args.train_group_field) if args.train_group_field else row.get("seed_id")
        )
        for row in eval_rows
        if (
            row.get(args.train_group_field) if args.train_group_field else row.get("seed_id")
        )
        is not None
    }
    if held_out_ids and args.train_group_field:
        context_rows = [
            row
            for row in context_rows
            if normalize_eval_id(row.get("seed_id")) not in held_out_ids
        ]

    if len(context_rows) < args.train_limit:
        raise SystemExit(
            f"Need at least {args.train_limit} usable training contexts, found {len(context_rows)}."
        )

    train_context_rows = context_rows[: args.train_limit]
    train_contexts = [row["context"] for row in train_context_rows]
    eval_contexts = [row["context"] for row in eval_rows]

    layer_indices = parse_int_list(args.layers)
    alphas = parse_float_list(args.alphas)
    gaussian_center = (
        args.gaussian_center
        if args.gaussian_center is not None
        else (min(layer_indices) + max(layer_indices)) / 2.0
    )
    gaussian_weights = build_gaussian_layer_weights(
        layer_indices=layer_indices,
        center=gaussian_center,
        sigma=args.gaussian_sigma,
        min_weight=args.gaussian_min_weight,
    )
    if args.steering_mode == "gaussian" and not gaussian_weights:
        raise SystemExit("Gaussian steering produced no usable layer weights.")

    tokenizer, model = load_model_and_tokenizer(
        model_path=model_path,
        adapter_path=adapter_path,
        torch_dtype=args.torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )
    layer_modules = find_transformer_layers(model)
    validate_layer_indices(layer_indices, len(layer_modules))

    positive_prompts = [
        build_prompt(
            context,
            args.positive_instruction,
            tokenizer=tokenizer,
            model_path=model_path,
            prompt_format=args.prompt_format,
        )
        for context in train_contexts
    ]
    negative_prompts = [
        build_prompt(
            context,
            args.negative_instruction,
            tokenizer=tokenizer,
            model_path=model_path,
            prompt_format=args.prompt_format,
        )
        for context in train_contexts
    ]
    neutral_prompts = [
        build_prompt(
            context,
            args.neutral_instruction,
            tokenizer=tokenizer,
            model_path=model_path,
            prompt_format=args.prompt_format,
        )
        for context in eval_contexts
    ]

    vectors_by_layer = {}
    for layer_idx in tqdm(layer_indices, desc="Building steering vectors"):
        vector = build_steering_vector(
            model=model,
            tokenizer=tokenizer,
            prompts_positive=positive_prompts,
            prompts_negative=negative_prompts,
            layer_idx=layer_idx,
            normalize=args.normalize_vector,
        )
        vectors_by_layer[layer_idx] = vector
        if save_vector_dir is not None:
            save_vector_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_vector_dir / f"layer_{layer_idx}.pt"
            torch.save(
                {
                    "layer_idx": layer_idx,
                    "vector": vector,
                    "positive_instruction": args.positive_instruction,
                    "negative_instruction": args.negative_instruction,
                    "neutral_instruction": args.neutral_instruction,
                    "train_limit": args.train_limit,
                    "model_path": model_path,
                    "adapter_path": str(adapter_path) if adapter_path else None,
                },
                save_path,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "do_sample": args.do_sample,
    }

    with output_path.open("w", encoding="utf-8") as handle:
        for example_index, (context, prompt) in enumerate(
            tqdm(list(zip(eval_contexts, neutral_prompts)), desc="Generating")
        ):
            eval_row = eval_rows[example_index]
            base_seed = args.seed + example_index
            base_response = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                seed=base_seed,
                **generation_kwargs,
            )
            base_payload = build_response_payload(
                raw_response=base_response,
                trim_to_first_utterance=args.trim_to_first_utterance,
            )
            handle.write(
                json.dumps(
                    {
                        "example_index": example_index,
                        "seed_id": eval_row.get("seed_id"),
                        "mode": "base",
                        "context": context,
                        "prompt": prompt,
                        **base_payload,
                        "positive_instruction": args.positive_instruction,
                        "negative_instruction": args.negative_instruction,
                        "neutral_instruction": args.neutral_instruction,
                        "model_path": model_path,
                        "adapter_path": str(adapter_path) if adapter_path else None,
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )

            if args.steering_mode == "gaussian":
                steering_config = {
                    layer_idx: (vectors_by_layer[layer_idx], weight)
                    for layer_idx, weight in gaussian_weights.items()
                }
                for alpha in alphas:
                    steered_response = generate_text(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        seed=base_seed,
                        multi_layer_steering=steering_config,
                        steering_alpha=alpha,
                        **generation_kwargs,
                    )
                    steered_payload = build_response_payload(
                        raw_response=steered_response,
                        trim_to_first_utterance=args.trim_to_first_utterance,
                    )
                    handle.write(
                        json.dumps(
                            {
                                "example_index": example_index,
                                "seed_id": eval_row.get("seed_id"),
                                "mode": "steered_gaussian",
                                "steering_mode": "gaussian",
                                "context": context,
                                "prompt": prompt,
                                **steered_payload,
                                "alpha": alpha,
                                "gaussian_center": gaussian_center,
                                "gaussian_sigma": args.gaussian_sigma,
                                "gaussian_weights": {
                                    str(layer_idx): weight
                                    for layer_idx, weight in gaussian_weights.items()
                                },
                                "positive_instruction": args.positive_instruction,
                                "negative_instruction": args.negative_instruction,
                                "neutral_instruction": args.neutral_instruction,
                                "model_path": model_path,
                                "adapter_path": str(adapter_path) if adapter_path else None,
                            },
                            ensure_ascii=True,
                        )
                        + "\n"
                    )
            else:
                for layer_idx in layer_indices:
                    vector = vectors_by_layer[layer_idx]
                    for alpha in alphas:
                        steered_response = generate_text(
                            model=model,
                            tokenizer=tokenizer,
                            prompt=prompt,
                            seed=base_seed,
                            steering_vector=vector,
                            steering_layer=layer_idx,
                            steering_alpha=alpha,
                            **generation_kwargs,
                        )
                        steered_payload = build_response_payload(
                            raw_response=steered_response,
                            trim_to_first_utterance=args.trim_to_first_utterance,
                        )
                        handle.write(
                            json.dumps(
                                {
                                    "example_index": example_index,
                                    "seed_id": eval_row.get("seed_id"),
                                    "mode": "steered",
                                    "steering_mode": "single",
                                    "context": context,
                                    "prompt": prompt,
                                    **steered_payload,
                                    "layer_idx": layer_idx,
                                    "alpha": alpha,
                                    "positive_instruction": args.positive_instruction,
                                    "negative_instruction": args.negative_instruction,
                                    "neutral_instruction": args.neutral_instruction,
                                    "model_path": model_path,
                                    "adapter_path": str(adapter_path) if adapter_path else None,
                                },
                                ensure_ascii=True,
                            )
                            + "\n"
                        )

    print(f"Wrote generations to {output_path}")
    if save_vector_dir is not None:
        print(f"Saved vectors to {save_vector_dir}")


def resolve_required_path(path_value: str, root: Path) -> Path:
    resolved = resolve_path(path_value, root)
    if resolved is None or not resolved.exists():
        raise SystemExit(f"Path not found: {resolved}")
    return resolved


def resolve_required_output_path(path_value: str, root: Path) -> Path:
    resolved = resolve_path(path_value, root)
    if resolved is None:
        raise SystemExit("Output path could not be resolved.")
    return resolved


def parse_int_list(raw: str) -> list[int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise SystemExit("At least one layer index is required.")
    return [int(item) for item in values]


def parse_float_list(raw: str) -> list[float]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise SystemExit("At least one alpha value is required.")
    return [float(item) for item in values]


def build_gaussian_layer_weights(
    layer_indices: list[int],
    center: float,
    sigma: float,
    min_weight: float,
) -> dict[int, float]:
    if sigma <= 0:
        raise SystemExit("--gaussian-sigma must be > 0.")
    weights = {}
    for layer_idx in layer_indices:
        exponent = -((layer_idx - center) ** 2) / (2 * sigma * sigma)
        weight = float(torch.exp(torch.tensor(exponent)).item())
        weights[layer_idx] = weight

    max_weight = max(weights.values()) if weights else 0.0
    if max_weight <= 0:
        return {}

    normalized = {
        layer_idx: weight / max_weight
        for layer_idx, weight in weights.items()
        if (weight / max_weight) >= min_weight
    }
    return normalized


def extract_contexts(
    records: list[dict],
    context_field: str | None,
    strip_history_labels: bool,
    require_last_speaker: str | None,
    min_turns: int,
    max_context_chars: int,
) -> list[str]:
    return [
        row["context"]
        for row in extract_context_rows(
            records=records,
            context_field=context_field,
            strip_history_labels=strip_history_labels,
            require_last_speaker=require_last_speaker,
            min_turns=min_turns,
            max_context_chars=max_context_chars,
        )
    ]


def extract_context_rows(
    records: list[dict],
    context_field: str | None,
    strip_history_labels: bool,
    require_last_speaker: str | None,
    min_turns: int,
    max_context_chars: int,
    id_field: str | None = None,
) -> list[dict]:
    contexts: list[dict] = []
    for item in records:
        context = ""
        if context_field:
            value = item.get(context_field)
            if isinstance(value, str):
                context = value.strip()
        else:
            context = extract_history(item).strip()
            if not context:
                prompt = item.get("prompt")
                if isinstance(prompt, str):
                    context = prompt.strip()
        if strip_history_labels and context:
            context = strip_annotation_labels(context)
        if context and is_usable_context(
            context=context,
            require_last_speaker=require_last_speaker,
            min_turns=min_turns,
            max_context_chars=max_context_chars,
        ):
            row = {"context": context}
            if id_field:
                row["seed_id"] = item.get(id_field)
            contexts.append(row)
    return contexts


def load_eval_manifest_rows(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    for item in load_records(path):
        context = item.get("context")
        if not isinstance(context, str) or not context.strip():
            continue
        row = dict(item)
        row["context"] = strip_annotation_labels(context.strip())
        row["seed_id"] = resolve_eval_id(item)
        rows.append(row)
    return rows


def resolve_eval_id(item: dict) -> str | int | None:
    for field in ("seed_id", "eval_id", "seed_record_index", "original_index", "group_id"):
        value = item.get(field)
        if value is not None:
            return value
    return None


def normalize_eval_id(value) -> str | None:
    if value is None:
        return None
    return str(value)


def strip_annotation_labels(history: str) -> str:
    cleaned_lines: list[str] = []
    for raw_line in history.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        labeled = TURN_WITH_LABEL_RE.match(line)
        if labeled:
            speaker, utterance = labeled.groups()
            cleaned_lines.append(f"{normalize_speaker(speaker)}: {utterance.strip()}")
            continue

        plain = TURN_WITHOUT_LABEL_RE.match(line)
        if plain:
            speaker, utterance = plain.groups()
            cleaned_lines.append(f"{normalize_speaker(speaker)}: {utterance.strip()}")
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def normalize_speaker(raw: str) -> str:
    return SPEAKER_MAP.get(raw.strip(), raw.strip().title())


def extract_line_speaker(line: str) -> str | None:
    labeled = TURN_WITHOUT_LABEL_RE.match(line.strip())
    if not labeled:
        return None
    speaker, _ = labeled.groups()
    return normalize_speaker(speaker)


def is_usable_context(
    context: str,
    require_last_speaker: str | None,
    min_turns: int,
    max_context_chars: int,
) -> bool:
    lines = [line.strip() for line in context.splitlines() if line.strip()]
    if len(lines) < min_turns:
        return False
    if max_context_chars and len(context) > max_context_chars:
        return False
    if require_last_speaker:
        last_speaker = extract_line_speaker(lines[-1])
        if last_speaker != require_last_speaker:
            return False
    return True


def build_raw_prompt(context: str, instruction: str) -> str:
    parts = [PROMPT_PREAMBLE]
    context = context.strip()
    instruction = instruction.strip()
    if context:
        parts.append(context)
    if instruction:
        parts.append(instruction)
    parts.append("Klient:")
    return "\n\n".join(parts)


def build_chat_messages(context: str, instruction: str) -> list[dict[str, str]]:
    user_parts = []
    context = context.strip()
    instruction = instruction.strip()
    if context:
        user_parts.append(context)
    if instruction:
        user_parts.append(instruction)
    user_parts.append("Klient:")
    return [
        {"role": "system", "content": PROMPT_PREAMBLE},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def build_prompt(
    context: str,
    instruction: str,
    *,
    tokenizer=None,
    model_path: str | None = None,
    prompt_format: str = "chat",
) -> str:
    if prompt_format == "chat":
        if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError("tokenizer with apply_chat_template is required for prompt_format='chat'")
        messages = build_chat_messages(context, instruction)
        return render_chat_prompt(
            tokenizer,
            messages,
            model_name_or_path=model_path,
        )
    return build_raw_prompt(context, instruction)


def join_prompt_and_response(prompt_prefix: str, response_text: str, *, prompt_format: str) -> str:
    response = response_text.strip()
    if prompt_format == "chat":
        return f"{prompt_prefix}{response}"
    return f"{prompt_prefix} {response}"


def prompt_prefix_token_length(tokenizer, prompt_prefix: str, *, prompt_format: str) -> int:
    prefix_text = prompt_prefix if prompt_format == "chat" else f"{prompt_prefix} "
    return int(tokenizer(prefix_text, return_tensors="pt")["input_ids"].shape[1])


def clean_first_client_utterance(text: str) -> tuple[str, str | None, bool]:
    cleaned = text.strip()
    while True:
        next_cleaned = LEADING_THINK_RE.sub("", cleaned, count=1).strip()
        if next_cleaned == cleaned:
            break
        cleaned = next_cleaned
    while True:
        next_cleaned = LEADING_ROLE_RE.sub("", cleaned, count=1).strip()
        if next_cleaned == cleaned:
            break
        cleaned = next_cleaned

    stop_marker = None
    stop_pos = len(cleaned)
    for marker in DEFAULT_STOP_MARKERS:
        pos = cleaned.find(marker)
        if pos > 0 and pos < stop_pos:
            stop_pos = pos
            stop_marker = marker

    if stop_marker is not None:
        cleaned = cleaned[:stop_pos].rstrip()

    return cleaned, stop_marker, cleaned != text.strip()


def build_response_payload(raw_response: str, trim_to_first_utterance: bool) -> dict:
    if not trim_to_first_utterance:
        return {
            "response": raw_response,
            "response_raw": raw_response,
            "response_was_trimmed": False,
            "response_stop_marker": None,
        }

    cleaned_response, stop_marker, was_trimmed = clean_first_client_utterance(raw_response)
    return {
        "response": cleaned_response,
        "response_raw": raw_response,
        "response_was_trimmed": was_trimmed,
        "response_stop_marker": stop_marker,
    }


def resolve_torch_dtype(name: str) -> torch.dtype | None:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    if torch.cuda.is_available():
        return torch.bfloat16
    return torch.float32


def resolve_path(path_value: str | None, root: Path) -> Path | None:
    if not path_value:
        return None
    expanded = os.path.expandvars(os.path.expanduser(path_value))
    path = Path(expanded)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def load_model_and_tokenizer(
    model_path: str,
    adapter_path: Path | None,
    torch_dtype: str,
    trust_remote_code: bool,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "torch_dtype": resolve_torch_dtype(torch_dtype),
        "trust_remote_code": trust_remote_code,
    }
    accelerate_available = importlib.util.find_spec("accelerate") is not None
    if torch.cuda.is_available() and accelerate_available:
        load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **load_kwargs,
    )
    if "device_map" not in load_kwargs:
        if torch.cuda.is_available():
            model = model.to("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            model = model.to("mps")
    if adapter_path is not None:
        if PeftModel is None:
            raise SystemExit("peft is required to load --adapter-path but is not installed.")
        model = PeftModel.from_pretrained(model, str(adapter_path))

    model.eval()
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
    return tokenizer, model


def find_transformer_layers(model: nn.Module) -> nn.ModuleList:
    visited = set()
    queue: list[nn.Module] = [model]

    while queue:
        current = queue.pop(0)
        if id(current) in visited:
            continue
        visited.add(id(current))

        layers = getattr(current, "layers", None)
        if isinstance(layers, nn.ModuleList):
            return layers

        for attr in ("model", "base_model", "language_model"):
            child = getattr(current, attr, None)
            if isinstance(child, nn.Module):
                queue.append(child)

    raise RuntimeError("Could not find transformer layers on the loaded model.")


def validate_layer_indices(layer_indices: list[int], num_layers: int) -> None:
    invalid = [idx for idx in layer_indices if idx < 0 or idx >= num_layers]
    if invalid:
        raise SystemExit(
            f"Layer indices out of range for model with {num_layers} layers: {invalid}"
        )


def model_input_device(model: nn.Module) -> torch.device:
    return model.get_input_embeddings().weight.device


def capture_last_token_hidden(
    model: nn.Module,
    tokenizer,
    prompt: str,
    layer_idx: int,
) -> torch.Tensor:
    layers = find_transformer_layers(model)
    captured: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        captured.append(hidden_states[:, -1, :].detach().float().cpu())

    handle = layers[layer_idx].register_forward_hook(hook)
    inputs = tokenizer(prompt, return_tensors="pt").to(model_input_device(model))
    with torch.no_grad():
        model(**inputs)
    handle.remove()

    if not captured:
        raise RuntimeError(f"Failed to capture hidden state at layer {layer_idx}.")
    return captured[-1].squeeze(0)


def build_steering_vector(
    model: nn.Module,
    tokenizer,
    prompts_positive: list[str],
    prompts_negative: list[str],
    layer_idx: int,
    normalize: bool,
) -> torch.Tensor:
    positive_hiddens = []
    negative_hiddens = []

    for positive_prompt, negative_prompt in zip(prompts_positive, prompts_negative):
        positive_hiddens.append(
            capture_last_token_hidden(model, tokenizer, positive_prompt, layer_idx)
        )
        negative_hiddens.append(
            capture_last_token_hidden(model, tokenizer, negative_prompt, layer_idx)
        )

    positive_mean = torch.stack(positive_hiddens, dim=0).mean(dim=0)
    negative_mean = torch.stack(negative_hiddens, dim=0).mean(dim=0)
    vector = positive_mean - negative_mean

    if normalize:
        norm = vector.norm().item()
        if norm > 0:
            vector = vector / norm

    return vector


@contextmanager
def steering_hook(
    model: nn.Module,
    steering_vector: torch.Tensor,
    steering_layer: int,
    steering_alpha: float,
) -> Iterator[None]:
    layers = find_transformer_layers(model)
    vector = steering_vector

    def hook(_module, _inputs, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        steer = vector.to(device=hidden_states.device, dtype=hidden_states.dtype)
        hidden_states = hidden_states.clone()
        hidden_states[:, -1, :] += steering_alpha * steer
        if isinstance(output, tuple):
            return (hidden_states, *output[1:])
        return hidden_states

    handle = layers[steering_layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def multi_layer_steering_hook(
    model: nn.Module,
    steering_vectors: dict[int, tuple[torch.Tensor, float]],
    steering_alpha: float,
) -> Iterator[None]:
    layers = find_transformer_layers(model)
    handles = []

    def make_hook(vector: torch.Tensor, layer_scale: float):
        def hook(_module, _inputs, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            steer = vector.to(device=hidden_states.device, dtype=hidden_states.dtype)
            hidden_states = hidden_states.clone()
            hidden_states[:, -1, :] += steering_alpha * layer_scale * steer
            if isinstance(output, tuple):
                return (hidden_states, *output[1:])
            return hidden_states

        return hook

    for layer_idx, (vector, layer_scale) in steering_vectors.items():
        handles.append(layers[layer_idx].register_forward_hook(make_hook(vector, layer_scale)))

    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_text(
    model: nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    do_sample: bool,
    seed: int,
    steering_vector: torch.Tensor | None = None,
    steering_layer: int | None = None,
    multi_layer_steering: dict[int, tuple[torch.Tensor, float]] | None = None,
    steering_alpha: float = 0.0,
) -> str:
    seed_everything(seed)
    inputs = tokenizer(prompt, return_tensors="pt").to(model_input_device(model))
    if multi_layer_steering:
        context = multi_layer_steering_hook(
            model=model,
            steering_vectors=multi_layer_steering,
            steering_alpha=steering_alpha,
        )
    elif steering_vector is not None and steering_layer is not None:
        context = steering_hook(
            model=model,
            steering_vector=steering_vector,
            steering_layer=steering_layer,
            steering_alpha=steering_alpha,
        )
    else:
        context = nullcontext()

    with context:
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p
            generation_kwargs["top_k"] = top_k

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                **generation_kwargs,
            )

    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = output_ids[0][prompt_length:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


@contextmanager
def nullcontext() -> Iterator[None]:
    yield


if __name__ == "__main__":
    main()
