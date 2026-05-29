#!/usr/bin/env python3
"""Compute dense activation and sparse transcoder feature stats by style.

Given a flat JSONL dataset with rows like:
  {context, response, style}

this script:
1. Builds a neutral prompt ending in the provided client response.
2. Captures the MLP input activations at one or more Qwen layers.
3. Averages those activations over the response tokens.
4. Computes per-style dense activation means and sparse transcoder feature stats.

The transcoder path uses layer-specific Qwen3-4B transcoders from
`mwhanna/qwen3-4b-transcoders`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_ft_comparison.data import load_records
from steering_vector_experiment import (
    PROMPT_PREAMBLE,
    find_transformer_layers,
    load_model_and_tokenizer,
    model_input_device,
    resolve_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute dense activation and transcoder feature stats by style."
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Flat JSONL dataset with fields: context, response, style.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for dense stats and top feature reports.",
    )
    parser.add_argument(
        "--model-path",
        default="Qwen/Qwen3-4B",
        help="Base model name or local path.",
    )
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="Optional LoRA adapter path to load on top of the base model.",
    )
    parser.add_argument(
        "--layers",
        default="18",
        help="Comma-separated layer indices to analyze.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of most positive / negative features to save per layer.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for forward passes.",
    )
    parser.add_argument(
        "--max-examples-per-style",
        type=int,
        default=0,
        help="Optional cap per style. Use 0 for all available examples.",
    )
    parser.add_argument(
        "--styles",
        default="defensive,open",
        help="Comma-separated style labels to compare.",
    )
    parser.add_argument(
        "--transcoder-repo",
        default="mwhanna/qwen3-4b-transcoders",
        help="Hugging Face repo id containing layer_{idx}.safetensors files.",
    )
    parser.add_argument(
        "--neutral-instruction",
        default="Der Klient antwortet auf die letzte Aussage des Beraters.",
        help="Neutral instruction inserted before the client reply.",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
        help="Torch dtype for model loading.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to trust remote code when loading the tokenizer/model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir, ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_records(args.data_path)
    style_order = [part.strip() for part in args.styles.split(",") if part.strip()]
    if len(style_order) != 2:
        raise SystemExit("--styles must contain exactly two comma-separated labels.")

    selected_rows = select_rows(rows, style_order, args.max_examples_per_style)
    counts = {style: sum(1 for row in selected_rows if row["style"] == style) for style in style_order}
    print(f"Selected rows by style: {counts}")
    if min(counts.values()) == 0:
        raise SystemExit(f"Need at least one example for each style. Got: {counts}")

    tokenizer, model = load_model_and_tokenizer(
        model_path=args.model_path,
        adapter_path=resolve_path(args.adapter_path, ROOT),
        torch_dtype=args.torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )
    layers = find_transformer_layers(model)
    layer_indices = parse_layers(args.layers, len(layers))

    config_row = {
        "data_path": str(resolve_path(args.data_path, ROOT)),
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "layer_indices": layer_indices,
        "styles": style_order,
        "batch_size": args.batch_size,
        "max_examples_per_style": args.max_examples_per_style,
        "transcoder_repo": args.transcoder_repo,
        "neutral_instruction": args.neutral_instruction,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config_row, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for layer_idx in layer_indices:
        print(f"\nProcessing layer {layer_idx}")
        transcoder = load_transcoder(args.transcoder_repo, layer_idx, model_input_device(model))
        layer_result = compute_layer_stats(
            model=model,
            tokenizer=tokenizer,
            rows=selected_rows,
            style_order=style_order,
            layer_idx=layer_idx,
            batch_size=args.batch_size,
            neutral_instruction=args.neutral_instruction,
            transcoder=transcoder,
        )
        save_layer_outputs(output_dir, layer_idx, style_order, layer_result, args.top_k)


def select_rows(rows: list[dict], style_order: list[str], max_examples_per_style: int) -> list[dict]:
    buckets = {style: [] for style in style_order}
    for row in rows:
        style = row.get("style")
        if style in buckets and isinstance(row.get("context"), str) and isinstance(row.get("response"), str):
            buckets[style].append(row)
    if max_examples_per_style > 0:
        for style in style_order:
            buckets[style] = buckets[style][:max_examples_per_style]

    selected: list[dict] = []
    for style in style_order:
        selected.extend(buckets[style])
    return selected


def parse_layers(raw: str, num_layers: int) -> list[int]:
    indices = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    invalid = [idx for idx in indices if idx < 0 or idx >= num_layers]
    if invalid:
        raise SystemExit(f"Layer indices out of range for model with {num_layers} layers: {invalid}")
    return indices


def build_prefix(context: str, neutral_instruction: str) -> str:
    return "\n\n".join(
        [
            PROMPT_PREAMBLE,
            context.strip(),
            neutral_instruction.strip(),
            "Klient:",
        ]
    )


def tokenize_batch(tokenizer, prefixes: list[str], responses: list[str], device: torch.device):
    full_texts = []
    prefix_lens = []
    for prefix, response in zip(prefixes, responses):
        prefix_with_space = f"{prefix} "
        prefix_ids = tokenizer(prefix_with_space, return_tensors="pt")["input_ids"][0]
        prefix_lens.append(int(prefix_ids.shape[0]))
        full_texts.append(prefix_with_space + response.strip())

    inputs = tokenizer(full_texts, return_tensors="pt", padding=True).to(device)
    return inputs, prefix_lens


def load_transcoder(repo_id: str, layer_idx: int, device: torch.device) -> dict[str, torch.Tensor]:
    path = hf_hub_download(repo_id=repo_id, filename=f"layer_{layer_idx}.safetensors")
    with safe_open(path, framework="pt") as handle:
        w_enc = handle.get_tensor("W_enc").to(device=device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
        b_enc = handle.get_tensor("b_enc").to(device=device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
    return {"W_enc": w_enc, "b_enc": b_enc}


def init_running_stats(size: int) -> dict[str, torch.Tensor | int]:
    return {
        "count": 0,
        "sum": torch.zeros(size, dtype=torch.float64),
        "sumsq": torch.zeros(size, dtype=torch.float64),
        "nonzero": torch.zeros(size, dtype=torch.float64),
    }


def update_running_stats(bucket: dict[str, torch.Tensor | int], values: torch.Tensor) -> None:
    values = values.detach().float().cpu().to(dtype=torch.float64)
    bucket["count"] = int(bucket["count"]) + values.shape[0]
    bucket["sum"] += values.sum(dim=0)
    bucket["sumsq"] += (values * values).sum(dim=0)
    bucket["nonzero"] += (values > 0).sum(dim=0)


def finalize_running_stats(bucket: dict[str, torch.Tensor | int]) -> dict[str, torch.Tensor | int]:
    count = int(bucket["count"])
    if count == 0:
        raise RuntimeError("Cannot finalize stats with zero observations.")
    mean = bucket["sum"] / count
    mean_sq = bucket["sumsq"] / count
    var = torch.clamp(mean_sq - (mean * mean), min=0.0)
    std = torch.sqrt(var)
    nonzero_frac = bucket["nonzero"] / count
    return {
        "count": count,
        "mean": mean.float(),
        "std": std.float(),
        "nonzero_frac": nonzero_frac.float(),
    }


def compute_layer_stats(
    model,
    tokenizer,
    rows: list[dict],
    style_order: list[str],
    layer_idx: int,
    batch_size: int,
    neutral_instruction: str,
    transcoder: dict[str, torch.Tensor],
) -> dict:
    device = model_input_device(model)
    layers = find_transformer_layers(model)
    mlp = layers[layer_idx].mlp

    dense_dim = int(transcoder["W_enc"].shape[1])
    feat_dim = int(transcoder["W_enc"].shape[0])
    dense_stats = {style: init_running_stats(dense_dim) for style in style_order}
    feat_stats = {style: init_running_stats(feat_dim) for style in style_order}

    ordered_rows = []
    for style in style_order:
        ordered_rows.extend([row for row in rows if row["style"] == style])

    for start in tqdm(range(0, len(ordered_rows), batch_size), desc=f"Layer {layer_idx} batches"):
        batch = ordered_rows[start : start + batch_size]
        prefixes = [build_prefix(row["context"], neutral_instruction) for row in batch]
        responses = [row["response"] for row in batch]
        styles = [row["style"] for row in batch]
        inputs, prefix_lens = tokenize_batch(tokenizer, prefixes, responses, device)
        attention_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        captured: list[torch.Tensor] = []

        def pre_hook(_module, module_inputs):
            hidden = module_inputs[0]
            captured.append(hidden.detach())

        handle = mlp.register_forward_pre_hook(pre_hook)
        with torch.no_grad():
            model(**inputs)
        handle.remove()

        if not captured:
            raise RuntimeError(f"Failed to capture MLP input at layer {layer_idx}.")

        hidden = captured[-1]
        hidden_means = []
        for row_idx, (prefix_len, seq_len) in enumerate(zip(prefix_lens, attention_lengths)):
            if prefix_len >= seq_len:
                raise RuntimeError(
                    f"Empty response token span for row {start + row_idx} at layer {layer_idx}."
                )
            response_hidden = hidden[row_idx, prefix_len:seq_len, :]
            hidden_means.append(response_hidden.mean(dim=0))
        hidden_means = torch.stack(hidden_means, dim=0)

        feature_acts = torch.relu(hidden_means.to(transcoder["W_enc"].dtype) @ transcoder["W_enc"].T + transcoder["b_enc"])

        for style in style_order:
            mask = [idx for idx, row_style in enumerate(styles) if row_style == style]
            if not mask:
                continue
            dense_batch = hidden_means[mask]
            feat_batch = feature_acts[mask]
            update_running_stats(dense_stats[style], dense_batch)
            update_running_stats(feat_stats[style], feat_batch)

    dense_final = {style: finalize_running_stats(bucket) for style, bucket in dense_stats.items()}
    feat_final = {style: finalize_running_stats(bucket) for style, bucket in feat_stats.items()}
    return {
        "dense": dense_final,
        "features": feat_final,
    }


def save_layer_outputs(
    output_dir: Path,
    layer_idx: int,
    style_order: list[str],
    layer_result: dict,
    top_k: int,
) -> None:
    dense = layer_result["dense"]
    features = layer_result["features"]
    style_a, style_b = style_order

    dense_delta = dense[style_b]["mean"] - dense[style_a]["mean"]
    torch.save(
        {
            "layer": layer_idx,
            "style_order": style_order,
            "dense": dense,
            "dense_delta": dense_delta,
        },
        output_dir / f"layer_{layer_idx}_dense_stats.pt",
    )

    feat_delta = features[style_b]["mean"] - features[style_a]["mean"]
    pooled_std = torch.sqrt((features[style_a]["std"] ** 2 + features[style_b]["std"] ** 2) / 2.0 + 1e-8)
    effect = feat_delta / pooled_std

    top_pos = torch.topk(effect, k=min(top_k, effect.numel()))
    top_neg = torch.topk(-effect, k=min(top_k, effect.numel()))

    rows = []
    for direction, indices in (("positive", top_pos.indices), ("negative", top_neg.indices)):
        for idx in indices.tolist():
            rows.append(
                {
                    "layer": layer_idx,
                    "direction": direction,
                    "feature_idx": idx,
                    "effect": float(effect[idx].item()),
                    "delta_mean": float(feat_delta[idx].item()),
                    f"{style_a}_mean": float(features[style_a]["mean"][idx].item()),
                    f"{style_b}_mean": float(features[style_b]["mean"][idx].item()),
                    f"{style_a}_nonzero_frac": float(features[style_a]["nonzero_frac"][idx].item()),
                    f"{style_b}_nonzero_frac": float(features[style_b]["nonzero_frac"][idx].item()),
                }
            )

    with (output_dir / f"layer_{layer_idx}_top_features.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)

    fieldnames = list(rows[0].keys()) if rows else [
        "layer",
        "direction",
        "feature_idx",
        "effect",
        "delta_mean",
        f"{style_a}_mean",
        f"{style_b}_mean",
        f"{style_a}_nonzero_frac",
        f"{style_b}_nonzero_frac",
    ]
    with (output_dir / f"layer_{layer_idx}_top_features.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "layer": layer_idx,
        "style_order": style_order,
        "counts": {style: int(features[style]["count"]) for style in style_order},
        "dense_delta_norm": float(torch.norm(dense_delta).item()),
        "feature_effect_abs_mean": float(torch.mean(torch.abs(effect)).item()),
        "top_positive_feature": int(top_pos.indices[0].item()),
        "top_negative_feature": int(top_neg.indices[0].item()),
        "top_positive_effect": float(effect[top_pos.indices[0]].item()),
        "top_negative_effect": float(effect[top_neg.indices[0]].item()),
    }
    (output_dir / f"layer_{layer_idx}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
