#!/usr/bin/env python3
"""Measure layerwise separability for defensive vs open client replies.

This script uses a flat JSONL dataset with rows like:
  {context, response, style}

For each example, it:
1. Builds the neutral client prompt plus the provided response text.
2. Captures response-token hidden states at one or more transformer layers.
3. Averages the response-token hidden states into one vector per example/layer.
4. Fits a simple centroid-direction classifier on a train split.
5. Reports held-out separability metrics per layer.

The saved per-layer vectors can later be reused for steering experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_ft_comparison.data import load_records
from steering_vector_experiment import (
    build_prompt,
    find_transformer_layers,
    join_prompt_and_response,
    load_model_and_tokenizer,
    model_input_device,
    prompt_prefix_token_length,
    resolve_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure layerwise defensive/open separability on response-token activations."
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Flat JSONL dataset with fields: context, response, style.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for layerwise summaries and saved layer vectors.",
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
        default="all",
        help="Comma-separated layer indices to analyze, or 'all'.",
    )
    parser.add_argument(
        "--styles",
        default="defensive,open",
        help="Comma-separated style labels to compare. Must contain exactly two labels.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for forward passes.",
    )
    parser.add_argument(
        "--max-examples-per-style",
        type=int,
        default=0,
        help="Optional cap per style. Use 0 for all available examples.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.25,
        help="Held-out fraction per style for separability evaluation.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Seed for the stratified train/test split.",
    )
    parser.add_argument(
        "--neutral-instruction",
        default="Der Klient antwortet auf die letzte Aussage des Beraters.",
        help="Neutral instruction inserted before the client reply.",
    )
    parser.add_argument(
        "--prompt-format",
        choices=("raw", "chat"),
        default="chat",
        help="Use the legacy raw prompt string or the model's chat template.",
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

    style_order = [part.strip() for part in args.styles.split(",") if part.strip()]
    if len(style_order) != 2:
        raise SystemExit("--styles must contain exactly two comma-separated labels.")
    if not 0.0 < args.test_fraction < 1.0:
        raise SystemExit("--test-fraction must be between 0 and 1.")

    rows = load_records(args.data_path)
    selected_rows = select_rows(rows, style_order, args.max_examples_per_style)
    counts = {style: sum(1 for row in selected_rows if row["style"] == style) for style in style_order}
    print(f"Selected rows by style: {counts}")
    if min(counts.values()) < 2:
        raise SystemExit(f"Need at least two examples for each style. Got: {counts}")

    tokenizer, model = load_model_and_tokenizer(
        model_path=args.model_path,
        adapter_path=resolve_path(args.adapter_path, ROOT),
        torch_dtype=args.torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )
    layers = find_transformer_layers(model)
    layer_indices = parse_layers(args.layers, len(layers))

    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "data_path": str(resolve_path(args.data_path, ROOT)),
                "model_path": args.model_path,
                "adapter_path": args.adapter_path,
                "styles": style_order,
                "layer_indices": layer_indices,
                "batch_size": args.batch_size,
                "max_examples_per_style": args.max_examples_per_style,
                "test_fraction": args.test_fraction,
                "split_seed": args.split_seed,
                "neutral_instruction": args.neutral_instruction,
                "prompt_format": args.prompt_format,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    dense_by_layer, labels, row_meta = collect_layerwise_response_means(
        model=model,
        tokenizer=tokenizer,
        rows=selected_rows,
        layer_indices=layer_indices,
        batch_size=args.batch_size,
        model_path=args.model_path,
        neutral_instruction=args.neutral_instruction,
        prompt_format=args.prompt_format,
    )
    train_idx, test_idx = make_stratified_split(
        labels=labels,
        style_order=style_order,
        test_fraction=args.test_fraction,
        seed=args.split_seed,
    )

    summaries = []
    for layer_idx in layer_indices:
        print(f"Scoring layer {layer_idx}")
        layer_vectors = dense_by_layer[layer_idx]
        layer_result = score_layer(
            vectors=layer_vectors,
            labels=labels,
            style_order=style_order,
            train_idx=train_idx,
            test_idx=test_idx,
            row_meta=row_meta,
        )
        layer_result["summary"]["layer"] = layer_idx
        summaries.append(layer_result["summary"])
        save_layer_outputs(output_dir, layer_idx, layer_result)

    summaries.sort(key=lambda row: row["layer"])
    write_summary_tables(output_dir, summaries)
    best_by_auc = max(summaries, key=lambda row: row["test_auc"])
    best_by_acc = max(summaries, key=lambda row: row["test_accuracy"])
    best_row = {
        "best_by_test_auc": best_by_auc,
        "best_by_test_accuracy": best_by_acc,
    }
    (output_dir / "best_layers.json").write_text(
        json.dumps(best_row, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\nBest by held-out AUC:", best_by_auc)
    print("Best by held-out accuracy:", best_by_acc)


def select_rows(rows: list[dict], style_order: list[str], max_examples_per_style: int) -> list[dict]:
    buckets = {style: [] for style in style_order}
    for row in rows:
        style = row.get("style")
        if style not in buckets:
            continue
        context = row.get("context")
        response = row.get("response")
        if not isinstance(context, str) or not context.strip():
            continue
        if not isinstance(response, str) or not response.strip():
            continue
        buckets[style].append(
            {
                "seed_id": row.get("seed_id"),
                "style": style,
                "context": context.strip(),
                "response": response.strip(),
            }
        )
    if max_examples_per_style > 0:
        for style in style_order:
            buckets[style] = buckets[style][:max_examples_per_style]

    selected = []
    for style in style_order:
        selected.extend(buckets[style])
    return selected


def parse_layers(raw: str, num_layers: int) -> list[int]:
    if raw.strip().lower() == "all":
        return list(range(num_layers))
    indices = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    invalid = [idx for idx in indices if idx < 0 or idx >= num_layers]
    if invalid:
        raise SystemExit(f"Layer indices out of range for model with {num_layers} layers: {invalid}")
    return indices


def tokenize_batch(
    tokenizer,
    prompts: list[str],
    responses: list[str],
    device: torch.device,
    prompt_format: str,
):
    full_texts = []
    prefix_lens = []
    for prompt, response in zip(prompts, responses):
        prefix_lens.append(
            prompt_prefix_token_length(
                tokenizer,
                prompt,
                prompt_format=prompt_format,
            )
        )
        full_texts.append(
            join_prompt_and_response(
                prompt,
                response,
                prompt_format=prompt_format,
            )
        )
    inputs = tokenizer(full_texts, return_tensors="pt", padding=True).to(device)
    return inputs, prefix_lens


def collect_layerwise_response_means(
    model,
    tokenizer,
    rows: list[dict],
    layer_indices: list[int],
    batch_size: int,
    model_path: str,
    neutral_instruction: str,
    prompt_format: str,
):
    device = model_input_device(model)
    layers = find_transformer_layers(model)
    dense_by_layer = {layer_idx: [] for layer_idx in layer_indices}
    labels = []
    row_meta = []

    for start in tqdm(range(0, len(rows), batch_size), desc="Capturing layerwise activations"):
        batch = rows[start : start + batch_size]
        prompts = [
            build_prompt(
                row["context"],
                neutral_instruction,
                tokenizer=tokenizer,
                model_path=model_path,
                prompt_format=prompt_format,
            )
            for row in batch
        ]
        responses = [row["response"] for row in batch]
        inputs, prefix_lens = tokenize_batch(
            tokenizer,
            prompts,
            responses,
            device,
            prompt_format,
        )
        attention_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        captured: dict[int, torch.Tensor] = {}

        def make_hook(layer_idx: int):
            def hook(_module, _inputs, output):
                hidden_states = output[0] if isinstance(output, tuple) else output
                per_example = []
                for row_idx, (prefix_len, seq_len) in enumerate(zip(prefix_lens, attention_lengths)):
                    if prefix_len >= seq_len:
                        raise RuntimeError(
                            f"Empty response token span for row {start + row_idx} at layer {layer_idx}."
                        )
                    response_hidden = hidden_states[row_idx, prefix_len:seq_len, :]
                    per_example.append(response_hidden.mean(dim=0))
                captured[layer_idx] = torch.stack(per_example, dim=0).detach().float().cpu()

            return hook

        handles = [
            layers[layer_idx].register_forward_hook(make_hook(layer_idx))
            for layer_idx in layer_indices
        ]
        with torch.no_grad():
            model(**inputs)
        for handle in handles:
            handle.remove()

        for layer_idx in layer_indices:
            if layer_idx not in captured:
                raise RuntimeError(f"Failed to capture hidden states at layer {layer_idx}.")
            dense_by_layer[layer_idx].append(captured[layer_idx])

        labels.extend(row["style"] for row in batch)
        row_meta.extend(
            {
                "seed_id": row.get("seed_id"),
                "style": row["style"],
                "context": row["context"],
                "response": row["response"],
            }
            for row in batch
        )

    dense_by_layer = {
        layer_idx: torch.cat(chunks, dim=0) for layer_idx, chunks in dense_by_layer.items()
    }
    return dense_by_layer, labels, row_meta


def make_stratified_split(
    labels: list[str],
    style_order: list[str],
    test_fraction: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    train_parts = []
    test_parts = []
    for style in style_order:
        style_indices = [idx for idx, label in enumerate(labels) if label == style]
        if len(style_indices) < 2:
            raise RuntimeError(f"Need at least two examples for style '{style}'.")
        perm = torch.randperm(len(style_indices), generator=generator).tolist()
        shuffled = [style_indices[idx] for idx in perm]
        n_test = max(1, int(round(len(shuffled) * test_fraction)))
        n_test = min(n_test, len(shuffled) - 1)
        test_parts.extend(shuffled[:n_test])
        train_parts.extend(shuffled[n_test:])
    return torch.tensor(sorted(train_parts)), torch.tensor(sorted(test_parts))


def score_layer(
    vectors: torch.Tensor,
    labels: list[str],
    style_order: list[str],
    train_idx: torch.Tensor,
    test_idx: torch.Tensor,
    row_meta: list[dict],
) -> dict:
    label_to_int = {style_order[0]: 0, style_order[1]: 1}
    y = torch.tensor([label_to_int[label] for label in labels], dtype=torch.long)

    train_x = vectors[train_idx]
    train_y = y[train_idx]
    test_x = vectors[test_idx]
    test_y = y[test_idx]

    neg_train = train_x[train_y == 0]
    pos_train = train_x[train_y == 1]
    if neg_train.numel() == 0 or pos_train.numel() == 0:
        raise RuntimeError("Both classes need at least one train example.")

    mu_neg = neg_train.mean(dim=0)
    mu_pos = pos_train.mean(dim=0)
    direction = mu_pos - mu_neg
    midpoint = (mu_pos + mu_neg) / 2.0

    train_scores = project_scores(train_x, midpoint, direction)
    test_scores = project_scores(test_x, midpoint, direction)
    train_pred = (train_scores > 0).long()
    test_pred = (test_scores > 0).long()

    neg_test_scores = test_scores[test_y == 0]
    pos_test_scores = test_scores[test_y == 1]

    neg_within = torch.norm(neg_train - mu_neg, dim=1).mean()
    pos_within = torch.norm(pos_train - mu_pos, dim=1).mean()
    mean_within = ((neg_within + pos_within) / 2.0).item()

    per_example = []
    for split_name, indices, scores, preds, gold in (
        ("train", train_idx, train_scores, train_pred, train_y),
        ("test", test_idx, test_scores, test_pred, test_y),
    ):
        for local_idx, global_idx in enumerate(indices.tolist()):
            per_example.append(
                {
                    "split": split_name,
                    "index": global_idx,
                    "seed_id": row_meta[global_idx].get("seed_id"),
                    "style": row_meta[global_idx]["style"],
                    "label": int(gold[local_idx].item()),
                    "score": float(scores[local_idx].item()),
                    "predicted_label": int(preds[local_idx].item()),
                    "correct": bool(preds[local_idx].item() == gold[local_idx].item()),
                }
            )

    summary = {
        "layer": None,
        "style_order": style_order,
        "train_count": int(train_idx.numel()),
        "test_count": int(test_idx.numel()),
        f"train_{style_order[0]}_count": int((train_y == 0).sum().item()),
        f"train_{style_order[1]}_count": int((train_y == 1).sum().item()),
        f"test_{style_order[0]}_count": int((test_y == 0).sum().item()),
        f"test_{style_order[1]}_count": int((test_y == 1).sum().item()),
        "delta_norm": float(torch.norm(direction).item()),
        "mean_within_norm": float(mean_within),
        "signal_to_noise": float(torch.norm(direction).item() / (mean_within + 1e-8)),
        "train_accuracy": float(accuracy(train_pred, train_y)),
        "test_accuracy": float(accuracy(test_pred, test_y)),
        "train_balanced_accuracy": float(balanced_accuracy(train_pred, train_y)),
        "test_balanced_accuracy": float(balanced_accuracy(test_pred, test_y)),
        "test_auc": float(binary_auc(pos_test_scores, neg_test_scores)),
        "test_score_gap": float(pos_test_scores.mean().item() - neg_test_scores.mean().item()),
    }
    return {
        "summary": summary,
        "mu_neg": mu_neg,
        "mu_pos": mu_pos,
        "direction": direction,
        "midpoint": midpoint,
        "train_scores": train_scores.cpu(),
        "test_scores": test_scores.cpu(),
        "train_idx": train_idx.cpu(),
        "test_idx": test_idx.cpu(),
        "per_example": per_example,
    }


def project_scores(vectors: torch.Tensor, midpoint: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    centered = vectors - midpoint.unsqueeze(0)
    return centered @ direction


def accuracy(pred: torch.Tensor, gold: torch.Tensor) -> float:
    return float((pred == gold).float().mean().item())


def balanced_accuracy(pred: torch.Tensor, gold: torch.Tensor) -> float:
    pieces = []
    for target in (0, 1):
        mask = gold == target
        if int(mask.sum().item()) == 0:
            continue
        pieces.append((pred[mask] == gold[mask]).float().mean())
    return float(torch.stack(pieces).mean().item()) if pieces else 0.0


def binary_auc(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> float:
    if pos_scores.numel() == 0 or neg_scores.numel() == 0:
        return 0.5
    comparisons = (pos_scores[:, None] > neg_scores[None, :]).float()
    ties = (pos_scores[:, None] == neg_scores[None, :]).float()
    return float((comparisons + 0.5 * ties).mean().item())


def save_layer_outputs(output_dir: Path, layer_idx: int, layer_result: dict) -> None:
    summary = dict(layer_result["summary"])
    summary["layer"] = layer_idx
    (output_dir / f"layer_{layer_idx}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    torch.save(
        {
            "layer": layer_idx,
            "style_order": summary["style_order"],
            "mu_neg": layer_result["mu_neg"],
            "mu_pos": layer_result["mu_pos"],
            "direction": layer_result["direction"],
            "midpoint": layer_result["midpoint"],
            "train_scores": layer_result["train_scores"],
            "test_scores": layer_result["test_scores"],
            "train_idx": layer_result["train_idx"],
            "test_idx": layer_result["test_idx"],
        },
        output_dir / f"layer_{layer_idx}_vectors.pt",
    )
    (output_dir / f"layer_{layer_idx}_per_example.json").write_text(
        json.dumps(layer_result["per_example"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_summary_tables(output_dir: Path, summaries: list[dict]) -> None:
    if not summaries:
        raise RuntimeError("No layer summaries to write.")
    fieldnames = [
        "layer",
        "train_count",
        "test_count",
        "delta_norm",
        "mean_within_norm",
        "signal_to_noise",
        "train_accuracy",
        "test_accuracy",
        "train_balanced_accuracy",
        "test_balanced_accuracy",
        "test_auc",
        "test_score_gap",
    ]
    with (output_dir / "layerwise_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow({key: row.get(key) for key in fieldnames})

    (output_dir / "layerwise_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
