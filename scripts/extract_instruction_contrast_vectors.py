#!/usr/bin/env python3
"""Extract persona vectors via contrastive INSTRUCTIONS at the last prompt token.

Uses the same position as LoReFT (last prompt token) but creates contrast
through different style instructions rather than different responses.

For each context:
  positive prompt = context + positive_instruction  (e.g. "antwortet hoffnungsvoll")
  negative prompt = context + negative_instruction  (e.g. "antwortet resigniert")

Captures the last-token hidden state at each layer for both prompt variants,
then computes diff-of-means. Outputs the same ``layer_<idx>_vectors.pt``
schema so downstream analysis scripts work unchanged.
"""

from __future__ import annotations

import argparse
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
    capture_last_token_hidden,
    find_transformer_layers,
    load_model_and_tokenizer,
    resolve_path,
    strip_annotation_labels,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract instruction-contrast persona vectors at last prompt token."
    )
    p.add_argument("--data-path", required=True,
                   help="JSONL with at least a 'context' field (paired or flat format).")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-path", default="Qwen/Qwen3-4B")
    p.add_argument("--layers", default="all",
                   help="Comma-separated layer indices or 'all'.")
    p.add_argument("--positive-instruction",
                   default="Der Klient antwortet hoffnungsvoll auf die letzte Aussage des Beraters.")
    p.add_argument("--negative-instruction",
                   default="Der Klient antwortet resigniert auf die letzte Aussage des Beraters.")
    p.add_argument("--positive-label", default="hopeful")
    p.add_argument("--negative-label", default="resigned")
    p.add_argument("--context-field", default="context")
    p.add_argument("--max-examples", type=int, default=200,
                   help="Max contexts to use (0 = all).")
    p.add_argument("--prompt-format", choices=("raw", "chat"), default="chat")
    p.add_argument("--torch-dtype", choices=("auto", "float16", "bfloat16", "float32"),
                   default="auto")
    p.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def parse_layers(raw: str, num_layers: int) -> list[int]:
    if raw.strip().lower() == "all":
        return list(range(num_layers))
    idx = sorted({int(x.strip()) for x in raw.split(",") if x.strip()})
    bad = [i for i in idx if i < 0 or i >= num_layers]
    if bad:
        raise SystemExit(f"Layer indices out of range (model has {num_layers}): {bad}")
    return idx


def extract_contexts(records: list[dict], context_field: str, max_examples: int) -> list[str]:
    seen = set()
    contexts = []
    for rec in records:
        ctx = rec.get(context_field)
        if not isinstance(ctx, str) or not ctx.strip():
            continue
        cleaned = strip_annotation_labels(ctx.strip())
        if cleaned in seen:
            continue
        seen.add(cleaned)
        contexts.append(cleaned)
    if max_examples > 0:
        contexts = contexts[:max_examples]
    return contexts


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(args.data_path)
    contexts = extract_contexts(records, args.context_field, args.max_examples)
    print(f"Using {len(contexts)} unique contexts")
    if len(contexts) < 2:
        raise SystemExit("Need at least 2 contexts.")

    tokenizer, model = load_model_and_tokenizer(
        args.model_path, adapter_path=None,
        torch_dtype=args.torch_dtype, trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    all_layers = find_transformer_layers(model)
    layer_indices = parse_layers(args.layers, len(all_layers))

    # Build prompt pairs
    pos_prompts = [
        build_prompt(ctx, args.positive_instruction,
                     tokenizer=tokenizer, model_path=args.model_path,
                     prompt_format=args.prompt_format)
        for ctx in contexts
    ]
    neg_prompts = [
        build_prompt(ctx, args.negative_instruction,
                     tokenizer=tokenizer, model_path=args.model_path,
                     prompt_format=args.prompt_format)
        for ctx in contexts
    ]

    # Capture per layer
    summary_rows = []
    for li in tqdm(layer_indices, desc="Layers"):
        pos_hiddens = []
        neg_hiddens = []
        for pp, np_ in zip(pos_prompts, neg_prompts):
            pos_hiddens.append(capture_last_token_hidden(model, tokenizer, pp, li))
            neg_hiddens.append(capture_last_token_hidden(model, tokenizer, np_, li))

        pos_stack = torch.stack(pos_hiddens, dim=0).float()  # (N, hidden)
        neg_stack = torch.stack(neg_hiddens, dim=0).float()

        mu_pos = pos_stack.mean(dim=0)
        mu_neg = neg_stack.mean(dim=0)
        direction = mu_pos - mu_neg
        midpoint = (mu_pos + mu_neg) / 2.0

        delta_norm = float(direction.norm().item())
        within_pos = (pos_stack - mu_pos).norm(dim=1).mean()
        within_neg = (neg_stack - mu_neg).norm(dim=1).mean()
        within = float(((within_pos + within_neg) / 2).item())
        snr = delta_norm / (within + 1e-8)

        torch.save(
            {
                "layer": li,
                "style_order": [args.negative_label, args.positive_label],
                "mu_neg": mu_neg.cpu(),
                "mu_pos": mu_pos.cpu(),
                "direction": direction.cpu(),
                "midpoint": midpoint.cpu(),
                "extraction_method": "instruction_contrast",
                "positive_instruction": args.positive_instruction,
                "negative_instruction": args.negative_instruction,
                "n_contexts": len(contexts),
            },
            out_dir / f"layer_{li}_vectors.pt",
        )
        summary_rows.append({
            "layer": li,
            "extraction_method": "instruction_contrast",
            "delta_norm": delta_norm,
            "mean_within_norm": within,
            "signal_to_noise": snr,
        })
        print(f"  layer={li:3d} delta={delta_norm:.4f} within={within:.4f} snr={snr:.4f}")

    (out_dir / "layerwise_summary.json").write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (out_dir / "run_config.json").write_text(
        json.dumps({
            "data_path": str(resolve_path(args.data_path, ROOT)),
            "model_path": args.model_path,
            "positive_instruction": args.positive_instruction,
            "negative_instruction": args.negative_instruction,
            "positive_label": args.positive_label,
            "negative_label": args.negative_label,
            "layer_indices": layer_indices,
            "n_contexts": len(contexts),
            "prompt_format": args.prompt_format,
            "extraction_method": "instruction_contrast",
        }, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"\nWrote {len(summary_rows)} layer vectors to {out_dir}")


if __name__ == "__main__":
    main()
