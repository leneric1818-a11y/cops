#!/usr/bin/env python3
"""Extract persona directions at specific token positions (not response-token mean).

Rationale
---------
``compute_layerwise_separability.py`` extracts persona vectors by averaging over
response-token hidden states. LoReFT, however, intervenes at the **last prompt
token** (before the response is generated). Comparing geometry between those
two artifacts is unfair — different positions in the forward pass encode
concepts differently.

This script produces persona vectors at a **chosen position**:
  * ``prompt_end``        — last token of the prompt (LoReFT's intervention site)
  * ``first_response``    — first token of the response
  * ``last_response``     — last token of the response (EOS / end-of-generation)

Inputs match ``compute_layerwise_separability.py``: a flat JSONL with
``context``, ``response``, ``style`` fields. Outputs the same
``layer_<idx>_vectors.pt`` schema so downstream scripts
(``analyze_reft_persona_alignment.py``, ``score_persona_monitoring.py``)
work unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

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


POSITIONS = ("prompt_end", "first_response", "last_response")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract persona vectors at a specific token position."
    )
    parser.add_argument("--data-path", required=True, help="Flat JSONL with context/response/style.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", default="Qwen/Qwen3-4B")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--layers", default="all", help="Comma-separated layer indices, or 'all'.")
    parser.add_argument("--styles", default="resigned,hopeful",
                        help="Two comma-separated style labels (negative,positive).")
    parser.add_argument("--position", choices=POSITIONS, default="prompt_end",
                        help="Token position from which to capture the hidden state.")
    parser.add_argument("--max-examples-per-style", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--neutral-instruction",
                        default="Der Klient antwortet auf die letzte Aussage des Beraters.")
    parser.add_argument("--prompt-format", choices=("raw", "chat"), default="chat")
    parser.add_argument("--torch-dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def select_rows(rows: list[dict], styles: list[str], cap: int) -> list[dict]:
    buckets: dict[str, list[dict]] = {s: [] for s in styles}
    for row in rows:
        if row.get("style") not in buckets:
            continue
        if not isinstance(row.get("context"), str) or not row["context"].strip():
            continue
        if not isinstance(row.get("response"), str) or not row["response"].strip():
            continue
        buckets[row["style"]].append(
            {"seed_id": row.get("seed_id"), "style": row["style"],
             "context": row["context"].strip(), "response": row["response"].strip()}
        )
    if cap > 0:
        for s in styles:
            buckets[s] = buckets[s][:cap]
    out: list[dict] = []
    for s in styles:
        out.extend(buckets[s])
    return out


def parse_layers(raw: str, num_layers: int) -> list[int]:
    if raw.strip().lower() == "all":
        return list(range(num_layers))
    idx = sorted({int(x.strip()) for x in raw.split(",") if x.strip()})
    bad = [i for i in idx if i < 0 or i >= num_layers]
    if bad:
        raise SystemExit(f"Layer indices out of range (model has {num_layers}): {bad}")
    return idx


def compute_position_vectors(
    *,
    model,
    tokenizer,
    rows: list[dict],
    layer_indices: list[int],
    batch_size: int,
    model_path: str,
    neutral_instruction: str,
    prompt_format: str,
    position: str,
):
    layers = find_transformer_layers(model)
    device = model_input_device(model)

    # Pre-compute prompts and full texts.
    prompts = [
        build_prompt(row["context"], neutral_instruction,
                     tokenizer=tokenizer, model_path=model_path, prompt_format=prompt_format)
        for row in rows
    ]
    full_texts = [join_prompt_and_response(p, r["response"], prompt_format=prompt_format)
                  for p, r in zip(prompts, rows)]

    # Per-row prompt-prefix lengths (in tokens of the full text).
    prefix_lens = [
        prompt_prefix_token_length(tokenizer, p, prompt_format=prompt_format)
        for p in prompts
    ]

    hiddens_by_layer: dict[int, list[torch.Tensor]] = {l: [] for l in layer_indices}
    labels: list[str] = []
    row_meta: list[dict] = []

    for start in tqdm(range(0, len(rows), batch_size), desc=f"Capture@{position}"):
        batch = rows[start: start + batch_size]
        batch_texts = full_texts[start: start + batch_size]
        batch_prefixes = prefix_lens[start: start + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True).to(device)
        attention = inputs["attention_mask"].sum(dim=1).tolist()

        # Resolve target token index per row based on position.
        target_indices: list[int] = []
        for prefix_len, seq_len in zip(batch_prefixes, attention):
            if position == "prompt_end":
                idx = max(prefix_len - 1, 0)
            elif position == "first_response":
                idx = min(prefix_len, seq_len - 1)
            elif position == "last_response":
                idx = seq_len - 1
            else:
                raise ValueError(position)
            target_indices.append(int(idx))

        captured: dict[int, torch.Tensor] = {}

        def make_hook(layer_idx: int):
            def hook(_module, _inputs, output):
                hs = output[0] if isinstance(output, tuple) else output
                rows_hiddens = []
                for row_idx, tok_idx in enumerate(target_indices):
                    rows_hiddens.append(hs[row_idx, tok_idx, :])
                captured[layer_idx] = torch.stack(rows_hiddens, dim=0).detach().float().cpu()
            return hook

        handles = [layers[li].register_forward_hook(make_hook(li)) for li in layer_indices]
        with torch.no_grad():
            model(**inputs)
        for h in handles:
            h.remove()

        for li in layer_indices:
            hiddens_by_layer[li].append(captured[li])

        labels.extend(row["style"] for row in batch)
        row_meta.extend(
            {"seed_id": r.get("seed_id"), "style": r["style"],
             "context": r["context"], "response": r["response"]}
            for r in batch
        )

    hiddens_by_layer = {l: torch.cat(chunks, dim=0) for l, chunks in hiddens_by_layer.items()}
    return hiddens_by_layer, labels, row_meta


def main() -> None:
    args = parse_args()
    styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    if len(styles) != 2:
        raise SystemExit("--styles must have exactly two comma-separated labels.")
    out_dir = resolve_path(args.output_dir, ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_records(args.data_path)
    selected = select_rows(rows, styles, args.max_examples_per_style)
    counts = {s: sum(1 for r in selected if r["style"] == s) for s in styles}
    print(f"Selected rows by style: {counts}")
    if min(counts.values()) < 2:
        raise SystemExit("Need at least two rows per style.")

    tokenizer, model = load_model_and_tokenizer(
        args.model_path,
        adapter_path=resolve_path(args.adapter_path, ROOT),
        torch_dtype=args.torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )
    layers = find_transformer_layers(model)
    layer_indices = parse_layers(args.layers, len(layers))

    hiddens_by_layer, labels, row_meta = compute_position_vectors(
        model=model, tokenizer=tokenizer, rows=selected,
        layer_indices=layer_indices, batch_size=args.batch_size,
        model_path=args.model_path, neutral_instruction=args.neutral_instruction,
        prompt_format=args.prompt_format, position=args.position,
    )

    summary_rows = []
    for li in layer_indices:
        vectors = hiddens_by_layer[li]
        y = torch.tensor([0 if s == styles[0] else 1 for s in labels])
        neg = vectors[y == 0]
        pos = vectors[y == 1]
        mu_neg = neg.mean(dim=0)
        mu_pos = pos.mean(dim=0)
        direction = mu_pos - mu_neg
        midpoint = (mu_pos + mu_neg) / 2.0
        delta_norm = float(direction.norm().item())
        within = ((neg - mu_neg).norm(dim=1).mean() + (pos - mu_pos).norm(dim=1).mean()) / 2
        snr = delta_norm / (float(within.item()) + 1e-8)

        torch.save(
            {
                "layer": li,
                "style_order": styles,
                "mu_neg": mu_neg,
                "mu_pos": mu_pos,
                "direction": direction,
                "midpoint": midpoint,
                "position": args.position,
            },
            out_dir / f"layer_{li}_vectors.pt",
        )
        summary_rows.append({
            "layer": li,
            "position": args.position,
            "delta_norm": delta_norm,
            "mean_within_norm": float(within.item()),
            "signal_to_noise": snr,
        })
        print(f"layer={li:3d} position={args.position} "
              f"delta={delta_norm:.3f} within={float(within.item()):.3f} snr={snr:.3f}")

    (out_dir / "layerwise_summary.json").write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "data_path": str(resolve_path(args.data_path, ROOT)),
                "model_path": args.model_path,
                "adapter_path": args.adapter_path,
                "styles": styles,
                "position": args.position,
                "layer_indices": layer_indices,
                "neutral_instruction": args.neutral_instruction,
                "prompt_format": args.prompt_format,
                "max_examples_per_style": args.max_examples_per_style,
                "batch_size": args.batch_size,
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(summary_rows)} layer vectors to {out_dir}")


if __name__ == "__main__":
    main()
