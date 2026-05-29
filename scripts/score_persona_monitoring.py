#!/usr/bin/env python3
"""Score steering outputs by projection onto a learned persona vector.

The vector artifact is expected to come from `compute_layerwise_separability.py`
and contain at least:

- layer
- style_order
- direction
- midpoint
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest, wilcoxon
from tqdm import tqdm

from compute_layerwise_separability import find_transformer_layers, load_model_and_tokenizer, model_input_device
from steering_vector_experiment import join_prompt_and_response, prompt_prefix_token_length


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project base/steered responses onto a learned persona vector."
    )
    parser.add_argument(
        "--vector-path",
        required=True,
        help="Path to a layerwise-separability vector artifact (.pt).",
    )
    parser.add_argument(
        "--score-paths",
        required=True,
        nargs="+",
        help="One or more JSONL files from steering runs to score.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for projection-monitor summaries.",
    )
    parser.add_argument(
        "--model-path",
        default="Qwen/Qwen3-4B",
        help="Base model name or local path.",
    )
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="Optional adapter to load on top of the base model.",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
        help="Torch dtype forwarded to the shared model loader.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forwarded to the shared model loader.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for response re-encoding.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=5000,
        help="Bootstrap samples for mean-delta confidence intervals.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vector_artifact = torch.load(Path(args.vector_path), map_location="cpu")
    if not isinstance(vector_artifact, dict):
        raise SystemExit(
            "Projection monitoring requires a dict artifact from compute_layerwise_separability.py."
        )
    required = {"layer", "style_order", "direction", "midpoint"}
    missing = sorted(required - set(vector_artifact.keys()))
    if missing:
        raise SystemExit(f"Vector artifact is missing required keys: {missing}")

    style_order = list(vector_artifact["style_order"])
    if len(style_order) != 2:
        raise SystemExit(f"Expected binary style_order, got: {style_order}")
    layer_idx = int(vector_artifact["layer"])
    midpoint = vector_artifact["midpoint"].float().cpu()
    direction = vector_artifact["direction"].float().cpu()
    direction = direction / max(float(direction.norm().item()), 1e-12)

    tokenizer, model = load_model_and_tokenizer(
        args.model_path,
        adapter_path=args.adapter_path,
        torch_dtype=args.torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    device = model_input_device(model)
    layers = find_transformer_layers(model)
    if layer_idx < 0 or layer_idx >= len(layers):
        raise SystemExit(
            f"Layer {layer_idx} not available for model with {len(layers)} transformer layers."
        )

    all_summary_rows = []
    for score_path_raw in args.score_paths:
        score_path = Path(score_path_raw)
        rows = load_jsonl(score_path)
        scored_rows = score_rows(
            rows=rows,
            tokenizer=tokenizer,
            model=model,
            layer_module=layers[layer_idx],
            layer_idx=layer_idx,
            midpoint=midpoint,
            direction=direction,
            batch_size=args.batch_size,
        )
        summary_rows = summarize_rows(
            scored_rows,
            style_order=style_order,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        stem = score_path.stem
        (output_dir / f"{stem}.summary.json").write_text(
            json.dumps(summary_rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with (output_dir / f"{stem}.summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        (output_dir / f"{stem}.scored_rows.json").write_text(
            json.dumps(scored_rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        for row in summary_rows:
            enriched = dict(row)
            enriched["source_file"] = str(score_path.resolve())
            all_summary_rows.append(enriched)

    if all_summary_rows:
        with (output_dir / "all_runs_summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_summary_rows)

    print("\nTop projection-monitor configs by mean delta:")
    for row in sorted(all_summary_rows, key=lambda item: item["mean_delta_vs_base"], reverse=True)[:10]:
        print(
            row["source_file"],
            row["config_id"],
            "mean_delta",
            round(row["mean_delta_vs_base"], 4),
            "wilcoxon_p",
            row["wilcoxon_p"],
        )


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def score_rows(
    rows: list[dict],
    tokenizer,
    model,
    layer_module,
    layer_idx: int,
    midpoint: torch.Tensor,
    direction: torch.Tensor,
    batch_size: int,
) -> list[dict]:
    device = model_input_device(model)
    scored_rows: list[dict] = []
    skipped_rows = 0
    for start in tqdm(range(0, len(rows), batch_size), desc=f"Projecting layer {layer_idx}"):
        batch = rows[start : start + batch_size]
        prompts = [resolve_prompt(row) for row in batch]
        full_texts = []
        prefix_lens = []
        for row, prompt in zip(batch, prompts):
            prompt_format = row.get("prompt_format", "chat")
            prefix_lens.append(
                prompt_prefix_token_length(tokenizer, prompt, prompt_format=prompt_format)
            )
            full_texts.append(
                join_prompt_and_response(prompt, row["response"], prompt_format=prompt_format)
            )

        inputs = tokenizer(full_texts, return_tensors="pt", padding=True).to(device)
        attention_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        captured: dict[str, torch.Tensor] = {}

        def hook(_module, _inputs, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            projections = []
            for row_idx, (prefix_len, seq_len) in enumerate(zip(prefix_lens, attention_lengths)):
                if prefix_len >= seq_len:
                    projections.append(float("nan"))
                    continue
                response_hidden = hidden_states[row_idx, prefix_len:seq_len, :].mean(dim=0).detach().float().cpu()
                projection = torch.dot(response_hidden - midpoint, direction).item()
                projections.append(float(projection))
            captured["projection"] = torch.tensor(projections, dtype=torch.float32)

        handle = layer_module.register_forward_hook(hook)
        with torch.no_grad():
            model(**inputs)
        handle.remove()

        if "projection" not in captured:
            raise RuntimeError(f"Failed to capture layerwise activations at layer {layer_idx}.")

        for row, projection in zip(batch, captured["projection"].tolist()):
            if not np.isfinite(projection):
                skipped_rows += 1
                continue
            enriched = dict(row)
            enriched["projection_score"] = float(projection)
            scored_rows.append(enriched)

    if skipped_rows:
        print(f"Skipped {skipped_rows} rows with empty response token spans.")

    return scored_rows


def resolve_prompt(row: dict) -> str:
    prompt = row.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise SystemExit(
            "Projection monitoring currently requires stored prompt strings in the run outputs."
        )
    return prompt


def summarize_rows(
    rows: list[dict],
    style_order: list[str],
    bootstrap_samples: int,
    seed: int,
) -> list[dict]:
    by_example = defaultdict(dict)
    for row in rows:
        example_index = row["example_index"]
        config_key = "base" if row["mode"] == "base" else build_config_key(row)
        by_example[example_index][config_key] = row

    config_keys = sorted({key for item in by_example.values() for key in item if key != "base"})
    summary_rows = []
    for config_key in config_keys:
        base_scores = []
        steered_scores = []
        deltas = []
        wins = 0
        losses = 0
        ties = 0

        for item in by_example.values():
            if "base" not in item or config_key not in item:
                continue
            base_score = float(item["base"]["projection_score"])
            steered_score = float(item[config_key]["projection_score"])
            delta = steered_score - base_score
            base_scores.append(base_score)
            steered_scores.append(steered_score)
            deltas.append(delta)
            if delta > 1e-12:
                wins += 1
            elif delta < -1e-12:
                losses += 1
            else:
                ties += 1

        deltas_arr = np.array(deltas, dtype=float)
        mean_delta = float(deltas_arr.mean()) if len(deltas_arr) else 0.0
        median_delta = float(np.median(deltas_arr)) if len(deltas_arr) else 0.0
        ci_low, ci_high = bootstrap_mean_ci(deltas_arr, bootstrap_samples, seed)
        sign_p = sign_test_pvalue(wins=wins, losses=losses)
        try:
            wilcoxon_stat, wilcoxon_p = wilcoxon(
                deltas_arr,
                alternative="greater",
                zero_method="wilcox",
            )
            wilcoxon_stat = float(wilcoxon_stat)
            wilcoxon_p = float(wilcoxon_p)
        except ValueError:
            wilcoxon_stat = 0.0
            wilcoxon_p = 1.0

        summary_rows.append(
            {
                "axis_name": f"{style_order[0]}_vs_{style_order[1]}",
                "negative_label": style_order[0],
                "positive_label": style_order[1],
                "config_id": config_key,
                "n_examples": len(deltas),
                "mean_base_projection": float(np.mean(base_scores)) if base_scores else 0.0,
                "mean_projection": float(np.mean(steered_scores)) if steered_scores else 0.0,
                "mean_delta_vs_base": mean_delta,
                "median_delta_vs_base": median_delta,
                "delta_ci_low": ci_low,
                "delta_ci_high": ci_high,
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "sign_test_p": sign_p,
                "wilcoxon_stat": wilcoxon_stat,
                "wilcoxon_p": wilcoxon_p,
            }
        )

    summary_rows.sort(key=lambda row: row["mean_delta_vs_base"], reverse=True)
    return summary_rows


def build_config_key(row: dict) -> str:
    explicit = row.get("config_id")
    if explicit:
        return str(explicit)
    if row.get("steering_mode") == "weighted":
        layers = row.get("layers", [])
        return f"weighted_{'-'.join(str(layer) for layer in layers)}_a{row['alpha']}"
    if row.get("steering_mode") == "gaussian":
        layers = row.get("layers", [])
        return f"gaussian_{'-'.join(str(layer) for layer in layers)}_a{row['alpha']}"
    layer_idx = row.get("layer_idx")
    alpha = row.get("alpha")
    if layer_idx is None or alpha is None:
        return "unknown"
    return f"layer{layer_idx}_a{alpha}"


def bootstrap_mean_ci(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)
    n = values.size
    for idx in range(samples):
        sample = values[rng.integers(0, n, size=n)]
        means[idx] = sample.mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def sign_test_pvalue(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    return float(binomtest(wins, n=n, p=0.5, alternative="greater").pvalue)


if __name__ == "__main__":
    main()
