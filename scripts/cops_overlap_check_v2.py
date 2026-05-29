#!/usr/bin/env python3
"""COPS overlap check v2 — probes at RESPONSE-TOKEN positions (matches v_open).

Fixes the methodological flaw of v1: v_open is computed at the last token of
(context + response) for paired dense steering. v1 trained probes at the last
token of context-only, which is a different position.

This version:
  1. Loads paired open/defensive responses (persona_pairs_openness_gpt54mini_1000).
  2. For each pair, extracts hidden states at the last token of (context + response)
     for BOTH responses at chosen layers.
  3. Recomputes v_open_fresh = mean(h_open) - mean(h_defensive) at each layer.
     Sanity-checks against stored v_open.
  4. Trains content probes on the 2000 response-last-token hidden states, using
     hauptanliegen-derived binary labels.
  5. Reports cosine similarity + projection ratio between v_open_fresh and probes.
"""

from __future__ import annotations

import argparse
import json
import re
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


PROBE_KEYWORDS = {
    "mentions_son": ["Sohn", "Max"],
    "mentions_cannabis": ["Cannabis", "Kiffen", "kiffen", "Joint", "Marihuana"],
    "mentions_husband": ["Mann", "Ehemann", "Partner"],
    "mentions_school": ["Schule", "Noten"],
    "mentions_separation": ["Trennung", "verschwunden", "abgehauen"],
    "mentions_addiction": ["Drogen", "Entzug", "Konsum", "Abhängig"],
}


# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------

def find_transformer_layers(model: nn.Module) -> nn.ModuleList:
    visited: set[int] = set()
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


def capture_last_token_hidden_multi(
    model: nn.Module, tokenizer, prompt: str, layer_indices: list[int],
) -> dict[int, torch.Tensor]:
    """Forward pass once, capture hidden state at last token for ALL requested layers."""
    layers = find_transformer_layers(model)
    captured: dict[int, list[torch.Tensor]] = {i: [] for i in layer_indices}
    handles = []

    def make_hook(idx):
        def hook(_m, _i, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[idx].append(hidden[:, -1, :].detach().float().cpu())
        return hook

    for idx in layer_indices:
        handles.append(layers[idx].register_forward_hook(make_hook(idx)))

    device = model.get_input_embeddings().weight.device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        model(**inputs)
    for h in handles:
        h.remove()

    return {idx: captured[idx][-1].squeeze(0) for idx in layer_indices}


# ---------------------------------------------------------------------------
# Probe utilities
# ---------------------------------------------------------------------------

def fit_binary_probe(X: np.ndarray, y: np.ndarray, seed: int = 42):
    if len(np.unique(y)) < 2:
        return np.zeros(X.shape[1]), float("nan"), float("nan")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    clf = LogisticRegression(C=0.1, max_iter=500, random_state=seed, solver="liblinear")
    clf.fit(X_tr, y_tr)
    w = clf.coef_.flatten()
    w = w / (np.linalg.norm(w) + 1e-12)
    return w, clf.score(X_tr, y_tr), clf.score(X_te, y_te)


def fit_multiclass_probe(X: np.ndarray, y: np.ndarray, seed: int = 42):
    if len(np.unique(y)) < 2:
        return np.zeros((1, X.shape[1])), float("nan"), float("nan")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    base = LogisticRegression(C=0.1, max_iter=500, random_state=seed, solver="liblinear")
    clf = OneVsRestClassifier(base, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    W = np.stack([e.coef_.flatten() for e in clf.estimators_], axis=0)
    W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
    return W, clf.score(X_tr, y_tr), clf.score(X_te, y_te)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def projection_ratio(v: np.ndarray, U: np.ndarray) -> float:
    if U.ndim == 1:
        U = U[None, :]
    Q, _ = np.linalg.qr(U.T)
    v_proj = Q @ (Q.T @ v)
    return float(np.linalg.norm(v_proj) / (np.linalg.norm(v) + 1e-12))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pairs(path: Path, n: int, seed: int) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:n]


def load_hauptanliegen_map(ncp_path: Path) -> dict[int, str]:
    """Map original_index → hauptanliegen (parsed from personality_condition)."""
    mapping: dict[int, str] = {}
    with ncp_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            idx = r.get("original_index")
            pc = r.get("personality_condition", "")
            if idx is None or not pc:
                continue
            m = re.search(r"Hauptanliegen:\s*(.+?)(?=\n(?:Nebenanliegen|Sprachliche|Emotionale|Prinzipien):|$)", pc, re.DOTALL)
            if m:
                mapping[int(idx)] = m.group(1).strip()
    return mapping


def make_binary_label(hauptanliegen: str, keywords: list[str]) -> int:
    return int(any(kw in hauptanliegen for kw in keywords))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", default="Qwen/Qwen3-4B")
    p.add_argument(
        "--pairs-path",
        default=str(ROOT / "outputs/metrics/persona_pairs_openness_gpt54mini_1000.jsonl"),
    )
    p.add_argument(
        "--ncp-path",
        default=str(ROOT / "data/processed/ncp_eval.jsonl"),
        help="Used to map seed_record_index → hauptanliegen.",
    )
    p.add_argument(
        "--v-open-dir",
        default=str(ROOT / "outputs/metrics/layerwise_separability_openness_qwen3_4b_gpt54mini_1000"),
    )
    p.add_argument("--layers", default="1,5,9,13,18")
    p.add_argument("--n-pairs", type=int, default=200)
    p.add_argument(
        "--output-path",
        default=str(ROOT / "outputs/metrics/cops_overlap_check_v2.json"),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--torch-dtype", default="bfloat16", choices=("float32", "float16", "bfloat16"))
    p.add_argument("--force-extract", action="store_true")
    return p.parse_args()


def resolve_torch_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def load_stored_v_open(v_open_dir: Path, layer: int) -> np.ndarray:
    pt_path = v_open_dir / f"layer_{layer}_vectors.pt"
    if not pt_path.exists():
        return None
    blob = torch.load(pt_path, map_location="cpu", weights_only=False)
    d = torch.as_tensor(blob["direction"]).detach().float().numpy()
    return d / (np.linalg.norm(d) + 1e-12)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    layers = [int(x) for x in args.layers.split(",")]
    pairs = load_pairs(Path(args.pairs_path), args.n_pairs, args.seed)
    print(f"Loaded {len(pairs)} paired records from {args.pairs_path}", flush=True)

    print("Loading hauptanliegen map from ncp_eval.jsonl...", flush=True)
    ha_map = load_hauptanliegen_map(Path(args.ncp_path))
    print(f"  mapped {len(ha_map)} original_index → hauptanliegen", flush=True)

    # Filter pairs to those where we have hauptanliegen
    pairs = [p for p in pairs if p.get("seed_record_index") in ha_map]
    print(f"After hauptanliegen filter: {len(pairs)} pairs", flush=True)

    # Cache — include pairs path stem so case-blind/case-aware don't collide
    cache_dir = Path(args.output_path).parent / "cops_hidden_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    pairs_stem = Path(args.pairs_path).stem
    cache_file = cache_dir / f"v2_{pairs_stem}_n{len(pairs)}_seed{args.seed}_layers{args.layers.replace(',', '-')}.npz"

    if cache_file.exists() and not args.force_extract:
        print(f"Loading cached hidden states from {cache_file}", flush=True)
        npz = np.load(cache_file)
        h_open = {l: npz[f"open_layer_{l}"] for l in layers}
        h_def = {l: npz[f"def_layer_{l}"] for l in layers}
    else:
        print(f"\nLoading model {args.model_path}...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=resolve_torch_dtype(args.torch_dtype),
            trust_remote_code=True,
        )
        if torch.cuda.is_available():
            model = model.to("cuda")
        model.eval()
        print(f"Model on {next(model.parameters()).device}", flush=True)

        h_open_lists: dict[int, list[np.ndarray]] = {l: [] for l in layers}
        h_def_lists: dict[int, list[np.ndarray]] = {l: [] for l in layers}

        print(f"\nExtracting hidden states at response-last-token for layers {layers}...", flush=True)
        for i, p in enumerate(pairs):
            prompt_open = f"{p['context']}\nKlient: {p['open_response']}"
            prompt_def = f"{p['context']}\nKlient: {p['defensive_response']}"
            captured_open = capture_last_token_hidden_multi(model, tokenizer, prompt_open, layers)
            captured_def = capture_last_token_hidden_multi(model, tokenizer, prompt_def, layers)
            for l in layers:
                h_open_lists[l].append(captured_open[l].numpy())
                h_def_lists[l].append(captured_def[l].numpy())
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(pairs)} pairs", flush=True)

        h_open = {l: np.stack(h_open_lists[l], axis=0) for l in layers}
        h_def = {l: np.stack(h_def_lists[l], axis=0) for l in layers}

        np.savez_compressed(
            cache_file,
            **{f"open_layer_{l}": h_open[l] for l in layers},
            **{f"def_layer_{l}": h_def[l] for l in layers},
        )
        print(f"Cached to {cache_file}", flush=True)
        del model
        torch.cuda.empty_cache()

    # ---- Analysis ----
    print("\n" + "=" * 70, flush=True)

    hauptanliegen_per_pair = [ha_map[p["seed_record_index"]] for p in pairs]
    persona_per_pair = [p["persona_name"] for p in pairs]
    personas = sorted(set(persona_per_pair))
    persona_to_id = {n: i for i, n in enumerate(personas)}

    results = {
        "n_pairs": len(pairs),
        "model_path": args.model_path,
        "layers": {},
    }

    for layer in layers:
        print(f"\n=== Layer {layer} ===", flush=True)

        # Recompute v_open at the exact position
        v_open_fresh = h_open[layer].mean(axis=0) - h_def[layer].mean(axis=0)
        v_open_fresh = v_open_fresh / (np.linalg.norm(v_open_fresh) + 1e-12)

        # Sanity: compare with stored v_open
        stored = load_stored_v_open(Path(args.v_open_dir), layer)
        sanity_cos = cosine(v_open_fresh, stored) if stored is not None else None
        print(f"  v_open_fresh vs stored v_open: cos = {sanity_cos}", flush=True)

        # Stack all response hidden states (open + def) for content-probe training.
        # Each row is one response; labels come from the underlying case's hauptanliegen.
        X = np.concatenate([h_open[layer], h_def[layer]], axis=0)  # [2N, d]
        ha_doubled = hauptanliegen_per_pair + hauptanliegen_per_pair
        personas_doubled = persona_per_pair + persona_per_pair

        layer_result = {
            "sanity_cos_fresh_vs_stored": sanity_cos,
            "v_open_fresh_norm": float(np.linalg.norm(h_open[layer].mean(axis=0) - h_def[layer].mean(axis=0))),
            "probes": {},
        }

        # Binary content probes
        for probe_name, kws in PROBE_KEYWORDS.items():
            y = np.array([make_binary_label(ha, kws) for ha in ha_doubled])
            w, tr_acc, te_acc = fit_binary_probe(X, y, args.seed)
            cos = cosine(v_open_fresh, w)
            layer_result["probes"][probe_name] = {
                "test_acc": te_acc, "train_acc": tr_acc,
                "cosine_with_v_open": cos, "abs_cosine": abs(cos),
                "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
            }
            print(f"  {probe_name}: test_acc={te_acc:.3f}  cos(w, v_open)={cos:+.4f}  |cos|={abs(cos):.4f}", flush=True)

        # Combined binary subspace
        dirs = []
        for probe_name, kws in PROBE_KEYWORDS.items():
            y = np.array([make_binary_label(ha, kws) for ha in ha_doubled])
            w, _, _ = fit_binary_probe(X, y, args.seed)
            if np.linalg.norm(w) > 1e-6:
                dirs.append(w)
        if dirs:
            U = np.stack(dirs, axis=0)
            proj = projection_ratio(v_open_fresh, U)
            layer_result["combined_content_subspace"] = {
                "n_dirs": len(dirs),
                "projection_ratio_of_v_open": proj,
            }
            print(f"  combined binary → proj ratio of v_open: {proj:.4f}", flush=True)

        # Persona multiclass probe
        y_persona = np.array([persona_to_id[n] for n in personas_doubled])
        W_persona, tr_acc, te_acc = fit_multiclass_probe(X, y_persona, args.seed)
        proj_persona = projection_ratio(v_open_fresh, W_persona)
        layer_result["persona_probe"] = {
            "train_acc": tr_acc, "test_acc": te_acc,
            "n_classes": W_persona.shape[0],
            "projection_ratio_of_v_open": proj_persona,
        }
        print(f"  persona(multiclass): test_acc={te_acc:.3f}  proj_ratio={proj_persona:.4f}", flush=True)

        results["layers"][str(layer)] = layer_result

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
