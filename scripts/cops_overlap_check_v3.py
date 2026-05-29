#!/usr/bin/env python3
"""COPS overlap check v3 — uses response-MEAN pooling (matches deployed v_open).

The paper's paired dense v_open = mean over response tokens (not last token).
This script recomputes hidden states with response-mean pooling so the sanity
check cos(fresh, stored) should be ~1.0 — confirming our projections reflect
the actually deployed vector.

Uses the same analysis pipeline as v2: persona subspace projection + binary
content probes.
"""

from __future__ import annotations

import argparse
import json
import random
import re
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


def find_transformer_layers(model):
    visited, queue = set(), [model]
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
    raise RuntimeError("Transformer layers not found.")


def capture_response_mean_hidden_multi(
    model, tokenizer, prompt_prefix: str, response_text: str, layer_indices: list[int]
) -> dict[int, torch.Tensor]:
    """Forward pass once; capture response-token MEAN hidden state per layer.

    prompt_prefix = everything before the response (e.g. context + "Klient: ")
    response_text = the open or defensive response text
    """
    layers = find_transformer_layers(model)
    captured: dict[int, list[torch.Tensor]] = {i: [] for i in layer_indices}
    handles = []

    def make_hook(idx):
        def hook(_m, _i, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[idx].append(hidden.detach())
        return hook

    for idx in layer_indices:
        handles.append(layers[idx].register_forward_hook(make_hook(idx)))

    device = model.get_input_embeddings().weight.device

    # Tokenize prefix alone to know where response starts
    prefix_ids = tokenizer(prompt_prefix, return_tensors="pt", add_special_tokens=True).input_ids
    prefix_len = int(prefix_ids.shape[1])

    full_text = prompt_prefix + response_text
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048).to(device)
    full_len = int(inputs["attention_mask"].sum().item())

    with torch.no_grad():
        model(**inputs)
    for h in handles:
        h.remove()

    if prefix_len >= full_len:
        raise RuntimeError(f"Empty response span: prefix_len={prefix_len} full_len={full_len}")

    out = {}
    for idx in layer_indices:
        hid = captured[idx][-1]  # [1, T, d]
        resp_hid = hid[:, prefix_len:full_len, :]
        out[idx] = resp_hid.mean(dim=1).squeeze(0).float().cpu()
    return out


def fit_binary_probe(X, y, seed=42):
    if len(np.unique(y)) < 2:
        return np.zeros(X.shape[1]), float("nan"), float("nan")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    clf = LogisticRegression(C=0.1, max_iter=500, random_state=seed, solver="liblinear")
    clf.fit(X_tr, y_tr)
    w = clf.coef_.flatten()
    return w / (np.linalg.norm(w) + 1e-12), clf.score(X_tr, y_tr), clf.score(X_te, y_te)


def fit_multiclass_probe(X, y, seed=42):
    if len(np.unique(y)) < 2:
        return np.zeros((1, X.shape[1])), float("nan"), float("nan")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    base = LogisticRegression(C=0.1, max_iter=500, random_state=seed, solver="liblinear")
    clf = OneVsRestClassifier(base, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    W = np.stack([e.coef_.flatten() for e in clf.estimators_], axis=0)
    W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
    return W, clf.score(X_tr, y_tr), clf.score(X_te, y_te)


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na < 1e-12 or nb < 1e-12 else float(np.dot(a, b) / (na * nb))


def projection_ratio(v, U):
    if U.ndim == 1:
        U = U[None, :]
    Q, _ = np.linalg.qr(U.T)
    v_proj = Q @ (Q.T @ v)
    return float(np.linalg.norm(v_proj) / (np.linalg.norm(v) + 1e-12))


def load_hauptanliegen_map(ncp_path: Path) -> dict[int, str]:
    mapping = {}
    with ncp_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--pairs-path", required=True)
    p.add_argument("--ncp-path", default=str(ROOT / "data/processed/ncp_eval.jsonl"))
    p.add_argument("--v-open-dir", required=True)
    p.add_argument("--layers", default="1")
    p.add_argument("--n-pairs", type=int, default=1000)
    p.add_argument("--output-path", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--torch-dtype", default="bfloat16", choices=("float32", "float16", "bfloat16"))
    p.add_argument("--prompt-format", default="plain", choices=("plain", "chat"))
    p.add_argument("--force-extract", action="store_true")
    return p.parse_args()


def resolve_torch_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def load_stored_v_open(v_open_dir: Path, layer: int):
    pt = v_open_dir / f"layer_{layer}_vectors.pt"
    if not pt.exists():
        return None
    blob = torch.load(pt, map_location="cpu", weights_only=False)
    d = torch.as_tensor(blob["direction"]).detach().float().numpy()
    return d / (np.linalg.norm(d) + 1e-12)


def load_pairs(path: Path, n: int, seed: int) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:n]


def build_prompt_prefix(context: str, prompt_format: str, tokenizer) -> str:
    # For matching the paper's paired dense setup, we use plain "{context}\nKlient: ".
    # The paper uses model-specific chat formatting; we mirror the simplest version
    # since the paired dense v_open was built with similar prefix conventions.
    return f"{context}\nKlient: "


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    layers = [int(x) for x in args.layers.replace(";", ",").split(",")]

    pairs = load_pairs(Path(args.pairs_path), args.n_pairs, args.seed)
    print(f"Loaded {len(pairs)} pairs", flush=True)

    ha_map = load_hauptanliegen_map(Path(args.ncp_path))
    pairs = [p for p in pairs if p.get("seed_record_index") in ha_map]
    print(f"After hauptanliegen filter: {len(pairs)}", flush=True)

    cache_dir = Path(args.output_path).parent / "cops_hidden_cache_v3"
    cache_dir.mkdir(parents=True, exist_ok=True)
    pairs_stem = Path(args.pairs_path).stem
    model_stem = args.model_path.replace("/", "_")
    cache_file = cache_dir / f"v3_{model_stem}_{pairs_stem}_n{len(pairs)}_seed{args.seed}_layers{'-'.join(map(str,layers))}.npz"

    if cache_file.exists() and not args.force_extract:
        print(f"Loading cached {cache_file}", flush=True)
        npz = np.load(cache_file)
        h_open = {l: npz[f"open_layer_{l}"] for l in layers}
        h_def = {l: npz[f"def_layer_{l}"] for l in layers}
    else:
        print(f"Loading model {args.model_path}...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=resolve_torch_dtype(args.torch_dtype), trust_remote_code=True
        )
        if torch.cuda.is_available():
            model = model.to("cuda")
        model.eval()
        print(f"Model on {next(model.parameters()).device}", flush=True)

        ho = {l: [] for l in layers}
        hd = {l: [] for l in layers}

        print(f"Extracting response-mean hidden states at layers {layers}...", flush=True)
        for i, p in enumerate(pairs):
            prefix = build_prompt_prefix(p["context"], args.prompt_format, tokenizer)
            # Use axis-aware field names (e.g. open_response vs explorative_response)
            pos_field = p.get("positive_field", "open_response")
            neg_field = p.get("negative_field", "defensive_response")
            cap_o = capture_response_mean_hidden_multi(model, tokenizer, prefix, p[pos_field], layers)
            cap_d = capture_response_mean_hidden_multi(model, tokenizer, prefix, p[neg_field], layers)
            for l in layers:
                ho[l].append(cap_o[l].numpy())
                hd[l].append(cap_d[l].numpy())
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(pairs)}", flush=True)

        h_open = {l: np.stack(ho[l], axis=0) for l in layers}
        h_def = {l: np.stack(hd[l], axis=0) for l in layers}

        np.savez_compressed(
            cache_file,
            **{f"open_layer_{l}": h_open[l] for l in layers},
            **{f"def_layer_{l}": h_def[l] for l in layers},
            pair_persona_name=np.asarray([p.get("persona_name", "") for p in pairs]),
            pair_seed_record_index=np.asarray([int(p["seed_record_index"]) for p in pairs], dtype=np.int64),
        )
        print(f"Cached {cache_file}", flush=True)
        del model
        torch.cuda.empty_cache()

    hauptanliegen_per_pair = [ha_map[p["seed_record_index"]] for p in pairs]
    personas = sorted({p["persona_name"] for p in pairs})
    p2id = {n: i for i, n in enumerate(personas)}

    results = {
        "n_pairs": len(pairs),
        "model_path": args.model_path,
        "pooling": "response_mean",
        "layers": {},
    }

    for layer in layers:
        print(f"\n=== Layer {layer} (response-mean) ===", flush=True)
        X_open = h_open[layer]
        X_def = h_def[layer]

        v_open_fresh = X_open.mean(0) - X_def.mean(0)
        v_open_fresh = v_open_fresh / (np.linalg.norm(v_open_fresh) + 1e-12)

        stored = load_stored_v_open(Path(args.v_open_dir), layer)
        sanity = cosine(v_open_fresh, stored) if stored is not None else None
        print(f"  SANITY cos(fresh_response_mean, stored) = {sanity}", flush=True)

        X = np.concatenate([X_open, X_def], axis=0)
        ha_doubled = hauptanliegen_per_pair + hauptanliegen_per_pair
        personas_doubled = [p["persona_name"] for p in pairs] * 2
        y_persona = np.array([p2id[n] for n in personas_doubled])

        layer_r = {"sanity_cos_fresh_vs_stored": sanity, "probes": {}}

        for probe_name, kws in PROBE_KEYWORDS.items():
            y = np.array([int(any(kw in ha for kw in kws)) for ha in ha_doubled])
            w, tr, te = fit_binary_probe(X, y)
            c = cosine(v_open_fresh, w)
            layer_r["probes"][probe_name] = {
                "test_acc": te, "cosine_with_v_open": c, "abs_cosine": abs(c),
                "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
            }
            print(f"  {probe_name}: test_acc={te:.3f}  cos={c:+.4f}  |cos|={abs(c):.4f}", flush=True)

        # Combined
        dirs = []
        for probe_name, kws in PROBE_KEYWORDS.items():
            y = np.array([int(any(kw in ha for kw in kws)) for ha in ha_doubled])
            w, _, _ = fit_binary_probe(X, y)
            if np.linalg.norm(w) > 1e-6:
                dirs.append(w)
        if dirs:
            U = np.stack(dirs, axis=0)
            pr = projection_ratio(v_open_fresh, U)
            layer_r["combined_content_subspace"] = {"n_dirs": len(dirs), "projection_ratio_of_v_open": pr}
            print(f"  combined content proj: {pr:.4f}", flush=True)

        W_persona, tr, te = fit_multiclass_probe(X, y_persona)
        pr_persona = projection_ratio(v_open_fresh, W_persona)
        layer_r["persona_probe"] = {
            "train_acc": tr, "test_acc": te, "n_classes": W_persona.shape[0],
            "projection_ratio_of_v_open": pr_persona,
        }
        print(f"  persona multiclass: test_acc={te:.3f}  proj_ratio={pr_persona:.4f}", flush=True)

        results["layers"][str(layer)] = layer_r

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    main()
