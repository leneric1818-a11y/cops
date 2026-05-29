#!/usr/bin/env python3
"""COPS overlap check: are v_open and content-probe directions entangled?

Loads existing paired-dense v_open for Qwen3-4B, extracts hidden states from
cops_contexts.jsonl at the canonical layer, trains linear content probes that
discriminate case-specific facts (persona name, key entities from hauptanliegen),
and measures cosine similarity + projection ratio between probe weights and v_open.

High overlap → orthogonalisation is the core COPS contribution.
Low overlap → the method needs rethinking before API spend.

Outputs a single JSON summary with per-layer, per-probe cosine similarities.
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


# ---------------------------------------------------------------------------
# Model utilities (adapted from steering_vector_experiment.py)
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


def capture_last_token_hidden(
    model: nn.Module,
    tokenizer,
    prompt: str,
    layer_idx: int,
) -> torch.Tensor:
    layers = find_transformer_layers(model)
    captured: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured.append(hidden[:, -1, :].detach().float().cpu())

    handle = layers[layer_idx].register_forward_hook(hook)
    device = model.get_input_embeddings().weight.device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        model(**inputs)
    handle.remove()

    if not captured:
        raise RuntimeError(f"No hidden state captured at layer {layer_idx}")
    return captured[-1].squeeze(0)


# ---------------------------------------------------------------------------
# Probe construction
# ---------------------------------------------------------------------------

# Key entities to probe — drawn from the 10 personas in parsed_vikl_f2f_personas.json
# Each entry: name → list of keywords whose presence in hauptanliegen defines the positive class
PROBE_KEYWORDS = {
    "mentions_son": ["Sohn", "Max"],
    "mentions_cannabis": ["Cannabis", "Kiffen", "kiffen", "Joint", "Marihuana"],
    "mentions_husband": ["Mann", "Ehemann", "Partner"],
    "mentions_school": ["Schule", "Noten"],
    "mentions_separation": ["Trennung", "verschwunden", "abgehauen"],
    "mentions_addiction": ["Drogen", "Entzug", "Konsum", "Abhängig"],
}


def make_binary_label(hauptanliegen: str, keywords: list[str]) -> int:
    """1 if any keyword appears in hauptanliegen."""
    return int(any(kw in hauptanliegen for kw in keywords))


def fit_probe_and_get_direction(
    X: np.ndarray, y: np.ndarray, seed: int = 42
) -> tuple[np.ndarray, float, float]:
    """Fit logistic regression, return (weight_vector, train_acc, test_acc).

    Weight vector is L2-normalised. If only one class, returns zeros + NaN accs.
    Uses liblinear + strong regularisation — high-d features, few samples.
    """
    if len(np.unique(y)) < 2:
        return np.zeros(X.shape[1]), float("nan"), float("nan")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    clf = LogisticRegression(
        C=0.1, max_iter=500, random_state=seed, solver="liblinear"
    )
    clf.fit(X_tr, y_tr)
    w = clf.coef_.flatten()
    w = w / (np.linalg.norm(w) + 1e-12)
    return w, clf.score(X_tr, y_tr), clf.score(X_te, y_te)


def fit_multiclass_probe(
    X: np.ndarray, y: np.ndarray, seed: int = 42
) -> tuple[np.ndarray, float, float]:
    """One-vs-rest multiclass probe — faster and more stable than multinomial
    on very underdetermined problems (d >> n).
    """
    if len(np.unique(y)) < 2:
        return np.zeros((1, X.shape[1])), float("nan"), float("nan")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    base = LogisticRegression(
        C=0.1, max_iter=500, random_state=seed, solver="liblinear"
    )
    clf = OneVsRestClassifier(base, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    # Stack per-class weight vectors
    W = np.stack([est.coef_.flatten() for est in clf.estimators_], axis=0)
    W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
    return W, clf.score(X_tr, y_tr), clf.score(X_te, y_te)


# ---------------------------------------------------------------------------
# Overlap metrics
# ---------------------------------------------------------------------------

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def projection_ratio(v: np.ndarray, U: np.ndarray) -> float:
    """Fraction of v's norm that lies in the span of U (rows of U)."""
    if U.ndim == 1:
        U = U[None, :]
    # orthonormalise U
    Q, _ = np.linalg.qr(U.T)  # Q: [d, k]
    v_proj = Q @ (Q.T @ v)
    return float(np.linalg.norm(v_proj) / (np.linalg.norm(v) + 1e-12))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model-path",
        default="Qwen/Qwen3-4B",
        help="HF model id or local path.",
    )
    p.add_argument(
        "--contexts-path",
        default=str(ROOT / "data/processed/cops_contexts.jsonl"),
    )
    p.add_argument(
        "--v-open-dir",
        default=str(ROOT / "outputs/metrics/layerwise_separability_openness_qwen3_4b_gpt54mini_1000"),
        help="Directory containing layer_<idx>_vectors.pt files.",
    )
    p.add_argument(
        "--layers",
        default="1,5,9,13,18",
        help="Comma-separated layer indices to analyse.",
    )
    p.add_argument(
        "--n-contexts",
        type=int,
        default=300,
        help="Number of contexts to sample (stratified by persona).",
    )
    p.add_argument(
        "--output-path",
        default=str(ROOT / "outputs/metrics/cops_overlap_check.json"),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--force-extract",
        action="store_true",
        help="Re-extract hidden states even if cache exists.",
    )
    p.add_argument(
        "--torch-dtype",
        default="bfloat16",
        choices=("float32", "float16", "bfloat16"),
    )
    return p.parse_args()


def load_v_open(v_open_dir: Path, layer: int) -> np.ndarray:
    pt_path = v_open_dir / f"layer_{layer}_vectors.pt"
    if not pt_path.exists():
        raise FileNotFoundError(f"No v_open at {pt_path}")
    blob = torch.load(pt_path, map_location="cpu", weights_only=False)
    direction = torch.as_tensor(blob["direction"]).detach().float().numpy()
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    return direction


def resolve_torch_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def load_contexts(path: Path, n: int, seed: int) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    # Stratified sampling by persona
    rng = random.Random(seed)
    by_persona: dict[str, list[dict]] = {}
    for r in rows:
        by_persona.setdefault(r["persona_name"], []).append(r)

    personas = list(by_persona.keys())
    per_persona = max(1, n // len(personas))
    sampled: list[dict] = []
    for name in personas:
        pool = by_persona[name]
        rng.shuffle(pool)
        sampled.extend(pool[:per_persona])
    rng.shuffle(sampled)
    return sampled[:n]


def extract_hidden_states(
    model, tokenizer, contexts: list[dict], layers: list[int]
) -> dict[int, np.ndarray]:
    """Return dict layer → [N, d] numpy array of last-token hidden states."""
    collected: dict[int, list[np.ndarray]] = {l: [] for l in layers}
    for i, row in enumerate(contexts):
        prompt = row["context"]
        for layer in layers:
            h = capture_last_token_hidden(model, tokenizer, prompt, layer)
            collected[layer].append(h.numpy())
        if (i + 1) % 20 == 0:
            print(f"  extracted {i + 1}/{len(contexts)}", flush=True)
    return {l: np.stack(v, axis=0) for l, v in collected.items()}


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    layers = [int(x) for x in args.layers.split(",")]
    contexts = load_contexts(Path(args.contexts_path), args.n_contexts, args.seed)
    print(f"Loaded {len(contexts)} contexts (stratified by persona)")

    # Build binary probe labels
    label_coverage = {}
    for probe_name, kws in PROBE_KEYWORDS.items():
        labels = [make_binary_label(r["hauptanliegen"], kws) for r in contexts]
        label_coverage[probe_name] = {
            "positive": sum(labels),
            "negative": len(labels) - sum(labels),
        }
    print("Probe label coverage:")
    for k, v in label_coverage.items():
        print(f"  {k}: pos={v['positive']} neg={v['negative']}")

    # Persona label (multiclass)
    personas = sorted({r["persona_name"] for r in contexts})
    persona_to_id = {name: i for i, name in enumerate(personas)}
    persona_labels = np.array([persona_to_id[r["persona_name"]] for r in contexts])
    print(f"Persona classes: {len(personas)}")

    # Hidden-state cache (keyed by contexts path + n + seed + layers)
    cache_dir = Path(args.output_path).parent / "cops_hidden_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = f"n{args.n_contexts}_seed{args.seed}_layers{args.layers.replace(',', '-')}"
    cache_file = cache_dir / f"{cache_key}.npz"

    if cache_file.exists() and not args.force_extract:
        print(f"Loading cached hidden states from {cache_file}", flush=True)
        npz = np.load(cache_file)
        hidden_by_layer = {l: npz[f"layer_{l}"] for l in layers}
    else:
        # Load model
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
        print(f"Model on {next(model.parameters()).device}, dtype {next(model.parameters()).dtype}", flush=True)

        # Extract hidden states
        print(f"\nExtracting hidden states at layers {layers}...", flush=True)
        hidden_by_layer = extract_hidden_states(model, tokenizer, contexts, layers)

        # Save cache
        np.savez_compressed(
            cache_file,
            **{f"layer_{l}": hidden_by_layer[l] for l in layers},
        )
        print(f"Cached hidden states to {cache_file}", flush=True)

        # Free model memory
        del model
        torch.cuda.empty_cache()

    # Compute overlap per layer
    results: dict = {
        "model_path": args.model_path,
        "n_contexts": len(contexts),
        "layers": {},
        "label_coverage": label_coverage,
    }

    for layer in layers:
        X = hidden_by_layer[layer]  # [N, d]
        print(f"\n=== Layer {layer} (d={X.shape[1]}) ===")

        v_open = load_v_open(Path(args.v_open_dir), layer)

        layer_result: dict = {
            "v_open_norm": float(np.linalg.norm(v_open)),
            "probes": {},
        }

        # Binary probes
        for probe_name, kws in PROBE_KEYWORDS.items():
            y = np.array([make_binary_label(r["hauptanliegen"], kws) for r in contexts])
            w, train_acc, test_acc = fit_probe_and_get_direction(X, y)
            cos = cosine(v_open, w)
            layer_result["probes"][probe_name] = {
                "train_acc": train_acc,
                "test_acc": test_acc,
                "cosine_with_v_open": cos,
                "abs_cosine": abs(cos),
                "n_pos": int(y.sum()),
                "n_neg": int((1 - y).sum()),
            }
            print(f"  {probe_name}: test_acc={test_acc:.3f}  cos(w, v_open)={cos:+.4f}  |cos|={abs(cos):.4f}")

        # Persona multiclass probe — stacked weights as U_content
        W_persona, train_acc, test_acc = fit_multiclass_probe(X, persona_labels)
        proj = projection_ratio(v_open, W_persona)
        layer_result["persona_probe"] = {
            "train_acc": train_acc,
            "test_acc": test_acc,
            "n_classes": W_persona.shape[0],
            "projection_ratio_of_v_open_onto_persona_subspace": proj,
        }
        print(f"  persona(multiclass): test_acc={test_acc:.3f}  proj_ratio={proj:.4f}")

        # Combined content subspace — stack all binary probe directions
        probe_dirs = []
        for probe_name in PROBE_KEYWORDS:
            y = np.array([make_binary_label(r["hauptanliegen"], kws) for kws in [PROBE_KEYWORDS[probe_name]] for r in contexts])
            w, _, _ = fit_probe_and_get_direction(X, y)
            if np.linalg.norm(w) > 1e-6:
                probe_dirs.append(w)
        if probe_dirs:
            U_content = np.stack(probe_dirs, axis=0)
            combined_proj = projection_ratio(v_open, U_content)
            layer_result["combined_content_subspace"] = {
                "n_dirs": len(probe_dirs),
                "projection_ratio_of_v_open": combined_proj,
            }
            print(f"  combined binary probes → projection ratio of v_open: {combined_proj:.4f}")

        results["layers"][str(layer)] = layer_result

    # Summary interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("|cos| = cosine similarity between probe direction and v_open (|.| to ignore sign).")
    print("projection_ratio = fraction of v_open's norm explained by the content subspace.")
    print("  Rule of thumb: > 0.3 means meaningful entanglement, > 0.5 means strong entanglement.")

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
