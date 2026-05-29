#!/usr/bin/env python3
"""Test per-persona v_open variants.

Variants:
  global      — mean(h_open) − mean(h_def) across ALL pairs (current baseline)
  within      — persona-centered: subtract per-persona mean first, then compute v_open
  per_persona — one v_open per persona: v_open_P = mean(h_open_P) − mean(h_def_P)

For each variant and each layer, measures:
  - Projection onto persona subspace
  - Pairwise cosine between per-persona v_open vectors (for per_persona only)
  - Cosine of each variant's v_open with the global v_open

Uses cached hidden states. CPU-only.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb + 1e-12))


def projection_ratio(v, U):
    if U.ndim == 1:
        U = U[None, :]
    Q, _ = np.linalg.qr(U.T)
    v_proj = Q @ (Q.T @ v)
    return float(np.linalg.norm(v_proj) / (np.linalg.norm(v) + 1e-12))


def normalize(v):
    return v / (np.linalg.norm(v) + 1e-12)


def compute_variants(h_open, h_def, personas):
    """Return dict {variant_name: direction_vector}."""
    personas = np.asarray(personas)
    unique = sorted(set(personas.tolist()))

    # Global
    v_global = normalize(h_open.mean(0) - h_def.mean(0))

    # Within-centered: subtract per-persona mean from each hidden state,
    # then compute v_open. Removes cross-persona bias.
    h_open_centered = h_open.copy()
    h_def_centered = h_def.copy()
    for p in unique:
        mask = personas == p
        if mask.sum() == 0:
            continue
        # joint mean (open + def) per persona — subtract this so each persona contributes
        # to the within-persona style direction only
        joint_mean = np.concatenate([h_open[mask], h_def[mask]]).mean(0)
        h_open_centered[mask] -= joint_mean
        h_def_centered[mask] -= joint_mean
    v_within = normalize(h_open_centered.mean(0) - h_def_centered.mean(0))

    # Per-persona
    v_per_persona = {}
    n_per = {}
    for p in unique:
        mask = personas == p
        if mask.sum() < 5:  # need at least a few samples
            continue
        v = h_open[mask].mean(0) - h_def[mask].mean(0)
        v_per_persona[p] = normalize(v)
        n_per[p] = int(mask.sum())

    return {
        "global": v_global,
        "within": v_within,
        "per_persona": v_per_persona,
        "n_per_persona": n_per,
    }


def fit_persona_subspace(X, y, seed=42):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    base = LogisticRegression(C=0.1, max_iter=500, random_state=seed, solver="liblinear")
    clf = OneVsRestClassifier(base, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    W = np.stack([e.coef_.flatten() for e in clf.estimators_], axis=0)
    W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
    return W, clf.score(X_te, y_te)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-file", required=True)
    p.add_argument("--pairs-path", required=True)
    p.add_argument("--layers", default="1,5,9,13,18")
    p.add_argument("--output-path", required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    layers = [int(x) for x in args.layers.split(",")]

    with Path(args.pairs_path).open() as f:
        pairs = [json.loads(l) for l in f if l.strip()]
    personas_per_pair = [p["persona_name"] for p in pairs]
    personas_doubled = personas_per_pair + personas_per_pair
    unique_personas = sorted(set(personas_doubled))
    persona_to_id = {n: i for i, n in enumerate(unique_personas)}
    y = np.array([persona_to_id[n] for n in personas_doubled])

    npz = np.load(args.cache_file)
    print(f"Loaded {len(pairs)} pairs, personas: {unique_personas}")

    results = {
        "personas": unique_personas,
        "n_pairs": len(pairs),
        "layers": {},
    }

    for layer in layers:
        h_open = npz[f"open_layer_{layer}"]
        h_def = npz[f"def_layer_{layer}"]

        variants = compute_variants(h_open, h_def, personas_per_pair)

        # Persona subspace from hidden states
        X = np.concatenate([h_open, h_def], axis=0)
        W_persona, persona_test_acc = fit_persona_subspace(X, y, args.seed)

        v_global = variants["global"]
        v_within = variants["within"]
        v_per = variants["per_persona"]

        layer_result = {
            "persona_probe_test_acc": persona_test_acc,
            "global_proj_onto_persona": projection_ratio(v_global, W_persona),
            "within_proj_onto_persona": projection_ratio(v_within, W_persona),
            "cos_global_within": cosine(v_global, v_within),
            "per_persona": {
                p: {
                    "n": variants["n_per_persona"].get(p, 0),
                    "proj_onto_persona": projection_ratio(v_per[p], W_persona),
                    "cos_with_global": cosine(v_per[p], v_global),
                    "cos_with_within": cosine(v_per[p], v_within),
                }
                for p in v_per
            },
        }

        # Pairwise cosines between per-persona vectors
        persona_list = list(v_per.keys())
        pairwise = {}
        cos_vals = []
        for i, pi in enumerate(persona_list):
            for j, pj in enumerate(persona_list):
                if i < j:
                    c = cosine(v_per[pi], v_per[pj])
                    pairwise[f"{pi}::{pj}"] = c
                    cos_vals.append(c)
        layer_result["pairwise_cos_per_persona"] = {
            "mean": float(np.mean(cos_vals)) if cos_vals else 0,
            "min": float(np.min(cos_vals)) if cos_vals else 0,
            "max": float(np.max(cos_vals)) if cos_vals else 0,
            "details": pairwise,
        }

        results["layers"][str(layer)] = layer_result

        print(f"\n=== Layer {layer} ===")
        print(f"  persona_probe_test_acc: {persona_test_acc:.3f}")
        print(f"  GLOBAL   v_open proj onto persona: {layer_result['global_proj_onto_persona']:.4f}")
        print(f"  WITHIN   v_open proj onto persona: {layer_result['within_proj_onto_persona']:.4f}")
        print(f"  cos(global, within):               {layer_result['cos_global_within']:.4f}")
        print(f"  PER-PERSONA pairwise cos: mean={layer_result['pairwise_cos_per_persona']['mean']:+.4f}  min={layer_result['pairwise_cos_per_persona']['min']:+.4f}  max={layer_result['pairwise_cos_per_persona']['max']:+.4f}")
        print(f"  PER-PERSONA details:")
        for p in sorted(layer_result["per_persona"], key=lambda x: -layer_result["per_persona"][x]["proj_onto_persona"]):
            pd = layer_result["per_persona"][p]
            print(f"    {p:20s}  n={pd['n']:4d}  proj_persona={pd['proj_onto_persona']:.3f}  cos(v_P, v_global)={pd['cos_with_global']:+.3f}")

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output_path).open("w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {args.output_path}")


if __name__ == "__main__":
    main()
