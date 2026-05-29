#!/usr/bin/env python3
"""Extract the persona-invariant style direction via PCA over per-persona v_open vectors.

For each layer:
  1. Compute per-persona v_P = mean(h_open_P) − mean(h_def_P) for each persona.
  2. PCA over the 10 per-persona vectors → PC1 is the common direction.
  3. Measure PC1's projection onto the persona subspace (should be lower than v_global).
  4. Measure PC1's cosine with v_global (should be high — shared component).
  5. Save PC1 as a candidate clean steering vector.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
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
    p.add_argument("--save-vectors-dir", default=None, help="Optional dir to save v_global, v_style_core, v_P vectors as .npz per layer.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    layers = [int(x) for x in args.layers.split(",")]

    with Path(args.pairs_path).open() as f:
        pairs = [json.loads(l) for l in f if l.strip()]
    personas_per_pair = np.array([p["persona_name"] for p in pairs])
    personas_doubled = list(personas_per_pair) + list(personas_per_pair)
    unique_personas = sorted(set(personas_doubled))
    persona_to_id = {n: i for i, n in enumerate(unique_personas)}
    y = np.array([persona_to_id[n] for n in personas_doubled])

    npz = np.load(args.cache_file)
    print(f"Loaded {len(pairs)} pairs, {len(unique_personas)} personas")

    results = {"personas": unique_personas, "layers": {}}

    save_dir = Path(args.save_vectors_dir) if args.save_vectors_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    for layer in layers:
        h_open = npz[f"open_layer_{layer}"]
        h_def = npz[f"def_layer_{layer}"]

        # Per-persona v_P
        v_per = {}
        for p in unique_personas:
            mask = personas_per_pair == p
            if mask.sum() < 5:
                continue
            v = h_open[mask].mean(0) - h_def[mask].mean(0)
            v_per[p] = v  # NOT normalized yet — preserve magnitude for PCA
        V = np.stack([v_per[p] for p in v_per], axis=0)  # [n_personas, d]

        # Global v_open
        v_global = normalize(h_open.mean(0) - h_def.mean(0))

        # PCA over per-persona vectors
        pca = PCA(n_components=min(5, V.shape[0]))
        pca.fit(V)
        components = pca.components_  # [n_components, d]
        explained = pca.explained_variance_ratio_
        # Ensure PC1 points in the same general direction as v_global (sign fix)
        v_style_core = components[0]
        if np.dot(v_style_core, v_global) < 0:
            v_style_core = -v_style_core
            components[0] = v_style_core
        v_style_core = normalize(v_style_core)

        # Persona subspace from hidden states
        X = np.concatenate([h_open, h_def], axis=0)
        W_persona, persona_test_acc = fit_persona_subspace(X, y, args.seed)

        proj_global = projection_ratio(v_global, W_persona)
        proj_core = projection_ratio(v_style_core, W_persona)

        layer_result = {
            "persona_probe_test_acc": persona_test_acc,
            "pca_explained_variance_ratio": explained.tolist(),
            "v_style_core_norm_of_first_pc": float(pca.singular_values_[0]),
            "cos_core_global": cosine(v_style_core, v_global),
            "global_proj_onto_persona": proj_global,
            "style_core_proj_onto_persona": proj_core,
            "reduction_vs_global": proj_global - proj_core,
            "per_persona_cos_with_core": {
                p: cosine(normalize(v_per[p]), v_style_core) for p in v_per
            },
        }
        results["layers"][str(layer)] = layer_result

        print(f"\n=== Layer {layer} ===")
        print(f"  persona probe test_acc:           {persona_test_acc:.3f}")
        print(f"  PCA explained variance:           PC1={explained[0]:.3f}  PC2={explained[1]:.3f}  PC3={explained[2]:.3f}  PC4={explained[3]:.3f}  PC5={explained[4]:.3f}")
        print(f"  cos(v_style_core, v_global):      {layer_result['cos_core_global']:.4f}")
        print(f"  v_global     proj onto persona:   {proj_global:.4f}")
        print(f"  v_style_core proj onto persona:   {proj_core:.4f}")
        print(f"  → reduction:                      {proj_global - proj_core:+.4f} ({100*(proj_global-proj_core)/max(proj_global,1e-9):+.1f}% relative)")
        print(f"  per-persona cos with core:")
        for p, c in sorted(layer_result["per_persona_cos_with_core"].items(), key=lambda kv: -kv[1]):
            print(f"    {p:20s}  {c:+.4f}")

        if save_dir:
            out = save_dir / f"layer_{layer}_vectors.npz"
            np.savez_compressed(
                out,
                v_global=v_global,
                v_style_core=v_style_core,
                pca_components=components,
                pca_explained=explained,
                **{f"v_{p}": normalize(v_per[p]) for p in v_per},
            )

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output_path).open("w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {args.output_path}")
    if save_dir:
        print(f"Saved per-layer vectors to {save_dir}")


if __name__ == "__main__":
    main()
