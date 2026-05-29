#!/usr/bin/env python3
"""Decompose v_open's projection onto per-persona directions.

Tests whether v_open's high projection onto the persona subspace reflects:
  (a) a systematic pull toward one specific persona — real drift
  (b) a symmetric "persona-ness" direction — natural correlate of openness

If (a), a few personas dominate cos(v_open, w_persona_i).
If (b), cosines are roughly uniform across personas.

Uses cached hidden states from cops_overlap_check_v2.py.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb + 1e-12))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cache-file",
        required=True,
        help="e.g. outputs/metrics/cops_hidden_cache/v2_persona_pairs_openness_gpt54mini_1000_n1000_seed42_layers1-5-9-13-18.npz",
    )
    p.add_argument(
        "--pairs-path",
        required=True,
        help="The jsonl used to generate this cache (for persona labels).",
    )
    p.add_argument("--layers", default="1,5,9,13,18")
    p.add_argument("--output-path", required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    layers = [int(x) for x in args.layers.split(",")]

    # Load pairs for persona labels
    with Path(args.pairs_path).open() as f:
        pairs = [json.loads(l) for l in f if l.strip()]
    print(f"Loaded {len(pairs)} pairs")

    personas_per_pair = [p["persona_name"] for p in pairs]
    personas_doubled = personas_per_pair + personas_per_pair  # open + def
    personas = sorted(set(personas_doubled))
    persona_to_id = {n: i for i, n in enumerate(personas)}
    y = np.array([persona_to_id[n] for n in personas_doubled])
    print(f"Persona classes: {personas}")

    # Load cache
    npz = np.load(args.cache_file)
    print(f"Cache keys: {list(npz.files)}")

    results = {"personas": personas, "layers": {}}

    for layer in layers:
        h_open = npz[f"open_layer_{layer}"]
        h_def = npz[f"def_layer_{layer}"]
        X = np.concatenate([h_open, h_def], axis=0)

        # Fresh v_open at this position
        v_open = h_open.mean(axis=0) - h_def.mean(axis=0)
        v_open = v_open / (np.linalg.norm(v_open) + 1e-12)

        # Train OvR persona probe
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=args.seed, stratify=y)
        base = LogisticRegression(C=0.1, max_iter=500, random_state=args.seed, solver="liblinear")
        clf = OneVsRestClassifier(base, n_jobs=-1)
        clf.fit(X_tr, y_tr)

        # Per-persona weight vectors
        W_list = [e.coef_.flatten() for e in clf.estimators_]
        W_list = [w / (np.linalg.norm(w) + 1e-12) for w in W_list]
        # Order matches persona_to_id order
        per_persona_cos = {personas[i]: cosine(v_open, W_list[i]) for i in range(len(personas))}

        # Also compute n_samples per persona (open + def)
        n_per_persona = {p: sum(1 for x in personas_doubled if x == p) for p in personas}

        abs_cos = np.array([abs(c) for c in per_persona_cos.values()])
        layer_result = {
            "test_acc": clf.score(X_te, y_te),
            "per_persona_cos": per_persona_cos,
            "abs_cos_mean": float(abs_cos.mean()),
            "abs_cos_max": float(abs_cos.max()),
            "abs_cos_std": float(abs_cos.std()),
            "gini": float(np.abs(np.diff(sorted(abs_cos))).sum() / (2 * abs_cos.sum() + 1e-12)),  # concentration
            "n_per_persona": n_per_persona,
        }
        results["layers"][str(layer)] = layer_result

        print(f"\n=== Layer {layer} (test_acc={layer_result['test_acc']:.3f}) ===")
        sorted_cos = sorted(per_persona_cos.items(), key=lambda kv: -abs(kv[1]))
        for name, c in sorted_cos:
            n = n_per_persona[name]
            marker = "⭐" if abs(c) > 2 * layer_result["abs_cos_mean"] else "  "
            print(f"  {marker} {name:20s}  cos={c:+.4f}  |cos|={abs(c):.4f}  n={n}")
        print(f"   abs_cos: mean={layer_result['abs_cos_mean']:.4f}  max={layer_result['abs_cos_max']:.4f}  max/mean={layer_result['abs_cos_max']/layer_result['abs_cos_mean']:.2f}")

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output_path).open("w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {args.output_path}")


if __name__ == "__main__":
    main()
