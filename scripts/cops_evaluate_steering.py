#!/usr/bin/env python3
"""COPS steering evaluation — generates outputs under multiple steering variants.

For one (model, axis, layer) combination: iterates over held-out contexts and
generates:
  - baseline (no steering)
  - for each specified variant × alpha: steered generation

Variants supported:
  global_blind   — use stored v_open from benchmark (layer_<L>_vectors.pt)
  global_aware   — compute v_open from case-aware pairs cache
  P_matched      — per-persona v_P where persona matches context persona
  P_mismatched   — per-persona v_P from a different persona (control)

Outputs JSONL with one row per (context, variant, alpha) combination.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Reuse the existing steering infrastructure
from steering_vector_experiment import (  # type: ignore
    find_transformer_layers,
    steering_hook,
    generate_text,
    seed_everything,
)


# ---------------------------------------------------------------------------
# Vector loaders
# ---------------------------------------------------------------------------

def load_stored_v_open(v_open_dir: Path, layer: int) -> torch.Tensor | None:
    pt = v_open_dir / f"layer_{layer}_vectors.pt"
    if not pt.exists():
        return None
    blob = torch.load(pt, map_location="cpu", weights_only=False)
    return torch.as_tensor(blob["direction"]).detach().float()


def load_case_aware_vectors_from_cache(
    cache_file: Path, pairs_path: Path, layer: int
) -> dict:
    """Return dict {v_global_aware, v_P_matched_{persona}, ...} from the v3 response-mean cache.

    Pair/persona metadata is loaded from the cache when available. Older caches
    require an exactly aligned pairs JSONL.
    """
    if not cache_file.exists():
        raise FileNotFoundError(f"Cache not found: {cache_file}")
    npz = np.load(cache_file)
    h_open = npz[f"open_layer_{layer}"]
    h_def = npz[f"def_layer_{layer}"]
    if h_open.shape[0] != h_def.shape[0]:
        raise ValueError(
            f"Case-aware cache is internally inconsistent: open rows={h_open.shape[0]} "
            f"but def rows={h_def.shape[0]} in {cache_file}"
        )
    n_rows = int(h_open.shape[0])

    if "pair_persona_name" in npz.files:
        raw_personas = np.asarray(npz["pair_persona_name"])
        if raw_personas.shape[0] != n_rows:
            raise ValueError(
                f"Case-aware cache metadata length mismatch in {cache_file}: "
                f"pair_persona_name has {raw_personas.shape[0]} rows, cache has {n_rows}"
            )
        personas = np.array([str(persona) for persona in raw_personas.tolist()])
    else:
        with pairs_path.open(encoding="utf-8") as f:
            pairs = [json.loads(l) for l in f if l.strip()]
        if len(pairs) != n_rows:
            raise ValueError(
                f"Case-aware cache rows ({n_rows}) do not match pair rows ({len(pairs)}) for {cache_file}. "
                "Regenerate the cache with embedded persona metadata or provide the exact aligned pairs JSONL "
                "used to create this cache."
            )
        personas = np.array([p["persona_name"] for p in pairs])

    v_global = h_open.mean(0) - h_def.mean(0)
    g_norm_sq = float(np.dot(v_global, v_global)) + 1e-12

    h_pool = np.concatenate([h_open, h_def], axis=0)
    p_pool = np.concatenate([personas, personas], axis=0)

    v_per: dict[str, np.ndarray] = {}
    v_per_A: dict[str, np.ndarray] = {}  # orthogonal residual to v_global
    v_per_B: dict[str, np.ndarray] = {}  # cross-persona contrast (pole-pooled)
    v_per_C: dict[str, np.ndarray] = {}  # interaction term: v_persona - v_global
    for p in sorted(set(personas.tolist())):
        mask = personas == p
        if mask.sum() < 5:
            continue
        v_orig = h_open[mask].mean(0) - h_def[mask].mean(0)
        v_per[p] = v_orig
        # A: residual orthogonal to v_global
        proj = (np.dot(v_orig, v_global) / g_norm_sq) * v_global
        v_per_A[p] = v_orig - proj
        # B: cross-persona contrast (pole-pooled)
        m_pool = p_pool == p
        v_per_B[p] = h_pool[m_pool].mean(0) - h_pool[~m_pool].mean(0)
        # C: interaction term
        v_per_C[p] = v_orig - v_global

    return {
        "v_global": torch.tensor(v_global, dtype=torch.float32),
        "v_per_persona": {p: torch.tensor(v, dtype=torch.float32) for p, v in v_per.items()},
        "v_per_persona_A": {p: torch.tensor(v, dtype=torch.float32) for p, v in v_per_A.items()},
        "v_per_persona_B": {p: torch.tensor(v, dtype=torch.float32) for p, v in v_per_B.items()},
        "v_per_persona_C": {p: torch.tensor(v, dtype=torch.float32) for p, v in v_per_C.items()},
    }


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_prompt(context: str, tokenizer, prompt_format: str, model_path: str = "", persona_instruction: str = "") -> str:
    """Build the generation prompt from context.

    plain: "{context}\nKlient: "
    chat:  apply the model's chat template (with thinking disabled for Qwen).

    persona_instruction (optional): if provided, prepended to the system
    prompt — used for the prompt-baseline ablation (see appendix).
    """
    if prompt_format == "plain":
        prefix = f"{persona_instruction}\n" if persona_instruction else ""
        return f"{prefix}{context}\nKlient: "
    system = (
        "Du spielst einen Klienten in einem Beratungsgespräch. "
        "Antworte in 1–3 Sätzen natürlich und falltreu als nächste Klientenäußerung. "
        "Keine Metakommentare, keine Rollenbeschreibungen."
    )
    if persona_instruction:
        system = persona_instruction.rstrip(".") + ". " + system
    messages = [
        {"role": "user", "content": f"{system}\n\nGesprächsverlauf:\n{context}\n\nDeine nächste Äußerung als Klient:"},
    ]
    common_kwargs = {"tokenize": False, "add_generation_prompt": True}
    is_qwen = "qwen" in (model_path or "").lower()
    if is_qwen:
        # Disable thinking mode (Qwen3 supports enable_thinking flag)
        try:
            return tokenizer.apply_chat_template(messages, enable_thinking=False, **common_kwargs)
        except TypeError:
            patched = [dict(m) for m in messages]
            patched[-1]["content"] = f"/no_think\n\n{patched[-1]['content']}"
            return tokenizer.apply_chat_template(patched, **common_kwargs)
    try:
        return tokenizer.apply_chat_template(messages, **common_kwargs)
    except Exception:
        return f"{context}\nKlient: "


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--axis", required=True, help="openness / initiative / cooperation / hopefulness")
    p.add_argument(
        "--variants",
        default="baseline,global_blind,global_aware,P_matched,P_mismatched",
        help="Comma-separated variants to run.",
    )
    p.add_argument(
        "--alphas",
        default="0.5,1.5,3.0",
        help="Comma-separated alpha values (ignored for baseline).",
    )
    p.add_argument(
        "--held-out-path",
        default=str(ROOT / "data/processed/cops_contexts.jsonl"),
    )
    p.add_argument("--held-out-start", type=int, default=200)
    p.add_argument("--held-out-count", type=int, default=200)
    p.add_argument("--held-out-shuffle-seed", type=int, default=42)
    p.add_argument(
        "--v-open-dir",
        required=True,
        help="Directory with layer_<L>_vectors.pt (e.g. layerwise_separability_openness_...).",
    )
    p.add_argument(
        "--case-aware-cache",
        default=None,
        help="Path to v3 response-mean cache for case-aware pairs; required if using aware/P_matched/P_mismatched.",
    )
    p.add_argument(
        "--case-aware-pairs",
        default=None,
        help="Path to case-aware pairs JSONL (used to recover persona labels per row in cache).",
    )
    p.add_argument("--output-path", required=True)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--repetition-penalty", type=float, default=1.05)
    p.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=42, help="Single-seed mode (when --seeds is not set).")
    p.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated list of seeds (e.g. '42,123,2024'). If set, outputs one file per seed with suffix _seed<N>. Overrides --seed.",
    )
    p.add_argument("--torch-dtype", default="bfloat16", choices=("float32", "float16", "bfloat16"))
    p.add_argument("--prompt-format", default="chat", choices=("chat", "plain"))
    p.add_argument("--persona-instruction", default="",
                   help="Optional German persona instruction prepended to the system prompt for the prompt-baseline ablation.")
    p.add_argument("--normalize-vector", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def resolve_torch_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def load_held_out(path: Path, seed: int, start: int, count: int) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[start : start + count]


def _seed_output_path(base_path: Path, seed: int, n_seeds: int) -> Path:
    """If n_seeds > 1, insert _seed<N> before extension; else use base path unchanged."""
    if n_seeds <= 1:
        return base_path
    return base_path.with_name(f"{base_path.stem}_seed{seed}{base_path.suffix}")


def run_one_seed(
    *, seed: int, n_seeds: int, args, contexts: list[dict], vectors: dict, variants: list[str],
    alphas: list[float], personas_in_caw: list[str], model, tokenizer,
) -> None:
    """Generate outputs for a single seed."""
    seed_everything(seed)
    out_path = _seed_output_path(Path(args.output_path), seed, n_seeds)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mismatch_rng = random.Random(seed + 7)

    with out_path.open("w", encoding="utf-8") as f:
        for ctx_idx, ctx in enumerate(contexts):
            prompt = build_prompt(ctx["context"], tokenizer, args.prompt_format, args.model_path,
                                  persona_instruction=args.persona_instruction)
            persona = ctx.get("persona_name", "")
            gen_seed = seed + ctx_idx

            # Baseline — no steering
            if "baseline" in variants:
                gen = generate_text(
                    model=model, tokenizer=tokenizer, prompt=prompt,
                    max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                    top_p=args.top_p, top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                    do_sample=args.do_sample, seed=gen_seed,
                )
                f.write(json.dumps({
                    "seed_id": ctx["seed_id"],
                    "run_seed": seed,
                    "persona_name": persona,
                    "target_label": ctx.get("target_label"),
                    "hauptanliegen": ctx["hauptanliegen"],
                    "context": ctx["context"],
                    "variant": "baseline",
                    "alpha": 0.0,
                    "used_persona_ref": None,
                    "axis": args.axis,
                    "layer": args.layer,
                    "model_path": args.model_path,
                    "response": gen,
                }, ensure_ascii=False) + "\n")
                f.flush()

            # Steering variants × alphas
            for variant in variants:
                if variant == "baseline":
                    continue
                for alpha in alphas:
                    used_persona_ref: str | None = None
                    if variant in ("global_blind", "global_aware"):
                        vec = vectors.get(variant)
                        if vec is None:
                            continue
                    elif variant == "P_matched":
                        if persona not in vectors.get("_per_persona", {}):
                            continue
                        vec = vectors["_per_persona"][persona]
                        used_persona_ref = persona
                    elif variant == "P_mismatched":
                        others = [p for p in personas_in_caw if p != persona]
                        if not others:
                            continue
                        mismatch = mismatch_rng.choice(others)
                        vec = vectors["_per_persona"][mismatch]
                        used_persona_ref = mismatch
                    elif variant in ("P_matched_A", "P_matched_B", "P_matched_C"):
                        suffix = variant.split("_")[-1]
                        store = vectors.get(f"_per_persona_{suffix}", {})
                        if persona not in store:
                            continue
                        vec = store[persona]
                        used_persona_ref = persona
                    elif variant in ("P_mismatched_A", "P_mismatched_B", "P_mismatched_C"):
                        suffix = variant.split("_")[-1]
                        store = vectors.get(f"_per_persona_{suffix}", {})
                        others = [p for p in personas_in_caw if p != persona and p in store]
                        if not others:
                            continue
                        mismatch = mismatch_rng.choice(others)
                        vec = store[mismatch]
                        used_persona_ref = mismatch
                    else:
                        continue

                    if args.normalize_vector:
                        vec = vec / (vec.norm() + 1e-12)

                    gen = generate_text(
                        model=model, tokenizer=tokenizer, prompt=prompt,
                        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                        top_p=args.top_p, top_k=args.top_k,
                        repetition_penalty=args.repetition_penalty,
                        do_sample=args.do_sample, seed=gen_seed,
                        steering_vector=vec, steering_layer=args.layer,
                        steering_alpha=alpha,
                    )
                    f.write(json.dumps({
                        "seed_id": ctx["seed_id"],
                        "run_seed": seed,
                        "persona_name": persona,
                        "target_label": ctx.get("target_label"),
                        "hauptanliegen": ctx["hauptanliegen"],
                        "context": ctx["context"],
                        "variant": variant,
                        "alpha": alpha,
                        "used_persona_ref": used_persona_ref,
                        "axis": args.axis,
                        "layer": args.layer,
                        "model_path": args.model_path,
                        "response": gen,
                    }, ensure_ascii=False) + "\n")
                    f.flush()
            if (ctx_idx + 1) % 20 == 0:
                print(f"  [seed={seed}] context {ctx_idx + 1}/{len(contexts)}", flush=True)

    print(f"Done seed={seed}. Wrote {out_path}", flush=True)


def main():
    args = parse_args()

    # Resolve seeds list
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = [args.seed]
    print(f"Seeds: {seeds}", flush=True)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]

    contexts = load_held_out(
        Path(args.held_out_path), args.held_out_shuffle_seed, args.held_out_start, args.held_out_count
    )
    print(f"Held-out contexts: {len(contexts)}", flush=True)

    # Load vectors (once — vectors don't depend on run seed)
    vectors: dict[str, torch.Tensor | dict] = {}

    if "global_blind" in variants:
        v = load_stored_v_open(Path(args.v_open_dir), args.layer)
        if v is None:
            print(f"[WARN] No stored v_open at {args.v_open_dir}/layer_{args.layer}_vectors.pt; skipping global_blind")
            variants = [x for x in variants if x != "global_blind"]
        else:
            vectors["global_blind"] = v if not args.normalize_vector else v / (v.norm() + 1e-12)

    P_VARIANTS = ("global_aware", "P_matched", "P_mismatched",
                  "P_matched_A", "P_matched_B", "P_matched_C",
                  "P_mismatched_A", "P_mismatched_B", "P_mismatched_C")
    if any(v in variants for v in P_VARIANTS):
        if not args.case_aware_cache or not args.case_aware_pairs:
            print("[WARN] --case-aware-cache and --case-aware-pairs required for aware/P variants; skipping those")
            variants = [v for v in variants if v not in P_VARIANTS]
        else:
            caw = load_case_aware_vectors_from_cache(
                Path(args.case_aware_cache), Path(args.case_aware_pairs), args.layer
            )
            if "global_aware" in variants:
                vg = caw["v_global"]
                vectors["global_aware"] = vg if not args.normalize_vector else vg / (vg.norm() + 1e-12)
            if "P_matched" in variants or "P_mismatched" in variants:
                vectors["_per_persona"] = caw["v_per_persona"]
            for suffix in ("A", "B", "C"):
                if any(f"P_matched_{suffix}" in variants or f"P_mismatched_{suffix}" in variants for _ in [0]):
                    if f"v_per_persona_{suffix}" in caw:
                        vectors[f"_per_persona_{suffix}"] = caw[f"v_per_persona_{suffix}"]

    print(f"Variants to run: {variants}")
    print(f"Alphas: {alphas}")

    # Load model (once)
    print(f"Loading model {args.model_path}...", flush=True)
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

    personas_in_caw = list(vectors.get("_per_persona", {}).keys())

    # Loop over seeds (reuse same loaded model)
    for seed in seeds:
        run_one_seed(
            seed=seed, n_seeds=len(seeds), args=args, contexts=contexts,
            vectors=vectors, variants=variants, alphas=alphas,
            personas_in_caw=personas_in_caw, model=model, tokenizer=tokenizer,
        )

    print(f"All seeds done: {seeds}", flush=True)


if __name__ == "__main__":
    main()
