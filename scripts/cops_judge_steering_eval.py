#!/usr/bin/env python3
"""LLM-Judge for COPS steering eval — thin adapter around score_llm_judge helpers.

For each (context, run_seed), forms (baseline, steered) pairs and runs the
gpt-5.4-mini pointwise+pairwise rubric on each pair.

Outputs a scored JSONL and a summary JSON per eval file:
- pointwise rubric means for base/steered (axis_alignment, case_fidelity, client_role_fidelity, training_utility)
- pairwise steered-win-rate vs baseline
- per-seed aggregation with mean+std over seeds

Caches each API call under outputs/steering_eval/judge_cache/<sha256>.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
AXIS_CONFIG_PATH = ROOT / "configs" / "persona_axes" / "client_persona_axes_v1.json"

# Reuse helpers from the canonical judge script
from score_llm_judge import (  # type: ignore
    build_system_prompt,
    build_user_prompt,
    parse_json_object,
    normalize_result,
)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def load_axis_specs(config_path: Path = AXIS_CONFIG_PATH) -> dict[str, dict[str, dict[str, str]]]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    axis_specs: dict[str, dict[str, dict[str, str]]] = {}
    for axis in payload.get("axes", []):
        name = axis["name"]
        axis_specs[name] = {
            "positive": {
                "style": axis["positive_style"],
                "instruction": axis["positive_instruction"],
            },
            "negative": {
                "style": axis["negative_style"],
                "instruction": axis["negative_instruction"],
            },
        }
    return axis_specs


AXIS_SPECS = load_axis_specs()


def resolve_axis_target(axis: str, alpha: float) -> dict[str, str]:
    try:
        axis_spec = AXIS_SPECS[axis]
    except KeyError as exc:
        raise KeyError(f"Unknown axis for judge target resolution: {axis}") from exc
    pole = "negative" if float(alpha) < 0 else "positive"
    target = axis_spec[pole]
    return {
        "target_pole": pole,
        "target_style": target["style"],
        "target_instruction": target["instruction"],
    }


def cache_key_for(payload: dict, model: str, temperature: float) -> str:
    """Deterministic cache key over payload + model parameters."""
    raw = json.dumps(
        {"payload": payload, "model": model, "temperature": temperature},
        ensure_ascii=False, sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def group_by_seed_id(rows: list[dict]) -> dict:
    """Return dict[(run_seed, seed_id)] = {'baseline': row, 'steered': [rows]}."""
    groups: dict = defaultdict(lambda: {"baseline": None, "steered": []})
    for r in rows:
        key = (r.get("run_seed", 42), r["seed_id"])
        if r["variant"] == "baseline":
            groups[key]["baseline"] = r
        else:
            groups[key]["steered"].append(r)
    return groups


def _call_judge_api(client, provider: str, model: str, system: str, user: str, temperature: float, reasoning_effort: str | None = None) -> str:
    """Route API call based on provider. Returns the raw judge text output."""
    if provider == "openai":
        kwargs = {"model": model, "instructions": system, "input": user}
        if temperature > 0:
            kwargs["temperature"] = temperature
        if reasoning_effort:
            kwargs["reasoning"] = {"effort": reasoning_effort}
        resp = client.responses.create(**kwargs)
        return resp.output_text
    elif provider in ("cluster_mistral", "ionos"):
        # Chat-completions API on cluster-hosted Mistral.
        # Magistral emits its output in `reasoning_content` (thinking model); content is None.
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        kwargs = {
            "model": model,
            "messages": messages,
            "seed": 42,
            "max_tokens": 4096,
        }
        if provider == "cluster_mistral":
            kwargs["metadata"] = {"tags": ["Technische Hochschule Nürnberg", "anon@example.com"]}
        if temperature > 0:
            kwargs["temperature"] = temperature
        if reasoning_effort and provider == "ionos":
            kwargs["reasoning_effort"] = reasoning_effort
        resp = client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        # Prefer explicit content; fall back to reasoning_content for thinking models
        text = getattr(msg, "content", None)
        if not text:
            text = getattr(msg, "reasoning_content", None)
        if not text:
            psf = getattr(msg, "provider_specific_fields", None) or {}
            text = psf.get("reasoning_content")
        if not text:
            raise ValueError(f"Mistral returned no usable text: {msg}")
        return text
    else:
        raise ValueError(f"Unknown judge provider: {provider}")


def judge_one_pair(
    client,
    *,
    provider: str,
    model: str,
    temperature: float,
    payload: dict,
    cache_dir: Path,
    max_retries: int = 3,
    retry_sleep: float = 2.0,
    reasoning_effort: str | None = None,
) -> dict:
    """Return the parsed judge result, using cache when possible."""
    # Cache key includes provider + reasoning_effort so ensembles/settings don't collide.
    cache_payload = {"provider": provider, **payload}
    if reasoning_effort:
        cache_payload["reasoning_effort"] = reasoning_effort
    key = cache_key_for(cache_payload, model, temperature)
    cache_file = cache_dir / f"{key}.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    system = build_system_prompt()
    user = build_user_prompt(payload)

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            raw_text = _call_judge_api(client, provider, model, system, user, temperature, reasoning_effort=reasoning_effort)
            parsed = parse_json_object(raw_text)
            normalized = normalize_result(parsed)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(normalized, ensure_ascii=False), encoding="utf-8")
            return normalized
        except Exception as exc:
            last_err = exc
            if attempt == max_retries:
                break
            time.sleep(retry_sleep)
    raise RuntimeError(f"Judge failed after {max_retries} attempts: {last_err}")


def build_judge_client(provider: str):
    """Build OpenAI-compatible client for the requested provider."""
    import os
    if OpenAI is None:
        raise SystemExit("pip install openai")
    if provider == "openai":
        return OpenAI()
    elif provider == "cluster_mistral":
        api_key = os.environ.get("CLUSTER_API_KEY")
        if not api_key:
            raise SystemExit("CLUSTER_API_KEY env var required for cluster_mistral provider")
        return OpenAI(
            api_key=api_key,
            base_url="https://kiz1.in.ohmportal.de/llmproxy/v1",
        )
    elif provider == "ionos":
        api_key = os.environ.get("IONOS_API_KEY")
        if not api_key:
            raise SystemExit("IONOS_API_KEY env var required for ionos provider")
        return OpenAI(
            api_key=api_key,
            base_url="https://openai.inference.de-txl.ionos.com/v1",
        )
    else:
        raise ValueError(f"Unknown judge provider: {provider}")


def judge_eval_file(
    eval_jsonl: Path,
    out_dir: Path,
    cache_dir: Path,
    *,
    provider: str = "openai",
    model: str = "gpt-5.4-mini",
    temperature: float = 0.0,
    reasoning_effort: str | None = None,
    workers: int = 6,
    max_examples: int | None = None,
    randomize_order_seed: int = 42,
) -> dict:
    client = build_judge_client(provider)

    rows = load_rows(eval_jsonl)
    if not rows:
        print(f"  [skip] empty: {eval_jsonl}")
        return {}

    axis = rows[0]["axis"]
    model_path = rows[0]["model_path"]

    # For each (run_seed, seed_id): baseline vs steered
    groups = group_by_seed_id(rows)

    # Flatten into per-pair jobs
    rng = random.Random(randomize_order_seed)
    jobs: list[dict] = []
    for (run_seed, sid), entry in groups.items():
        baseline = entry["baseline"]
        if baseline is None:
            continue
        for steered in entry["steered"]:
            target = resolve_axis_target(axis, steered["alpha"])
            # Randomize A/B assignment per example for bias control
            a_is_steered = rng.random() < 0.5
            if a_is_steered:
                response_a = steered["response"]
                response_b = baseline["response"]
            else:
                response_a = baseline["response"]
                response_b = steered["response"]
            payload = {
                "axis_name": axis,
                "target_style": target["target_style"],
                "target_instruction": target["target_instruction"],
                "context": steered["context"],
                "response_a": response_a,
                "response_b": response_b,
            }
            jobs.append({
                "run_seed": run_seed,
                "seed_id": sid,
                "variant": steered["variant"],
                "alpha": steered["alpha"],
                "a_is_steered": a_is_steered,
                "payload": payload,
                "baseline_response": baseline["response"],
                "steered_response": steered["response"],
                "persona_name": steered.get("persona_name"),
                "hauptanliegen": steered.get("hauptanliegen"),
                "used_persona_ref": steered.get("used_persona_ref"),
                "target_pole": target["target_pole"],
                "target_style": target["target_style"],
                "target_instruction": target["target_instruction"],
            })

    if max_examples is not None:
        jobs = jobs[:max_examples]
    total = len(jobs)
    print(f"  {total} (baseline, steered) pairs to judge")

    scored: list[dict] = []
    completed = 0

    def _run(job):
        result = judge_one_pair(client, provider=provider, model=model, temperature=temperature, payload=job["payload"], cache_dir=cache_dir, reasoning_effort=reasoning_effort)
        # Map A/B back to steered/base
        if job["a_is_steered"]:
            steered_scores = result.get("response_a", {})
            base_scores = result.get("response_b", {})
            pairwise = result.get("pairwise", {})
            raw_winner = str(pairwise.get("winner", "tie")).lower()
            winner = {"a": "steered", "b": "base", "tie": "tie"}.get(raw_winner, "tie")
        else:
            steered_scores = result.get("response_b", {})
            base_scores = result.get("response_a", {})
            pairwise = result.get("pairwise", {})
            raw_winner = str(pairwise.get("winner", "tie")).lower()
            winner = {"a": "base", "b": "steered", "tie": "tie"}.get(raw_winner, "tie")
        return {
            "run_seed": job["run_seed"],
            "seed_id": job["seed_id"],
            "variant": job["variant"],
            "alpha": job["alpha"],
            "persona_name": job["persona_name"],
            "used_persona_ref": job["used_persona_ref"],
            "hauptanliegen": job["hauptanliegen"],
            "steered_scores": steered_scores,
            "base_scores": base_scores,
            "pairwise_winner": winner,
            "pairwise_confidence": pairwise.get("confidence"),
            "pairwise_reason": pairwise.get("reason"),
            "axis": axis,
            "model_path": model_path,
            "target_pole": job["target_pole"],
            "target_style": job["target_style"],
            "judge_provider": provider,
            "judge_model": model,
        }

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_run, job) for job in jobs]
        for fut in as_completed(futures):
            try:
                scored.append(fut.result())
            except Exception as exc:
                print(f"  [ERROR] {exc}")
            completed += 1
            if completed % 25 == 0 or completed == total:
                print(f"  judged {completed}/{total}", flush=True)

    # Write scored JSONL — include provider tag so multi-provider ensembles don't overwrite
    out_dir.mkdir(parents=True, exist_ok=True)
    # Sanitize model name for filename (strip slashes + colons)
    model_tag = model.replace("/", "_").replace(":", "_")
    effort_tag = f"_{reasoning_effort}" if reasoning_effort else ""
    provider_suffix = f"__{provider}__{model_tag}{effort_tag}"
    # Keep the old filename for gpt-5.4-mini/openai default (backward compat)
    is_default_openai = (provider == "openai" and model == "gpt-5.4-mini" and not reasoning_effort)
    suffix = "" if is_default_openai else provider_suffix
    out_jsonl = out_dir / (eval_jsonl.stem + "_judged" + suffix + ".jsonl")
    with out_jsonl.open("w", encoding="utf-8") as f:
        for s in scored:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Aggregate per (variant, alpha) with per-seed variance
    by_va: dict = defaultdict(list)
    by_va_seed: dict = defaultdict(list)
    for s in scored:
        k = f"{s['variant']}__a{s['alpha']}"
        by_va[k].append(s)
        by_va_seed[(s["variant"], s["alpha"], s["run_seed"])].append(s)

    def _finite(xs):
        return [float(x) for x in xs if isinstance(x, (int, float)) and np.isfinite(x)]

    def _safe_mean(xs):
        vals = _finite(xs)
        return float(np.mean(vals)) if vals else float("nan")

    def _safe_std(xs):
        vals = _finite(xs)
        return float(np.std(vals, ddof=1)) if len(vals) >= 2 else float("nan")

    def _mean_dim(rows, side, dim):
        xs = [r[side].get(dim) for r in rows if isinstance(r[side].get(dim), (int, float))]
        return float(np.mean(xs)) if xs else float("nan")

    agg: dict = {}
    for key, group in sorted(by_va.items()):
        variant, alpha_str = key.split("__a")
        alpha_val = float(alpha_str)
        n = len(group)
        winners = [r["pairwise_winner"] for r in group]
        win_rate = winners.count("steered") / max(1, n)
        tie_rate = winners.count("tie") / max(1, n)
        base_rate = winners.count("base") / max(1, n)

        # Dimension deltas (steered − base) AND per-dimension pairwise win rates
        # Pairwise-from-pointwise: steered_wins_on_dim ⟺ steered.score[dim] > base.score[dim]
        dim_deltas = {}
        dim_pairwise = {}
        for dim in ("axis_alignment", "case_fidelity", "client_role_fidelity", "training_utility"):
            valid = [
                (r["steered_scores"].get(dim), r["base_scores"].get(dim))
                for r in group
                if isinstance(r["steered_scores"].get(dim), (int, float))
                and isinstance(r["base_scores"].get(dim), (int, float))
            ]
            deltas = [s - b for s, b in valid]
            dim_deltas[f"{dim}_delta_mean"] = float(np.mean(deltas)) if deltas else float("nan")
            dim_deltas[f"{dim}_delta_std_pooled"] = float(np.std(deltas)) if deltas else float("nan")

            # Pairwise derived: per-dimension win rates (tie when scores equal)
            steered_wins = sum(1 for s, b in valid if s > b)
            base_wins = sum(1 for s, b in valid if s < b)
            ties = sum(1 for s, b in valid if s == b)
            tot = max(1, len(valid))
            dim_pairwise[f"{dim}_pairwise_steered_win_rate"] = steered_wins / tot
            dim_pairwise[f"{dim}_pairwise_base_win_rate"] = base_wins / tot
            dim_pairwise[f"{dim}_pairwise_tie_rate"] = ties / tot
            # Net preference: (wins - losses) / total — range [-1, +1]
            dim_pairwise[f"{dim}_pairwise_net"] = (steered_wins - base_wins) / tot

        # Per-seed stats — including per-dimension pairwise
        run_seed_values = sorted({r["run_seed"] for r in group})
        per_seed_win_rates = []
        per_seed_case_fidelity_delta = []
        per_seed_dim_net: dict = defaultdict(list)
        for rs in run_seed_values:
            seed_group = by_va_seed.get((variant, alpha_val, rs), [])
            if not seed_group:
                continue
            sw = [r["pairwise_winner"] for r in seed_group]
            per_seed_win_rates.append(sw.count("steered") / max(1, len(sw)))
            case_del = [
                r["steered_scores"].get("case_fidelity") - r["base_scores"].get("case_fidelity")
                for r in seed_group
                if isinstance(r["steered_scores"].get("case_fidelity"), (int, float))
                and isinstance(r["base_scores"].get("case_fidelity"), (int, float))
            ]
            if case_del:
                per_seed_case_fidelity_delta.append(float(np.mean(case_del)))
            else:
                per_seed_case_fidelity_delta.append(float("nan"))
            # Per-seed per-dim net pairwise (steered wins − base wins) / N
            for dim in ("axis_alignment", "case_fidelity", "client_role_fidelity", "training_utility"):
                valid = [
                    (r["steered_scores"].get(dim), r["base_scores"].get(dim))
                    for r in seed_group
                    if isinstance(r["steered_scores"].get(dim), (int, float))
                    and isinstance(r["base_scores"].get(dim), (int, float))
                ]
                if valid:
                    sw_d = sum(1 for s, b in valid if s > b)
                    bw_d = sum(1 for s, b in valid if s < b)
                    per_seed_dim_net[dim].append((sw_d - bw_d) / len(valid))
                else:
                    per_seed_dim_net[dim].append(float("nan"))

        per_seed_dim_net_mean = {
            f"{d}_pairwise_net_mean_over_seeds": _safe_mean(per_seed_dim_net[d])
            for d in ("axis_alignment", "case_fidelity", "client_role_fidelity", "training_utility")
        }
        per_seed_dim_net_std = {
            f"{d}_pairwise_net_std_over_seeds": _safe_std(per_seed_dim_net[d])
            for d in ("axis_alignment", "case_fidelity", "client_role_fidelity", "training_utility")
        }

        agg[key] = {
            "n": n,
            "target_pole": group[0].get("target_pole"),
            "target_style": group[0].get("target_style"),
            "pairwise_steered_win_rate": win_rate,
            "pairwise_tie_rate": tie_rate,
            "pairwise_base_win_rate": base_rate,
            "pairwise_net_overall": (winners.count("steered") - winners.count("base")) / max(1, n),
            **dim_deltas,
            **dim_pairwise,
            "run_seed_values": run_seed_values,
            "win_rate_per_seed": per_seed_win_rates,
            "win_rate_mean_over_seeds": _safe_mean(per_seed_win_rates),
            "win_rate_std_over_seeds": _safe_std(per_seed_win_rates),
            "case_fidelity_delta_per_seed": per_seed_case_fidelity_delta,
            "case_fidelity_delta_mean_over_seeds": _safe_mean(per_seed_case_fidelity_delta),
            "case_fidelity_delta_std_over_seeds": _safe_std(per_seed_case_fidelity_delta),
            "axis_alignment_pairwise_net_per_seed": per_seed_dim_net["axis_alignment"],
            "case_fidelity_pairwise_net_per_seed": per_seed_dim_net["case_fidelity"],
            "client_role_fidelity_pairwise_net_per_seed": per_seed_dim_net["client_role_fidelity"],
            "training_utility_pairwise_net_per_seed": per_seed_dim_net["training_utility"],
            **per_seed_dim_net_mean,
            **per_seed_dim_net_std,
        }

    failed_jobs = total - len(scored)
    run_seeds = sorted({s["run_seed"] for s in scored})
    summary = {
        "eval_file": str(eval_jsonl),
        "judge_provider": provider,
        "judge_model": model,
        "reasoning_effort": reasoning_effort,
        "axis": axis,
        "model_path": model_path,
        "run_seeds": run_seeds,
        "n_seeds_detected": len(run_seeds),
        "n_pairs": len(scored),
        "n_pairs_attempted": total,
        "n_pairs_failed": failed_jobs,
        "by_variant_alpha": agg,
    }
    summary_path = out_dir / (eval_jsonl.stem + "_judge_summary" + suffix + ".json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if failed_jobs:
        print(f"  [WARN] skipped {failed_jobs}/{total} failed judgements", flush=True)
    print(f"  wrote {out_jsonl.name} + {summary_path.name}")
    return summary


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--eval-dir",
        default=str(ROOT / "outputs/steering_eval"),
    )
    p.add_argument(
        "--out-dir",
        default=str(ROOT / "outputs/steering_eval/judged"),
    )
    p.add_argument(
        "--cache-dir",
        default=str(ROOT / "outputs/steering_eval/judge_cache"),
    )
    p.add_argument("--provider", default="openai", choices=("openai", "cluster_mistral", "ionos"))
    p.add_argument("--model", default="gpt-5.4-mini")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--reasoning-effort", default=None, choices=(None, "minimal", "low", "medium", "high"), help="OpenAI/Ionos reasoning effort for thinking models (e.g. gpt-5.4-nano, gpt-oss-120b).")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--max-examples-per-file", type=int, default=None)
    p.add_argument("--file-pattern", default="*.jsonl")
    p.add_argument("--skip-pattern", default="SMOKE_")
    p.add_argument("--single-file", default=None, help="Run on a single eval jsonl instead of scanning eval-dir.")
    return p.parse_args()


def main():
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir)
    cache_dir = Path(args.cache_dir)

    if args.single_file:
        files = [Path(args.single_file)]
    else:
        files = [
            p for p in sorted(eval_dir.glob(args.file_pattern))
            if not (args.skip_pattern and args.skip_pattern in p.name)
            and "_scored" not in p.name and "_judged" not in p.name
        ]
    print(f"Will judge {len(files)} eval files")

    summaries = []
    for j in files:
        print(f"\n== {j.name} ==")
        summary = judge_eval_file(
            j, out_dir, cache_dir,
            provider=args.provider, model=args.model, temperature=args.temperature,
            reasoning_effort=args.reasoning_effort,
            workers=args.workers, max_examples=args.max_examples_per_file,
        )
        if summary:
            summaries.append(summary)

    master = out_dir / "all_judge_summaries.json"
    master.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nMaster: {master}")


if __name__ == "__main__":
    main()
