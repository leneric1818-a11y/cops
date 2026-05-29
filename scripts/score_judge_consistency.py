#!/usr/bin/env python3
"""Cross-judge consistency check to address W3 (self-preference bias).

Samples 30 examples per axis (120 total) from existing judge outputs,
stratified by method family and high-disagreement cases, and re-judges
them with three alternative models:
  1. Mistral-Small-3.2-24B via LiteLLM proxy
  2. Llama-3.3-70B via IONOS API
  3. GPT-OSS-120B via IONOS API

Computes per-judge agreement with the original gpt-5.4-mini judge:
  - Pairwise winner agreement rate (% same A/B/tie decision)
  - Krippendorff's alpha on axis_alignment (1-5) pointwise scores
  - Pearson r of steered win rates across runs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_ROOT = ROOT / "outputs" / "metrics" / "benchmarks"

AXIS_ORDER = ("openness", "initiative", "cooperation", "hopefulness")
FAMILY_ORDER = ("paired_dense", "prompt_diff", "reft_loreft")

AXIS_INSTRUCTIONS = {
    "openness": {
        "open": "Ziel: Die nächste Klientenäußerung soll klar Merkmale eines offenen Klienten zeigen. Die Antwort soll Gedanken, Gefühle oder Unsicherheit eher benennen, ohne neue Fakten, Ereignisse oder Zeitangaben hinzuzufügen.",
        "defensive": "Ziel: Die nächste Klientenäußerung soll klar Merkmale eines defensiven Klienten zeigen. Die Antwort soll eher zurückhaltend, selbstschützend, ausweichend oder rechtfertigend wirken, ohne neue Fakten, Ereignisse oder Zeitangaben hinzuzufügen.",
    },
    "initiative": {
        "explorative": "Ziel: Der Klient soll die nächste Äußerung durch eigene Fragen oder Reflexion aktiv mitgestalten.",
        "reactive": "Ziel: Der Klient soll minimal auf die letzte Aussage des Beraters reagieren, ohne eigene Impulse einzubringen.",
    },
    "cooperation": {
        "cooperative": "Ziel: Der Klient soll auf das Framing des Beraters eingehen und kooperativ antworten.",
        "resistant": "Ziel: Der Klient soll aktiv zurückweisen oder das Framing des Beraters ablehnen.",
    },
    "hopefulness": {
        "hopeful": "Ziel: Der Klient soll Zuversicht und Handlungsmotivation ausdrücken.",
        "resigned": "Ziel: Der Klient soll Resignation oder verlorene Handlungsfähigkeit ausdrücken.",
    },
}

ALTERNATIVE_JUDGES = [
    {
        "name": "mistral_small_24b",
        "model": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        "base_url": "https://kiz1.in.ohmportal.de/llmproxy/v1",
        "api_key_env": "LITELLM_API_KEY",
        "api_key_default": "sk-RgzbaiE9HM8w0I5IWgZz6g",
    },
    {
        "name": "llama_3_3_70b",
        "model": "meta-llama/Llama-3.3-70B-Instruct",
        "base_url": "https://openai.inference.de-txl.ionos.com/v1",
        "api_key_env": "IONOS_API_KEY",
        "api_key_default": "eyJ0eXAiOiJKV1QiLCJraWQiOiJjNjI3OWQzYS0yZjYxLTQ5ODUtYmIyYS0zM2M2MmQyZmU4N2QiLCJhbGciOiJSUzI1NiJ9.eyJpc3MiOiJpb25vc2Nsb3VkIiwiaWF0IjoxNzY5MDkwOTg0LCJjbGllbnQiOiJVU0VSIiwiaWRlbnRpdHkiOnsicHJpdmlsZWdlcyI6WyJEQVRBX0NFTlRFUl9DUkVBVEUiLCJTTkFQU0hPVF9DUkVBVEUiLCJJUF9CTE9DS19SRVNFUlZFIiwiTUFOQUdFX0RBVEFQTEFURk9STSIsIkFDQ0VTU19BQ1RJVklUWV9MT0ciLCJQQ0NfQ1JFQVRFIiwiQUNDRVNTX1MzX09CSkVDVF9TVE9SQUdFIiwiQkFDS1VQX1VOSVRfQ1JFQVRFIiwiQ1JFQVRFX0lOVEVSTkVUX0FDQ0VTUyIsIks4U19DTFVTVEVSX0NSRUFURSIsIkZMT1dfTE9HX0NSRUFURSIsIkFDQ0VTU19BTkRfTUFOQUdFX01PTklUT1JJTkciLCJBQ0NFU1NfQU5EX01BTkFHRV9DRVJUSUZJQ0FURVMiLCJBQ0NFU1NfQU5EX01BTkFHRV9MT0dHSU5HIiwiTUFOQUdFX0RCQUFTIiwiQUNDRVNTX0FORF9NQU5BR0VfRE5TIiwiTUFOQUdFX1JFR0lTVFJZIiwiQUNDRVNTX0FORF9NQU5BR0VfQ0ROIiwiQUNDRVNTX0FORF9NQU5BR0VfVlBOIiwiQUNDRVNTX0FORF9NQU5BR0VfQVBJX0dBVEVXQVkiLCJBQ0NFU1NfQU5EX01BTkFHRV9OR1MiLCJBQ0NFU1NfQU5EX01BTkFHRV9LQUFTIiwiQUNDRVNTX0FORF9NQU5BR0VfTkVUV09SS19GSUxFX1NUT1JBR0UiLCJBQ0NFU1NfQU5EX01BTkFHRV9BSV9NT0RFTF9IVUIiLCJDUkVBVEVfTkVUV09SS19TRUNVUklUWV9HUk9VUFMiLCJBQ0NFU1NfQU5EX01BTkFHRV9JQU1fUkVTT1VSQ0VTIl0sInV1aWQiOiJlZjlhNjhjMy02NTJmLTQxODAtYjZlYi05Y2JhNzc1N2RlMjgiLCJyZXNlbGxlcklkIjoxLCJyZWdEb21haW4iOiJpb25vcy5kZSIsInJvbGUiOiJvd25lciIsImNvbnRyYWN0TnVtYmVyIjozNjMwNzg3OCwiaXNQYXJlbnQiOmZhbHNlfSwiZXhwIjoxNzc2ODY2OTg0fQ.fcd1eKs-gEDjjANMPbRYaCsTnSTbAmk7pV36xzOMnM8H5zG3GToXzqMwCiuJRXiXcLPGhjE41YvnxVRoc2aDir5JqTZToToIzsCv3QYAmLaJn20Dru8joBkjnQiV-FHgJgAsR07VOCORj27pECxDIvM3oFJ2GG1p3ZXgdt-sXkXCMUWfbCPa9RSQ0lIMd3OTXHknfsGO_k0mRP8p2FzcGw0wyGs0lAgys9LeVGgEqrY6FS6ELLaSsHRw1uYfKq5xz7ryY715nJ-vgFAy_xetm3MCf8Gm1-zZLy7wdDAkfa2LyTtf76lu4mP5rFDSLPoJTfcV-CaNb5hSrK6CB1_P-A",
    },
    {
        "name": "gpt_oss_120b",
        "model": "openai/gpt-oss-120b",
        "base_url": "https://openai.inference.de-txl.ionos.com/v1",
        "api_key_env": "IONOS_API_KEY",
        "api_key_default": "eyJ0eXAiOiJKV1QiLCJraWQiOiJjNjI3OWQzYS0yZjYxLTQ5ODUtYmIyYS0zM2M2MmQyZmU4N2QiLCJhbGciOiJSUzI1NiJ9.eyJpc3MiOiJpb25vc2Nsb3VkIiwiaWF0IjoxNzY5MDkwOTg0LCJjbGllbnQiOiJVU0VSIiwiaWRlbnRpdHkiOnsicHJpdmlsZWdlcyI6WyJEQVRBX0NFTlRFUl9DUkVBVEUiLCJTTkFQU0hPVF9DUkVBVEUiLCJJUF9CTE9DS19SRVNFUlZFIiwiTUFOQUdFX0RBVEFQTEFURk9STSIsIkFDQ0VTU19BQ1RJVklUWV9MT0ciLCJQQ0NfQ1JFQVRFIiwiQUNDRVNTX1MzX09CSkVDVF9TVE9SQUdFIiwiQkFDS1VQX1VOSVRfQ1JFQVRFIiwiQ1JFQVRFX0lOVEVSTkVUX0FDQ0VTUyIsIks4U19DTFVTVEVSX0NSRUFURSIsIkZMT1dfTE9HX0NSRUFURSIsIkFDQ0VTU19BTkRfTUFOQUdFX01PTklUT1JJTkciLCJBQ0NFU1NfQU5EX01BTkFHRV9DRVJUSUZJQ0FURVMiLCJBQ0NFU1NfQU5EX01BTkFHRV9MT0dHSU5HIiwiTUFOQUdFX0RCQUFTIiwiQUNDRVNTX0FORF9NQU5BR0VfRE5TIiwiTUFOQUdFX1JFR0lTVFJZIiwiQUNDRVNTX0FORF9NQU5BR0VfQ0ROIiwiQUNDRVNTX0FORF9NQU5BR0VfVlBOIiwiQUNDRVNTX0FORF9NQU5BR0VfQVBJX0dBVEVXQVkiLCJBQ0NFU1NfQU5EX01BTkFHRV9OR1MiLCJBQ0NFU1NfQU5EX01BTkFHRV9LQUFTIiwiQUNDRVNTX0FORF9NQU5BR0VfTkVUV09SS19GSUxFX1NUT1JBR0UiLCJBQ0NFU1NfQU5EX01BTkFHRV9BSV9NT0RFTF9IVUIiLCJDUkVBVEVfTkVUV09SS19TRUNVUklUWV9HUk9VUFMiLCJBQ0NFU1NfQU5EX01BTkFHRV9JQU1fUkVTT1VSQ0VTIl0sInV1aWQiOiJlZjlhNjhjMy02NTJmLTQxODAtYjZlYi05Y2JhNzc1N2RlMjgiLCJyZXNlbGxlcklkIjoxLCJyZWdEb21haW4iOiJpb25vcy5kZSIsInJvbGUiOiJvd25lciIsImNvbnRyYWN0TnVtYmVyIjozNjMwNzg3OCwiaXNQYXJlbnQiOmZhbHNlfSwiZXhwIjoxNzc2ODY2OTg0fQ.fcd1eKs-gEDjjANMPbRYaCsTnSTbAmk7pV36xzOMnM8H5zG3GToXzqMwCiuJRXiXcLPGhjE41YvnxVRoc2aDir5JqTZToToIzsCv3QYAmLaJn20Dru8joBkjnQiV-FHgJgAsR07VOCORj27pECxDIvM3oFJ2GG1p3ZXgdt-sXkXCMUWfbCPa9RSQ0lIMd3OTXHknfsGO_k0mRP8p2FzcGw0wyGs0lAgys9LeVGgEqrY6FS6ELLaSsHRw1uYfKq5xz7ryY715nJ-vgFAy_xetm3MCf8Gm1-zZLy7wdDAkfa2LyTtf76lu4mP5rFDSLPoJTfcV-CaNb5hSrK6CB1_P-A",
    },
]

DIMENSIONS = ("axis_alignment", "case_fidelity", "client_role_fidelity", "training_utility")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-judge consistency check (W3).")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "metrics" / "judge_consistency"))
    parser.add_argument("--n-per-axis", type=int, default=30, help="Examples to sample per axis.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=3)
    return parser.parse_args()


# ---------- sampling ----------

def load_all_judge_rows(benchmarks_root: Path) -> list[dict]:
    """Load all existing scored_rows from judge outputs, enriched with axis and run metadata."""
    all_rows = []
    for bench_dir in sorted(benchmarks_root.glob("persona_*_v1__*")):
        judge_dir = bench_dir / "scoring" / "judge"
        if not judge_dir.exists():
            continue
        manifest_path = bench_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        run_meta_by_name = {r["name"]: r for r in manifest.get("runs", [])}
        axis_name = bench_dir.name.replace("persona_", "").split("_v1__")[0]
        model_name = bench_dir.name.split("_v1__")[1]

        for scored_path in sorted(judge_dir.glob("*.scored_rows.json")):
            run_stem = scored_path.stem.replace(".scored_rows", "")
            run_meta = run_meta_by_name.get(run_stem, {})
            family = run_meta.get("family") or infer_family(run_stem)
            target_style = run_meta.get("target_style", "")
            rows = json.loads(scored_path.read_text())

            # Load corresponding run JSONL for context + responses
            run_jsonl = bench_dir / "runs" / f"{run_stem}.jsonl"
            text_by_index: dict[int, dict] = {}
            if run_jsonl.exists():
                jsonl_rows = [json.loads(l) for l in run_jsonl.read_text().splitlines() if l.strip()]
                base_rows = {r["example_index"]: r for r in jsonl_rows if r.get("mode") == "base"}
                steered_rows = {r["example_index"]: r for r in jsonl_rows if r.get("mode") != "base"}
                for idx in set(base_rows) & set(steered_rows):
                    text_by_index[idx] = {
                        "context": base_rows[idx].get("context", ""),
                        "base_response": base_rows[idx].get("response", ""),
                        "steered_response": steered_rows[idx].get("response", ""),
                    }

            for row in rows:
                idx = row["example_index"]
                if idx not in text_by_index:
                    continue
                all_rows.append({
                    **row,
                    "axis_name": axis_name,
                    "model_name": model_name,
                    "family": family,
                    "target_style": target_style,
                    "run_stem": run_stem,
                    **text_by_index[idx],
                })
    return all_rows


def infer_family(run_stem: str) -> str:
    s = run_stem.lower()
    if "prompt_diff" in s:
        return "prompt_diff"
    if "loreft" in s or "reft" in s:
        return "reft_loreft"
    return "paired_dense"


def sample_examples(all_rows: list[dict], n_per_axis: int, seed: int) -> list[dict]:
    """Sample n_per_axis rows per axis, ~n/3 per family, prefer disagreement cases."""
    rng = random.Random(seed)
    sampled = []
    for axis in AXIS_ORDER:
        axis_rows = [r for r in all_rows if r["axis_name"] == axis]
        by_family: dict[str, list[dict]] = {f: [] for f in FAMILY_ORDER}
        for r in axis_rows:
            fam = r.get("family", "paired_dense")
            if fam in by_family:
                by_family[fam].append(r)

        per_family = max(1, n_per_axis // len(FAMILY_ORDER))
        for fam in FAMILY_ORDER:
            pool = by_family[fam]
            if not pool:
                continue
            # Prefer examples where pairwise_winner is steered (interesting cases)
            steered_wins = [r for r in pool if r.get("pairwise_winner") == "steered"]
            other = [r for r in pool if r.get("pairwise_winner") != "steered"]
            rng.shuffle(steered_wins)
            rng.shuffle(other)
            picks = (steered_wins + other)[:per_family]
            sampled.extend(picks)

    return sampled


# ---------- judging ----------

def build_client(judge_spec: dict):
    api_key = os.environ.get(judge_spec["api_key_env"], judge_spec["api_key_default"])
    return OpenAI(base_url=judge_spec["base_url"], api_key=api_key, timeout=90.0, max_retries=0)


def build_system_prompt() -> str:
    return """You are evaluating German simulated client responses in counseling dialogues.

Evaluate Response A and Response B using the rubric below.

Scoring rubric (each dimension 1-5):
- axis_alignment: 1=clearly misses target style, 3=partially reflects it, 5=strongly reflects it
- case_fidelity: 1=invents/changes facts, 3=mostly faithful, 5=fully faithful to the case
- client_role_fidelity: 1=sounds like therapist/assistant, 3=mixed, 5=clearly the client
- training_utility: 1=poor training example, 3=somewhat usable, 5=very useful

Pairwise decision: choose A, B, or tie for the response that better matches the target style while preserving the case.

Return ONLY valid JSON with this exact schema:
{
  "response_a": {"axis_alignment": 1, "case_fidelity": 1, "client_role_fidelity": 1, "training_utility": 1, "notes": ""},
  "response_b": {"axis_alignment": 1, "case_fidelity": 1, "client_role_fidelity": 1, "training_utility": 1, "notes": ""},
  "pairwise": {"winner": "A", "confidence": 1, "reason": ""}
}"""


def build_user_prompt(row: dict, axis_name: str, target_style: str, base_first: bool) -> str:
    instruction = AXIS_INSTRUCTIONS.get(axis_name, {}).get(target_style, target_style)
    resp_a = row["base_response"] if base_first else row["steered_response"]
    resp_b = row["steered_response"] if base_first else row["base_response"]
    return (
        f"Axis: {axis_name}\nTarget style: {target_style}\n"
        f"Target instruction: {instruction}\n\n"
        f"Context:\n{row['context']}\n\n"
        f"Response A:\n{resp_a}\n\nResponse B:\n{resp_b}\n\n"
        "Evaluate both responses and choose the better one for the target style under case and role fidelity."
    )


def parse_json_object(text: str) -> dict:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        return json.loads(raw[start:end + 1])
    raise ValueError(f"No JSON object found in: {text[:200]}")


def judge_one(
    row: dict,
    client,
    model: str,
    seed: int,
    max_retries: int,
    cache_dir: Path,
) -> dict | None:
    base_first = (int(hashlib.sha256(f"{seed}|{row['run_stem']}|{row['example_index']}".encode()).hexdigest()[:8], 16) % 2 == 0)
    prompt = build_user_prompt(row, row["axis_name"], row["target_style"], base_first)
    cache_key = hashlib.sha256(f"{model}|{prompt}".encode()).hexdigest()
    cache_path = cache_dir / f"{cache_key}.json"

    if cache_path.exists():
        return json.loads(cache_path.read_text())

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": build_system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            result = parse_json_object(response.choices[0].message.content)
            # Map A/B to base/steered
            winner_raw = str(result.get("pairwise", {}).get("winner", "")).strip().lower()
            if winner_raw == "tie":
                winner_mapped = "tie"
            elif winner_raw == "a":
                winner_mapped = "base" if base_first else "steered"
            else:
                winner_mapped = "steered" if base_first else "base"

            resp_base = result["response_a"] if base_first else result["response_b"]
            resp_steered = result["response_b"] if base_first else result["response_a"]

            out = {
                "pairwise_winner": winner_mapped,
                "base_first": base_first,
                "scores_base": {d: int(resp_base[d]) for d in DIMENSIONS if d in resp_base},
                "scores_steered": {d: int(resp_steered[d]) for d in DIMENSIONS if d in resp_steered},
            }
            cache_path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
            return out
        except Exception as exc:
            if attempt == max_retries:
                print(f"  [WARN] judge failed after {max_retries} attempts: {exc}")
                return None
            time.sleep(2.0 * attempt)
    return None


# ---------- agreement metrics ----------

def winner_agreement_rate(original_winners: list[str], new_winners: list[str]) -> float:
    if not original_winners:
        return float("nan")
    agree = sum(a == b for a, b in zip(original_winners, new_winners))
    return agree / len(original_winners)


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den > 0 else float("nan")


def krippendorff_alpha_ordinal(ratings: list[list[int]]) -> float:
    """Simplified ordinal Krippendorff's alpha for 2 raters."""
    if not ratings or len(ratings[0]) < 2:
        return float("nan")
    n = len(ratings[0])
    k = len(ratings)
    all_vals = [v for rater in ratings for v in rater]
    observed_disagreement = sum(
        (ratings[i][j] - ratings[i2][j]) ** 2
        for j in range(n)
        for i in range(k)
        for i2 in range(i + 1, k)
    ) / (n * k * (k - 1) / 2)
    mean_val = sum(all_vals) / len(all_vals)
    expected_disagreement = sum((v - mean_val) ** 2 for v in all_vals) / (len(all_vals) - 1)
    if expected_disagreement == 0:
        return 1.0
    return 1.0 - observed_disagreement / expected_disagreement


def compute_agreement(
    sampled: list[dict],
    new_results: dict[tuple, dict],
    judge_name: str,
) -> dict:
    orig_winners = []
    new_winners = []
    orig_alignment = []
    new_alignment = []

    for row in sampled:
        key = (row["run_stem"], row["example_index"])
        new = new_results.get(key)
        if new is None:
            continue
        orig_winners.append(row["pairwise_winner"])
        new_winners.append(new["pairwise_winner"])
        orig_alignment.append(row.get("axis_alignment_steered", 3))
        new_alignment.append(new["scores_steered"].get("axis_alignment", 3))

    winner_agree = winner_agreement_rate(orig_winners, new_winners)
    alpha = krippendorff_alpha_ordinal([orig_alignment, new_alignment])
    r = pearson(orig_alignment, new_alignment)

    return {
        "judge": judge_name,
        "n_judged": len(orig_winners),
        "winner_agreement_rate": round(winner_agree, 4) if not math.isnan(winner_agree) else None,
        "krippendorff_alpha_axis_alignment": round(alpha, 4) if not math.isnan(alpha) else None,
        "pearson_r_axis_alignment": round(r, 4) if not math.isnan(r) else None,
    }


# ---------- main ----------

def main() -> None:
    args = parse_args()
    if not HAS_OPENAI:
        raise SystemExit("Missing dependency: pip install openai")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading existing judge outputs...")
    all_rows = load_all_judge_rows(BENCHMARKS_ROOT)
    print(f"  Loaded {len(all_rows)} scored rows across all benchmarks")

    sampled = sample_examples(all_rows, args.n_per_axis, args.seed)
    print(f"  Sampled {len(sampled)} examples for cross-judge validation")

    # Save sample manifest
    sample_manifest = [
        {"run_stem": r["run_stem"], "example_index": r["example_index"],
         "axis_name": r["axis_name"], "family": r["family"], "pairwise_winner": r["pairwise_winner"]}
        for r in sampled
    ]
    (output_dir / "sample_manifest.json").write_text(json.dumps(sample_manifest, indent=2, ensure_ascii=False))

    all_agreement = []
    for judge_spec in ALTERNATIVE_JUDGES:
        judge_name = judge_spec["name"]
        print(f"\nRunning {judge_name} ({judge_spec['model']})...")
        cache_dir = output_dir / "cache" / judge_name
        cache_dir.mkdir(parents=True, exist_ok=True)

        client = build_client(judge_spec)
        new_results: dict[tuple, dict] = {}

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    judge_one, row, client, judge_spec["model"], args.seed, args.max_retries, cache_dir
                ): (row["run_stem"], row["example_index"])
                for row in sampled
            }
            done = 0
            for future in as_completed(futures):
                key = futures[future]
                result = future.result()
                if result:
                    new_results[key] = result
                done += 1
                if done % 20 == 0 or done == len(sampled):
                    print(f"  {done}/{len(sampled)} judged", flush=True)

        print(f"  Completed: {len(new_results)}/{len(sampled)} successful")
        agreement = compute_agreement(sampled, new_results, judge_name)
        all_agreement.append(agreement)
        print(f"  Winner agreement: {agreement['winner_agreement_rate']}, "
              f"Krippendorff α: {agreement['krippendorff_alpha_axis_alignment']}, "
              f"Pearson r: {agreement['pearson_r_axis_alignment']}")

        # Save per-judge results
        (output_dir / f"results_{judge_name}.json").write_text(
            json.dumps([{"key": list(k), "result": v} for k, v in new_results.items()], indent=2, ensure_ascii=False)
        )

    report = {
        "n_sampled": len(sampled),
        "axes": AXIS_ORDER,
        "n_per_axis": args.n_per_axis,
        "seed": args.seed,
        "agreement": all_agreement,
    }
    report_path = output_dir / "consistency_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nReport written to {report_path}")

    # Print summary table
    print("\n=== Cross-judge agreement summary ===")
    print(f"{'Judge':<25} {'Win agree':>10} {'Alpha (AA)':>12} {'Pearson r':>10}")
    for a in all_agreement:
        print(f"{a['judge']:<25} {str(a['winner_agreement_rate']):>10} "
              f"{str(a['krippendorff_alpha_axis_alignment']):>12} {str(a['pearson_r_axis_alignment']):>10}")


if __name__ == "__main__":
    main()
