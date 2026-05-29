#!/usr/bin/env python3
"""Score steering outputs with an LLM judge.

This scorer adds a bias-controlled validation layer on top of the existing
classifier/content/projection metrics. For each base-vs-steered pair it asks
the judge for:

1. pointwise rubric scores for both responses
2. a pairwise winner under the requested target style

The scorer writes per-run summaries plus a merged all_runs_summary.csv that can
be joined into benchmark leaderboards.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - environment dependent
    OpenAI = None


DIMENSIONS = (
    "axis_alignment",
    "case_fidelity",
    "client_role_fidelity",
    "training_utility",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score steering runs with an LLM judge.")
    parser.add_argument(
        "--benchmark-manifest",
        required=True,
        help="Benchmark manifest JSON created by run_steering_benchmark.py.",
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
        help="Directory for judge summaries.",
    )
    parser.add_argument(
        "--provider",
        default="openai",
        help="Judge provider. Currently only 'openai' is supported.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.4-mini",
        help="Judge model name.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Judge temperature.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap on judged examples per config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for deterministic example sampling and A/B randomization.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum API retries for one judged example.",
    )
    parser.add_argument(
        "--retry-sleep-seconds",
        type=float,
        default=2.0,
        help="Sleep time between API retries.",
    )
    parser.add_argument(
        "--rubric-version",
        default="v1",
        help="Version string to namespace cache entries and prompt variants.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Number of concurrent judge requests per config.",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=60.0,
        help="Per-request API timeout in seconds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.provider.lower() != "openai":
        raise SystemExit(f"Unsupported provider: {args.provider}")
    if OpenAI is None:
        raise SystemExit("Missing dependency: pip install openai")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.benchmark_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_by_source = {
        str(Path(run["output_path"]).resolve()): run
        for run in manifest.get("runs", [])
    }
    classifier = manifest["direction_classifier"]
    positive_label = classifier["positive_label"]
    negative_label = classifier["negative_label"]

    client = OpenAI(timeout=args.request_timeout_seconds, max_retries=0)
    all_summary_rows = []
    for score_path_raw in args.score_paths:
        score_path = Path(score_path_raw).resolve()
        run_meta = run_by_source.get(str(score_path))
        if run_meta is None:
            raise SystemExit(f"Could not find run metadata for score path: {score_path}")
        rows = load_jsonl(score_path)
        summary_rows, judged_rows = score_run(
            rows=rows,
            run_meta=run_meta,
            client=client,
            model=args.model,
            temperature=args.temperature,
            max_examples=args.max_examples,
            seed=args.seed,
            max_retries=args.max_retries,
            retry_sleep_seconds=args.retry_sleep_seconds,
            rubric_version=args.rubric_version,
            positive_label=positive_label,
            negative_label=negative_label,
            cache_dir=cache_dir,
            workers=args.workers,
        )

        stem = score_path.stem
        (output_dir / f"{stem}.summary.json").write_text(
            json.dumps(summary_rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if summary_rows:
            with (output_dir / f"{stem}.summary.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
                writer.writeheader()
                writer.writerows(summary_rows)
        (output_dir / f"{stem}.scored_rows.json").write_text(
            json.dumps(judged_rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        for row in summary_rows:
            enriched = dict(row)
            enriched["source_file"] = str(score_path)
            all_summary_rows.append(enriched)

    if all_summary_rows:
        with (output_dir / "all_runs_summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_summary_rows)

    if all_summary_rows:
        print("\nTop configs by judge steered win rate:")
        for row in sorted(
            all_summary_rows,
            key=lambda item: (
                item["pairwise_steered_win_rate"],
                item["mean_axis_alignment_delta"],
                item["mean_case_fidelity_delta"],
            ),
            reverse=True,
        )[:10]:
            print(
                row["source_file"],
                row["config_id"],
                "steered_win_rate",
                round(row["pairwise_steered_win_rate"], 4),
                "axis_delta",
                round(row["mean_axis_alignment_delta"], 4),
                "case_delta",
                round(row["mean_case_fidelity_delta"], 4),
            )


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def score_run(
    *,
    rows: list[dict],
    run_meta: dict,
    client,
    model: str,
    temperature: float,
    max_examples: int | None,
    seed: int,
    max_retries: int,
    retry_sleep_seconds: float,
    rubric_version: str,
    positive_label: str,
    negative_label: str,
    cache_dir: Path,
    workers: int,
) -> tuple[list[dict], list[dict]]:
    by_example = defaultdict(dict)
    for row in rows:
        example_index = row["example_index"]
        config_key = "base" if row["mode"] == "base" else build_config_key(row)
        by_example[example_index][config_key] = row

    config_keys = sorted({key for item in by_example.values() for key in item if key != "base"})
    judged_rows = []
    summary_rows = []

    for config_key in config_keys:
        paired_examples = []
        for example_index, item in by_example.items():
            base = item.get("base")
            steered = item.get(config_key)
            if base and steered:
                paired_examples.append((example_index, base, steered))
        paired_examples.sort(key=lambda item: item[0])
        if max_examples and len(paired_examples) > max_examples:
            rng = random.Random(seed)
            paired_examples = sorted(rng.sample(paired_examples, max_examples), key=lambda item: item[0])

        total_examples = len(paired_examples)
        config_rows = []
        if workers <= 1:
            for index, pair in enumerate(paired_examples, start=1):
                mapped_row = score_example_pair(
                    example_index=pair[0],
                    base=pair[1],
                    steered=pair[2],
                    config_key=config_key,
                    run_meta=run_meta,
                    client=client,
                    model=model,
                    temperature=temperature,
                    max_retries=max_retries,
                    retry_sleep_seconds=retry_sleep_seconds,
                    rubric_version=rubric_version,
                    positive_label=positive_label,
                    negative_label=negative_label,
                    cache_dir=cache_dir,
                    seed=seed,
                )
                config_rows.append(mapped_row)
                judged_rows.append(mapped_row)
                report_progress(run_meta["name"], config_key, index, total_examples)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        score_example_pair,
                        example_index=example_index,
                        base=base,
                        steered=steered,
                        config_key=config_key,
                        run_meta=run_meta,
                        client=client,
                        model=model,
                        temperature=temperature,
                        max_retries=max_retries,
                        retry_sleep_seconds=retry_sleep_seconds,
                        rubric_version=rubric_version,
                        positive_label=positive_label,
                        negative_label=negative_label,
                        cache_dir=cache_dir,
                        seed=seed,
                    ): example_index
                    for example_index, base, steered in paired_examples
                }
                completed = 0
                for future in as_completed(futures):
                    mapped_row = future.result()
                    config_rows.append(mapped_row)
                    judged_rows.append(mapped_row)
                    completed += 1
                    report_progress(run_meta["name"], config_key, completed, total_examples)

        config_rows.sort(key=lambda row: row["example_index"])
        judged_rows[-len(config_rows):] = config_rows

        summary_rows.append(
            summarize_config_rows(
                config_rows,
                config_id=config_key,
                axis_name=run_meta.get("axis_name") or "",
                target_style=run_meta["target_style"],
                judge_model=model,
                rubric_version=rubric_version,
            )
        )

    summary_rows.sort(
        key=lambda row: (
            row["pairwise_steered_win_rate"],
            row["mean_axis_alignment_delta"],
            row["mean_case_fidelity_delta"],
        ),
        reverse=True,
    )
    return summary_rows, judged_rows


def report_progress(run_name: str, config_key: str, completed: int, total: int) -> None:
    if completed == 1 or completed == total or completed % 10 == 0:
        print(f"[judge] {run_name} {config_key}: {completed}/{total}", flush=True)


def score_example_pair(
    *,
    example_index: int,
    base: dict,
    steered: dict,
    config_key: str,
    run_meta: dict,
    client,
    model: str,
    temperature: float,
    max_retries: int,
    retry_sleep_seconds: float,
    rubric_version: str,
    positive_label: str,
    negative_label: str,
    cache_dir: Path,
    seed: int,
) -> dict:
    base_first = randomized_base_order(seed, run_meta["name"], config_key, example_index)
    response_a = base["response"] if base_first else steered["response"]
    response_b = steered["response"] if base_first else base["response"]
    target_instruction = resolve_target_instruction(
        row=base,
        target_style=run_meta["target_style"],
        positive_label=positive_label,
        negative_label=negative_label,
    )
    payload = {
        "rubric_version": rubric_version,
        "judge_model": model,
        "axis_name": run_meta.get("axis_name") or "",
        "target_style": run_meta["target_style"],
        "target_instruction": target_instruction,
        "context": base.get("context", "") or "",
        "response_a": response_a,
        "response_b": response_b,
        "base_is_response_a": base_first,
    }
    result = evaluate_with_cache(
        client=client,
        model=model,
        temperature=temperature,
        payload=payload,
        cache_dir=cache_dir,
        max_retries=max_retries,
        retry_sleep_seconds=retry_sleep_seconds,
    )
    normalized = normalize_result(result)
    return map_result_to_base_steered(
        normalized=normalized,
        base_is_response_a=base_first,
        example_index=example_index,
        config_key=config_key,
        target_style=run_meta["target_style"],
        run_name=run_meta["name"],
        seed_id=base.get("seed_id"),
    )


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


def randomized_base_order(seed: int, run_name: str, config_id: str, example_index: int) -> bool:
    key = f"{seed}|{run_name}|{config_id}|{example_index}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 2 == 0


def resolve_target_instruction(
    *,
    row: dict,
    target_style: str,
    positive_label: str,
    negative_label: str,
) -> str:
    if target_style == positive_label:
        return row.get("positive_instruction", "") or target_style
    if target_style == negative_label:
        return row.get("negative_instruction", "") or target_style
    return row.get("neutral_instruction", "") or target_style


def evaluate_with_cache(
    *,
    client,
    model: str,
    temperature: float,
    payload: dict,
    cache_dir: Path,
    max_retries: int,
    retry_sleep_seconds: float,
) -> dict:
    cache_key = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    cache_path = cache_dir / f"{cache_key}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))["result"]

    instructions = build_system_prompt()
    prompt = build_user_prompt(payload)
    create_kwargs = {
        "model": model,
        "instructions": instructions,
        "input": prompt,
        "temperature": temperature,
    }

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.responses.create(**create_kwargs)
            result = parse_json_object(response.output_text)
            cache_path.write_text(
                json.dumps(
                    {
                        "payload": payload,
                        "result": result,
                        "raw_output_text": response.output_text,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            return result
        except Exception as exc:  # pragma: no cover - network/API failure path
            last_error = exc
            if attempt == max_retries:
                break
            time.sleep(retry_sleep_seconds)

    raise RuntimeError(f"Failed to get judge result after {max_retries} attempts.") from last_error


def build_system_prompt() -> str:
    return """You are evaluating German simulated client responses in counseling dialogues.

You must evaluate two candidate responses:
- Response A
- Response B

Return only valid JSON.

Scoring rubric:
- axis_alignment: 1-5
  1 = clearly misses the requested target style
  3 = partially reflects the target style
  5 = strongly and naturally reflects the target style
- case_fidelity: 1-5
  1 = invents or changes important facts, events, or timing
  3 = mostly faithful with some drift
  5 = fully faithful to the given counseling case
- client_role_fidelity: 1-5
  1 = sounds like a therapist/assistant/coach or breaks role
  3 = mixed or partially role-consistent
  5 = clearly sounds like the client in this dialogue
- training_utility: 1-5
  1 = poor simulation example for counselor training
  3 = somewhat usable
  5 = very useful simulation example

Pairwise decision:
- winner must be one of: A, B, tie
- choose the response that better matches the requested target style while preserving the case and staying in client role

Output schema:
{
  "response_a": {
    "axis_alignment": 1,
    "case_fidelity": 1,
    "client_role_fidelity": 1,
    "training_utility": 1,
    "notes": "short rationale"
  },
  "response_b": {
    "axis_alignment": 1,
    "case_fidelity": 1,
    "client_role_fidelity": 1,
    "training_utility": 1,
    "notes": "short rationale"
  },
  "pairwise": {
    "winner": "A",
    "confidence": 1,
    "reason": "short rationale"
  }
}
"""


def build_user_prompt(payload: dict) -> str:
    return f"""Axis: {payload['axis_name']}
Target style: {payload['target_style']}
Target instruction: {payload['target_instruction']}

Context:
{payload['context']}

Response A:
{payload['response_a']}

Response B:
{payload['response_b']}

Evaluate both responses using the rubric and then choose the better overall response for this target under case fidelity and client-role fidelity constraints.
"""


def parse_json_object(text: str) -> dict:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    if raw.startswith("{") and raw.endswith("}"):
        return json.loads(raw)
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        return json.loads(raw[start : end + 1])
    raise ValueError(f"Could not parse JSON object from output: {text}")


def normalize_result(result: dict) -> dict:
    if not isinstance(result, dict):
        raise ValueError("Judge result must be a JSON object.")
    response_a = normalize_response_scores(result.get("response_a"), label="response_a")
    response_b = normalize_response_scores(result.get("response_b"), label="response_b")
    pairwise = result.get("pairwise")
    if not isinstance(pairwise, dict):
        raise ValueError("Judge result missing pairwise object.")
    winner = str(pairwise.get("winner", "")).strip().lower()
    if winner not in {"a", "b", "tie"}:
        raise ValueError(f"Invalid pairwise winner: {winner}")
    confidence = int(pairwise.get("confidence", 1))
    confidence = max(1, min(3, confidence))
    return {
        "response_a": response_a,
        "response_b": response_b,
        "pairwise": {
            "winner": winner,
            "confidence": confidence,
            "reason": str(pairwise.get("reason", "")).strip(),
        },
    }


def normalize_response_scores(value: dict | None, *, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"Judge result missing {label} object.")
    normalized = {}
    for field in DIMENSIONS:
        raw = value.get(field)
        score = int(raw)
        if score < 1 or score > 5:
            raise ValueError(f"{label}.{field} must be in [1,5], got {score}")
        normalized[field] = score
    normalized["notes"] = str(value.get("notes", "")).strip()
    return normalized


def map_result_to_base_steered(
    *,
    normalized: dict,
    base_is_response_a: bool,
    example_index: int,
    config_key: str,
    target_style: str,
    run_name: str,
    seed_id: str | None,
) -> dict:
    base_scores = normalized["response_a"] if base_is_response_a else normalized["response_b"]
    steered_scores = normalized["response_b"] if base_is_response_a else normalized["response_a"]
    winner = normalized["pairwise"]["winner"]
    if winner == "tie":
        mapped_winner = "tie"
    elif winner == "a":
        mapped_winner = "base" if base_is_response_a else "steered"
    else:
        mapped_winner = "steered" if base_is_response_a else "base"
    row = {
        "example_index": example_index,
        "config_key": config_key,
        "run_name": run_name,
        "target_style": target_style,
        "seed_id": seed_id,
        "pairwise_winner": mapped_winner,
        "pairwise_confidence": normalized["pairwise"]["confidence"],
        "pairwise_reason": normalized["pairwise"]["reason"],
        "base_is_response_a": base_is_response_a,
        "base_notes": base_scores["notes"],
        "steered_notes": steered_scores["notes"],
    }
    for field in DIMENSIONS:
        row[f"{field}_base"] = base_scores[field]
        row[f"{field}_steered"] = steered_scores[field]
        row[f"{field}_delta"] = steered_scores[field] - base_scores[field]
    return row


def summarize_config_rows(
    rows: list[dict],
    *,
    config_id: str,
    axis_name: str,
    target_style: str,
    judge_model: str,
    rubric_version: str,
) -> dict:
    summary = {
        "config_id": config_id,
        "axis_name": axis_name,
        "target_style": target_style,
        "judge_model": judge_model,
        "rubric_version": rubric_version,
        "n_examples": len(rows),
    }
    winners = [row["pairwise_winner"] for row in rows]
    summary["pairwise_steered_win_rate"] = safe_rate(sum(w == "steered" for w in winners), len(winners))
    summary["pairwise_base_win_rate"] = safe_rate(sum(w == "base" for w in winners), len(winners))
    summary["pairwise_tie_rate"] = safe_rate(sum(w == "tie" for w in winners), len(winners))
    for field in DIMENSIONS:
        summary[f"mean_{field}_base"] = mean([row[f"{field}_base"] for row in rows])
        summary[f"mean_{field}_steered"] = mean([row[f"{field}_steered"] for row in rows])
        summary[f"mean_{field}_delta"] = mean([row[f"{field}_delta"] for row in rows])
    return summary


def safe_rate(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return count / total


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


if __name__ == "__main__":
    main()
