#!/usr/bin/env python3
"""Score steering outputs for content preservation and likely drift.

This scorer is intentionally heuristic. It compares each steered reply against
its paired base reply and shared context, and reports:
1. lexical overlap with the base reply
2. novelty relative to base + context
3. numeric drift
4. simple yes/no polarity flips
5. a coarse likely-drift flag

The goal is not to replace human judgment but to make large steering sweeps
comparable in an automatic benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


TOKEN_RE = re.compile(r"[A-Za-zÄÖÜäöüß]+(?:-[A-Za-zÄÖÜäöüß]+)?|\d{1,4}")
NUMBER_RE = re.compile(r"\b\d{1,4}\b")
YESNO_RE = re.compile(r"^\s*(ja|nein)\b", re.IGNORECASE)
NEGATION_RE = re.compile(r"\b(nicht|kein|keine|keinen|keinem|keiner|nie|nichts|ohne)\b", re.IGNORECASE)

GERMAN_STOPWORDS = {
    "aber",
    "als",
    "also",
    "am",
    "an",
    "auch",
    "auf",
    "aus",
    "bei",
    "bin",
    "bist",
    "da",
    "dann",
    "das",
    "dass",
    "dem",
    "den",
    "der",
    "des",
    "die",
    "doch",
    "du",
    "ein",
    "eine",
    "einer",
    "einem",
    "einen",
    "er",
    "es",
    "für",
    "hat",
    "habe",
    "haben",
    "hier",
    "ich",
    "ihr",
    "ihm",
    "ihn",
    "im",
    "in",
    "ist",
    "ja",
    "mal",
    "mir",
    "mit",
    "muss",
    "müssen",
    "nicht",
    "noch",
    "nur",
    "oder",
    "schon",
    "sie",
    "sind",
    "so",
    "und",
    "uns",
    "was",
    "weil",
    "wie",
    "wir",
    "zu",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score steering outputs for content preservation and likely drift."
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
        help="Directory for content-preservation summaries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summary_rows = []
    for score_path_raw in args.score_paths:
        score_path = Path(score_path_raw)
        rows = load_jsonl(score_path)
        summary_rows, scored_rows = score_run(rows)
        stem = score_path.stem
        (output_dir / f"{stem}.summary.json").write_text(
            json.dumps(summary_rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with (output_dir / f"{stem}.summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        (output_dir / f"{stem}.scored_rows.json").write_text(
            json.dumps(scored_rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        for row in summary_rows:
            enriched = dict(row)
            enriched["source_file"] = str(score_path.resolve())
            all_summary_rows.append(enriched)

    if all_summary_rows:
        with (output_dir / "all_runs_summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_summary_rows)

    print("\nTop configs by content preservation score:")
    for row in sorted(
        all_summary_rows,
        key=lambda item: (
            item["content_preservation_score"],
            -item["drift_flag_rate"],
        ),
        reverse=True,
    )[:10]:
        print(
            row["source_file"],
            row["config_id"],
            "content",
            round(row["content_preservation_score"], 4),
            "drift",
            round(row["drift_flag_rate"], 4),
        )


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def score_run(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    by_example = defaultdict(dict)
    for row in rows:
        example_index = row["example_index"]
        config_key = "base" if row["mode"] == "base" else build_config_key(row)
        by_example[example_index][config_key] = row

    config_keys = sorted({key for item in by_example.values() for key in item if key != "base"})
    summary_rows = []
    scored_rows = []

    for config_key in config_keys:
        jaccards = []
        novelty_ratios = []
        numeric_changes = 0
        yesno_flips = 0
        negation_flips = 0
        drift_flags = 0
        n = 0

        for example_index, item in by_example.items():
            base = item.get("base")
            steered = item.get(config_key)
            if not base or not steered:
                continue

            context = base.get("context", "") or ""
            base_response = base.get("response", "") or ""
            steered_response = steered.get("response", "") or ""

            base_tokens = content_tokens(base_response)
            steered_tokens = content_tokens(steered_response)
            context_tokens = content_tokens(context)

            jaccard = jaccard_similarity(base_tokens, steered_tokens)
            novelty_ratio = novel_token_ratio(steered_tokens, base_tokens | context_tokens)
            numeric_changed = extract_numbers(base_response) != extract_numbers(steered_response)
            yesno_flipped = detect_yesno_flip(base_response, steered_response)
            negation_flipped = detect_negation_flip(base_response, steered_response)
            drift_flag = is_likely_drift(
                base_response=base_response,
                steered_response=steered_response,
                jaccard=jaccard,
                novelty_ratio=novelty_ratio,
                numeric_changed=numeric_changed,
                yesno_flipped=yesno_flipped,
                negation_flipped=negation_flipped,
            )

            jaccards.append(jaccard)
            novelty_ratios.append(novelty_ratio)
            numeric_changes += int(numeric_changed)
            yesno_flips += int(yesno_flipped)
            negation_flips += int(negation_flipped)
            drift_flags += int(drift_flag)
            n += 1

            scored_rows.append(
                {
                    "example_index": example_index,
                    "config_key": config_key,
                    "base_response": base_response,
                    "steered_response": steered_response,
                    "token_jaccard_vs_base": jaccard,
                    "novelty_ratio": novelty_ratio,
                    "numeric_changed": numeric_changed,
                    "yesno_flipped": yesno_flipped,
                    "negation_flipped": negation_flipped,
                    "drift_flag": drift_flag,
                    "seed_id": steered.get("seed_id"),
                }
            )

        mean_jaccard = average(jaccards)
        mean_novelty = average(novelty_ratios)
        drift_rate = safe_rate(drift_flags, n)
        content_score = max(
            0.0,
            min(
                1.0,
                0.6 * mean_jaccard
                + 0.2 * (1.0 - mean_novelty)
                + 0.2 * (1.0 - drift_rate),
            ),
        )
        summary_rows.append(
            {
                "config_id": config_key,
                "n_examples": n,
                "mean_token_jaccard_vs_base": mean_jaccard,
                "mean_novelty_ratio": mean_novelty,
                "numeric_change_rate": safe_rate(numeric_changes, n),
                "yesno_flip_rate": safe_rate(yesno_flips, n),
                "negation_flip_rate": safe_rate(negation_flips, n),
                "drift_flag_rate": drift_rate,
                "content_preservation_score": content_score,
            }
        )

    summary_rows.sort(key=lambda row: row["content_preservation_score"], reverse=True)
    return summary_rows, scored_rows


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
    return f"layer{layer_idx}_a{row['alpha']}"


def content_tokens(text: str) -> set[str]:
    tokens = []
    for match in TOKEN_RE.finditer(text.lower()):
        token = match.group(0)
        if token.isdigit():
            tokens.append(token)
            continue
        if len(token) <= 2:
            continue
        if token in GERMAN_STOPWORDS:
            continue
        tokens.append(token)
    return set(tokens)


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def novel_token_ratio(tokens: set[str], allowed_tokens: set[str]) -> float:
    if not tokens:
        return 0.0
    return len(tokens - allowed_tokens) / len(tokens)


def extract_numbers(text: str) -> tuple[str, ...]:
    return tuple(NUMBER_RE.findall(text))


def detect_yesno_flip(base_text: str, steered_text: str) -> bool:
    base_match = YESNO_RE.match(base_text)
    steered_match = YESNO_RE.match(steered_text)
    if not base_match or not steered_match:
        return False
    return base_match.group(1).lower() != steered_match.group(1).lower()


def detect_negation_flip(base_text: str, steered_text: str) -> bool:
    return bool(NEGATION_RE.search(base_text)) != bool(NEGATION_RE.search(steered_text))


def is_likely_drift(
    *,
    base_response: str,
    steered_response: str,
    jaccard: float,
    novelty_ratio: float,
    numeric_changed: bool,
    yesno_flipped: bool,
    negation_flipped: bool,
) -> bool:
    base_len = len(content_tokens(base_response))
    steered_len = len(content_tokens(steered_response))
    if numeric_changed:
        return True
    if yesno_flipped and jaccard < 0.5:
        return True
    if negation_flipped and jaccard < 0.35:
        return True
    if novelty_ratio > 0.6 and jaccard < 0.2:
        return True
    if base_len <= 2 and steered_len >= 6 and novelty_ratio > 0.5:
        return True
    return False


def safe_rate(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return count / total


def average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


if __name__ == "__main__":
    main()
