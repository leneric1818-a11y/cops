#!/usr/bin/env python3
"""Generate paired open/defensive client continuations WITH case knowledge.

Minimal extension of generate_steering_contrast_dataset.py:
- Input: cops_contexts.jsonl (already has hauptanliegen + steckbrief + nebenanliegen)
- Same axis config, instruction pairs, shared rules as the existing generator
- ONLY difference: the case block is prepended to the system prompt, so the
  generator can invoke case facts that are not yet visible in the history
- Output: JSONL compatible with the existing paired steering pipeline,
  plus `case_included_in_prompt: True` to track provenance
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from json import JSONDecodeError

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Reuse machinery from the existing generator so we stay aligned.
from generate_steering_contrast_dataset import (  # type: ignore
    DEFAULT_AXIS_SPEC,
    DEFAULT_SHARED_RULES,
    clean_reply,
    load_completed_ids,
    parse_json_object,
    resolve_axis_spec,
    resolve_path,
)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--contexts-path",
        default=str(ROOT / "data/processed/cops_contexts.jsonl"),
        help="Input JSONL with contexts + case metadata (from build_cops_contexts.py).",
    )
    p.add_argument(
        "--axis-config",
        default=str(ROOT / "configs/persona_axes/client_persona_axes_v1.json"),
    )
    p.add_argument("--axis-name", default="openness")
    p.add_argument("--output-path", default=str(ROOT / "outputs/metrics/persona_pairs_openness_case_aware.jsonl"))
    p.add_argument("--flat-output-path", default=None)
    p.add_argument("--model", default="gpt-5-mini")
    p.add_argument("--reasoning-effort", choices=("none", "minimal", "low", "medium", "high"), default="high")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sleep-seconds", type=float, default=0.0)
    p.add_argument("--max-retries", type=int, default=4)
    p.add_argument("--retry-sleep-seconds", type=float, default=2.0)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def load_contexts(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def format_case_block(row: dict) -> str:
    """Assemble case description in the same style as the NDAP personality_condition."""
    parts: list[str] = []
    persona = row.get("persona_name", "")
    if persona:
        parts.append(f"Name: {persona}")

    steck = row.get("steckbrief", {})
    if steck:
        bits = [f"{k}: {v}" for k, v in steck.items() if v]
        if bits:
            parts.append("Steckbrief: " + "; ".join(bits))

    ha = row.get("hauptanliegen", "")
    if ha:
        parts.append(f"Hauptanliegen:\n{ha}")

    neben = row.get("nebenanliegen", [])
    if neben:
        parts.append("Nebenanliegen:\n- " + "\n- ".join(neben))

    return "\n".join(parts)


def build_system_prompt(
    axis_spec: dict,
    shared_rules: list[str],
    pos_instruction: str,
    neg_instruction: str,
    case_block: str,
) -> str:
    """Same structure as generate_steering_contrast_dataset.build_system_prompt,
    with an additional HINTERGRUND (Case) block before the instructions."""
    negative_field = axis_spec["negative_field"]
    positive_field = axis_spec["positive_field"]
    rules = list(shared_rules)
    if axis_spec.get("negative_extra_rule"):
        rules.append(axis_spec["negative_extra_rule"])
    if axis_spec.get("positive_extra_rule"):
        rules.append(axis_spec["positive_extra_rule"])
    # Case-aware generator: allow positive pole to invoke documented case facts
    # but never invent facts outside the case.
    rules.append(
        "Die Antworten dürfen auf dokumentierte Case-Fakten zurückgreifen, "
        "aber keine Fakten erfinden die nicht im Hintergrund-Block stehen."
    )
    rule_block = "\n".join(f"- {rule}" for rule in rules)
    research_question = axis_spec.get("research_question")
    research_block = (
        f"Forschungsfrage: {research_question}\n\n" if research_question else ""
    )
    return f"""Du erzeugst kontrastive Klientenantworten für deutschsprachige Beratungsgespräche.

Ziel:
- Halte Person, Situation und Fallfakten konstant.
- Variiere primär nur die Persona-Achse "{axis_spec['name']}".
- Mache beide Pole klar erkennbar, aber natürlich und falltreu.

=== HINTERGRUND (Case) ===
{case_block}
=== ENDE HINTERGRUND ===

{research_block}Instruktionen für diese Generierung:
- {negative_field}: {neg_instruction}
- {positive_field}: {pos_instruction}

Wichtige Regeln:
{rule_block}

Ausgabeformat:
- Gib nur gültiges JSON mit den Schlüsseln {negative_field} und {positive_field} zurück.
"""


def build_user_prompt(context: str) -> str:
    # Identical to the existing generator except for the label "History".
    return f"""Gesprächsverlauf (History):
{context}

Aufgabe:
1. Erzeuge die nächste Klientenäußerung für den negativen Pol der Achse.
2. Erzeuge die nächste Klientenäußerung für den positiven Pol der Achse.
3. Beide Antworten sollen auf denselben Gesprächsverlauf folgen und sich primär in dieser Persona-Achse unterscheiden.
"""


def generate_pair(
    client,
    model: str,
    reasoning_effort: str,
    case_block: str,
    context: str,
    axis_spec: dict,
    shared_rules: list[str],
    instr_pair: dict,
    max_retries: int,
    retry_sleep_seconds: float,
) -> dict[str, str]:
    create_kwargs: dict = {
        "model": model,
        "instructions": build_system_prompt(
            axis_spec, shared_rules, instr_pair["pos"], instr_pair["neg"], case_block
        ),
        "input": build_user_prompt(context),
    }
    if reasoning_effort not in ("none", ""):
        create_kwargs["reasoning"] = {"effort": reasoning_effort}

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.responses.create(**create_kwargs)
            payload = parse_json_object(response.output_text)
            negative_field = axis_spec["negative_field"]
            positive_field = axis_spec["positive_field"]
            neg = clean_reply(payload.get(negative_field, ""))
            pos = clean_reply(payload.get(positive_field, ""))
            if not neg or not pos:
                raise ValueError(f"Missing reply in model output: {response.output_text}")
            return {negative_field: neg, positive_field: pos}
        except (JSONDecodeError, ValueError) as exc:
            last_error = exc
            if attempt == max_retries:
                break
            time.sleep(retry_sleep_seconds)
        except Exception as exc:
            last_error = exc
            if attempt == max_retries:
                break
            time.sleep(retry_sleep_seconds)

    raise RuntimeError(f"Failed to generate a valid pair after {max_retries} attempts.") from last_error


def main() -> None:
    args = parse_args()
    axis_spec, shared_rules = resolve_axis_spec(args.axis_config, args.axis_name)

    rows = load_contexts(Path(args.contexts_path))
    rows = [r for r in rows if r.get("hauptanliegen") and r.get("context")]
    print(f"Usable contexts with hauptanliegen: {len(rows)}")

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(rows)

    selected = rows[args.offset : args.offset + args.limit]
    if not selected:
        raise SystemExit("No contexts selected after applying offset/limit.")

    instruction_pairs: list[dict] = axis_spec.get("instruction_pairs") or []
    if not instruction_pairs:
        pos_instr = axis_spec.get("positive_instruction") or axis_spec["positive_description"]
        neg_instr = axis_spec.get("negative_instruction") or axis_spec["negative_description"]
        instruction_pairs = [{"pos": pos_instr, "neg": neg_instr}]
    num_pairs = len(instruction_pairs)

    output_path = resolve_path(args.output_path)
    flat_output_path = resolve_path(
        args.flat_output_path or f"{output_path.with_suffix('')}_flat.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    flat_output_path.parent.mkdir(parents=True, exist_ok=True)

    completed_ids = load_completed_ids(output_path) if args.resume else set()
    pending_combos: list[dict] = []
    for item in selected:
        for pair_idx, instr_pair in enumerate(instruction_pairs):
            combo_seed_id = f"{item['seed_id']}_i{pair_idx}"
            if combo_seed_id not in completed_ids:
                pending_combos.append({
                    **item,
                    "seed_id": combo_seed_id,
                    "instruction_pair_idx": pair_idx,
                    "instr_pair": instr_pair,
                })

    total_target = len(selected) * num_pairs
    print(
        f"Selected contexts: {len(selected)} | Instruction pairs: {num_pairs} | "
        f"Total target: {total_target} | Pending after resume filter: {len(pending_combos)}"
    )

    if args.dry_run:
        for combo in pending_combos[: min(3, len(pending_combos))]:
            print("=" * 72)
            print(f"{combo['seed_id']}  (instr pair {combo['instruction_pair_idx']})")
            print("\n--- case block ---")
            print(format_case_block(combo))
            print("\n--- system prompt (first 800 chars) ---")
            print(build_system_prompt(
                axis_spec, shared_rules,
                combo["instr_pair"]["pos"], combo["instr_pair"]["neg"],
                format_case_block(combo),
            )[:800])
            print("\n--- context ---")
            print(combo["context"])
        return

    if OpenAI is None:
        raise SystemExit("Missing dependency: pip install openai")

    client = OpenAI()

    negative_field = axis_spec["negative_field"]
    positive_field = axis_spec["positive_field"]
    negative_style = axis_spec["negative_style"]
    positive_style = axis_spec["positive_style"]

    with output_path.open("a", encoding="utf-8") as paired_handle, \
         flat_output_path.open("a", encoding="utf-8") as flat_handle:
        for idx, combo in enumerate(pending_combos):
            case_block = format_case_block(combo)
            try:
                pair = generate_pair(
                    client=client,
                    model=args.model,
                    reasoning_effort=args.reasoning_effort,
                    case_block=case_block,
                    context=combo["context"],
                    axis_spec=axis_spec,
                    shared_rules=shared_rules,
                    instr_pair=combo["instr_pair"],
                    max_retries=args.max_retries,
                    retry_sleep_seconds=args.retry_sleep_seconds,
                )
            except Exception as exc:
                print(f"  [FAILED] {combo['seed_id']}: {exc}")
                continue

            paired_row = {
                "seed_id": combo["seed_id"],
                "seed_record_index": combo["original_index"],
                "instruction_pair_idx": combo["instruction_pair_idx"],
                "axis_name": axis_spec["name"],
                "negative_style": negative_style,
                "positive_style": positive_style,
                "negative_field": negative_field,
                "positive_field": positive_field,
                "persona_name": combo.get("persona_name"),
                "target_label": combo.get("target_label"),
                "source_dataset": str(args.contexts_path),
                "source": "synthetic_openai_case_aware",
                "generation_model": args.model,
                "reasoning_effort": args.reasoning_effort,
                "case_included_in_prompt": True,
                "hauptanliegen": combo["hauptanliegen"],
                "context": combo["context"],
                negative_field: pair[negative_field],
                positive_field: pair[positive_field],
            }
            paired_handle.write(json.dumps(paired_row, ensure_ascii=False) + "\n")
            paired_handle.flush()

            for style_key, field in [(negative_style, negative_field), (positive_style, positive_field)]:
                flat_row = {
                    "seed_id": combo["seed_id"],
                    "seed_record_index": combo["original_index"],
                    "instruction_pair_idx": combo["instruction_pair_idx"],
                    "persona_name": combo.get("persona_name"),
                    "target_label": combo.get("target_label"),
                    "source_dataset": str(args.contexts_path),
                    "source": "synthetic_openai_case_aware",
                    "generation_model": args.model,
                    "reasoning_effort": args.reasoning_effort,
                    "case_included_in_prompt": True,
                    "axis_name": axis_spec["name"],
                    "style": style_key,
                    "context": combo["context"],
                    "response": pair[field],
                }
                flat_handle.write(json.dumps(flat_row, ensure_ascii=False) + "\n")
            flat_handle.flush()

            print(f"  [{idx + 1}/{len(pending_combos)}] {combo['seed_id']} ok", flush=True)
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    print(f"\nPaired output: {output_path}")
    print(f"Flat output:   {flat_output_path}")


if __name__ == "__main__":
    main()
