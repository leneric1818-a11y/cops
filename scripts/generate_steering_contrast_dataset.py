#!/usr/bin/env python3
"""Build paired client continuations for configurable persona axes.

This script uses counseling dialogue contexts from the repo dataset as seeds and
asks an OpenAI model to produce one binary contrast pair for a configurable
persona axis, e.g.:

- defensive vs open
- reactive vs explorative
- resistant vs cooperative
- resigned vs hopeful
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

from llm_ft_comparison.data import extract_history, load_records

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - runtime dependency
    OpenAI = None


TURN_WITH_LABEL_RE = re.compile(r"^\s*([A-Za-zÄÖÜäöüß]+)\s*\([^)]*\)\s*:\s*(.+?)\s*$")
TURN_WITHOUT_LABEL_RE = re.compile(r"^\s*([A-Za-zÄÖÜäöüß]+)\s*:\s*(.+?)\s*$")
SPEAKER_MAP = {
    "K": "Klient",
    "B": "Berater",
}
DEFAULT_AXIS_SPEC = {
    "name": "openness",
    "negative_style": "defensive",
    "positive_style": "open",
    "negative_field": "defensive_response",
    "positive_field": "open_response",
    "negative_description": "zurückhaltender, selbstschützender, ausweichender oder rechtfertigender",
    "positive_description": "offener, reflektierter und eher bereit, Gefühle oder Unsicherheit zu benennen",
    "negative_instruction": "Ziel: Die nächste Klientenäußerung soll klar Merkmale eines defensiven Klienten zeigen. Die Antwort soll eher zurückhaltend, selbstschützend, ausweichend oder rechtfertigend wirken, ohne neue Fakten, Ereignisse oder Zeitangaben hinzuzufügen. Der Klient bleibt in seiner Rolle und antwortet nur mit der nächsten Äußerung.",
    "positive_instruction": "Ziel: Die nächste Klientenäußerung soll klar Merkmale eines offenen Klienten zeigen. Die Antwort soll Gedanken, Gefühle oder Unsicherheit eher benennen, ohne neue Fakten, Ereignisse oder Zeitangaben hinzuzufügen. Der Klient bleibt in seiner Rolle und antwortet nur mit der nächsten Äußerung.",
    "neutral_instruction": "Ziel: Der Klient antwortet als nächste Äußerung auf die letzte Aussage des Beraters. Bleibe eng am gegebenen Fallkontext und verändere keine bekannten Fakten, Zeitangaben oder Ereignisse.",
    "positive_extra_rule": "Die offene Antwort darf Gefühle oder Unsicherheit klarer benennen, aber keine neuen konkreten Verhaltensdetails oder Ereignisse hinzufügen.",
    "research_question": "Kann ich Klientenoffenheit kontrollieren, ohne den Fallinhalt zu verzerren?",
}
DEFAULT_SHARED_RULES = [
    "Beide Antworten müssen zur gleichen Person und zur gleichen Situation passen.",
    "Die beiden Antworten sollen sich primär in der benannten Persona-Achse unterscheiden, nicht in den Fakten des Falls.",
    "Der Unterschied soll spürbar, aber nicht karikaturenhaft sein.",
    "Keine übertriebenen \"Therapie-Demo\"-Antworten.",
    "Keine großen neuen Fakten, Diagnosen, Ortswechsel oder Nebenhandlungen erfinden.",
    "Natürliches gesprochenes Deutsch, 1 bis 3 Sätze je Antwort.",
    "Keine Sprecherlabels, keine Aufzählungen, keine Erklärungen, keine Metakommentare.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate paired defensive/open client replies from counseling contexts."
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Input JSON/JSONL dataset with conversation histories.",
    )
    parser.add_argument(
        "--axis-config",
        default=None,
        help="Optional JSON file with persona-axis definitions. Falls back to the built-in openness axis.",
    )
    parser.add_argument(
        "--axis-name",
        default=None,
        help="Axis name inside --axis-config. Defaults to the built-in openness axis.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="JSONL output path for paired examples.",
    )
    parser.add_argument(
        "--flat-output-path",
        default=None,
        help="Optional JSONL output path with one row per style. Defaults to '<output>_flat.jsonl'.",
    )
    parser.add_argument(
        "--model",
        default="gpt-5-mini",
        help="OpenAI model name. Pass gpt-5.4-mini explicitly if your account exposes it.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high"),
        default="high",
        help="Reasoning effort for supported reasoning models. Use 'none' for models that don't support it (e.g. gpt-4.1-mini).",
    )
    parser.add_argument(
        "--context-field",
        default=None,
        help="Optional field name to use instead of auto-detecting a history/text field.",
    )
    parser.add_argument(
        "--strip-history-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove annotation labels like 'K (K-... | ...):' from the seed history.",
    )
    parser.add_argument(
        "--require-last-speaker",
        choices=("Klient", "Berater", "none"),
        default="Berater",
        help="Keep only contexts whose last line belongs to this speaker.",
    )
    parser.add_argument(
        "--min-turns",
        type=int,
        default=3,
        help="Minimum number of non-empty dialogue lines required in a context.",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=1000,
        help="Discard contexts longer than this many characters after cleaning. Use 0 to disable.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum number of paired contexts to generate.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Number of usable contexts to skip before sampling.",
    )
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shuffle usable contexts before selecting the generation slice.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for shuffling.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional pause between API calls.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="How often to retry a seed when the API returns malformed output or a transient error.",
    )
    parser.add_argument(
        "--retry-sleep-seconds",
        type=float,
        default=2.0,
        help="Pause between retries for one seed.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip seed examples that are already present in the output file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected contexts without calling the API.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    axis_spec, shared_rules = resolve_axis_spec(args.axis_config, args.axis_name)

    records = load_records(args.data_path)
    seeded_items = extract_seed_items(
        records=records,
        context_field=args.context_field,
        strip_history_labels=args.strip_history_labels,
        require_last_speaker=None if args.require_last_speaker == "none" else args.require_last_speaker,
        min_turns=args.min_turns,
        max_context_chars=args.max_context_chars,
    )
    print(f"Usable seed contexts after filtering: {len(seeded_items)} / {len(records)}")
    if not seeded_items:
        raise SystemExit("No usable contexts found.")

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(seeded_items)

    selected = seeded_items[args.offset : args.offset + args.limit]
    if not selected:
        raise SystemExit("No contexts selected after applying offset/limit.")

    # Resolve instruction pairs: use axis_spec["instruction_pairs"] if available,
    # otherwise fall back to the single pos/neg instruction fields.
    instruction_pairs: list[dict] = axis_spec.get("instruction_pairs") or []
    if not instruction_pairs:
        pos_instr = axis_spec.get("positive_instruction") or axis_spec["positive_description"]
        neg_instr = axis_spec.get("negative_instruction") or axis_spec["negative_description"]
        instruction_pairs = [{"pos": pos_instr, "neg": neg_instr}]
    num_pairs = len(instruction_pairs)
    print(f"Instruction pairs: {num_pairs}")

    output_path = resolve_path(args.output_path)
    flat_output_path = resolve_path(
        args.flat_output_path or default_flat_output_path(output_path)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    flat_output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build flat list of (context, pair_idx) combos; seed_id encodes the pair index
    # so resume logic works correctly across partial runs.
    completed_ids = load_completed_ids(output_path) if args.resume else set()
    pending_combos = []
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
        for combo in pending_combos[: min(5, len(pending_combos))]:
            print("=" * 72)
            print(f"{combo['seed_id']}  (instr pair {combo['instruction_pair_idx']})")
            print(combo["context"])
        return

    if OpenAI is None:
        raise SystemExit("Missing dependency: pip install openai")

    client = OpenAI()

    with output_path.open("a", encoding="utf-8") as paired_handle, flat_output_path.open(
        "a", encoding="utf-8"
    ) as flat_handle:
        for combo in pending_combos:
            pair = generate_pair(
                client=client,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                context=combo["context"],
                axis_spec=axis_spec,
                shared_rules=shared_rules,
                instr_pair=combo["instr_pair"],
                max_retries=args.max_retries,
                retry_sleep_seconds=args.retry_sleep_seconds,
            )
            negative_style = axis_spec["negative_style"]
            positive_style = axis_spec["positive_style"]
            negative_field = axis_spec["negative_field"]
            positive_field = axis_spec["positive_field"]
            paired_row = {
                "seed_id": combo["seed_id"],
                "seed_record_index": combo["seed_record_index"],
                "instruction_pair_idx": combo["instruction_pair_idx"],
                "axis_name": axis_spec["name"],
                "negative_style": negative_style,
                "positive_style": positive_style,
                "negative_field": negative_field,
                "positive_field": positive_field,
                "persona_name": combo.get("persona_name"),
                "target_label": combo.get("target_label"),
                "source_dataset": str(args.data_path),
                "source": "synthetic_openai",
                "generation_model": args.model,
                "reasoning_effort": args.reasoning_effort,
                "context": combo["context"],
                negative_field: pair[negative_field],
                positive_field: pair[positive_field],
            }
            paired_handle.write(json.dumps(paired_row, ensure_ascii=False) + "\n")

            flat_rows = [
                {
                    "seed_id": combo["seed_id"],
                    "seed_record_index": combo["seed_record_index"],
                    "instruction_pair_idx": combo["instruction_pair_idx"],
                    "persona_name": combo.get("persona_name"),
                    "target_label": combo.get("target_label"),
                    "source_dataset": str(args.data_path),
                    "source": "synthetic_openai",
                    "generation_model": args.model,
                    "reasoning_effort": args.reasoning_effort,
                    "axis_name": axis_spec["name"],
                    "style": negative_style,
                    "context": combo["context"],
                    "response": pair[negative_field],
                },
                {
                    "seed_id": combo["seed_id"],
                    "seed_record_index": combo["seed_record_index"],
                    "instruction_pair_idx": combo["instruction_pair_idx"],
                    "persona_name": combo.get("persona_name"),
                    "target_label": combo.get("target_label"),
                    "source_dataset": str(args.data_path),
                    "source": "synthetic_openai",
                    "generation_model": args.model,
                    "reasoning_effort": args.reasoning_effort,
                    "axis_name": axis_spec["name"],
                    "style": positive_style,
                    "context": combo["context"],
                    "response": pair[positive_field],
                },
            ]
            for row in flat_rows:
                flat_handle.write(json.dumps(row, ensure_ascii=False) + "\n")

            paired_handle.flush()
            flat_handle.flush()
            print(f"Wrote {combo['seed_id']}")

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    print(f"Paired output: {output_path}")
    print(f"Flat output:   {flat_output_path}")


def resolve_axis_spec(axis_config: str | None, axis_name: str | None) -> tuple[dict, list[str]]:
    if not axis_config:
        return dict(DEFAULT_AXIS_SPEC), list(DEFAULT_SHARED_RULES)
    config_path = resolve_path(axis_config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    axes = config.get("axes")
    if not isinstance(axes, list) or not axes:
        raise SystemExit("Axis config must contain a non-empty 'axes' list.")
    if axis_name is None:
        if len(axes) == 1:
            chosen = axes[0]
        else:
            names = [axis.get("name") for axis in axes]
            raise SystemExit(
                f"Axis config contains multiple axes; pass --axis-name. Available: {names}"
            )
    else:
        chosen = next((axis for axis in axes if axis.get("name") == axis_name), None)
        if chosen is None:
            names = [axis.get("name") for axis in axes]
            raise SystemExit(f"Unknown axis_name={axis_name}. Available: {names}")
    required = [
        "name",
        "negative_style",
        "positive_style",
        "negative_field",
        "positive_field",
        "negative_description",
        "positive_description",
    ]
    missing = [key for key in required if not chosen.get(key)]
    if missing:
        raise SystemExit(f"Axis spec is missing required keys: {missing}")
    shared_rules = config.get("shared_rules") or DEFAULT_SHARED_RULES
    return dict(chosen), list(shared_rules)


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def default_flat_output_path(output_path: Path) -> str:
    return f"{output_path.with_suffix('')}_flat.jsonl"


def load_completed_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    completed: set[str] = set()
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            seed_id = row.get("seed_id")
            if isinstance(seed_id, str):
                completed.add(seed_id)
    return completed


def extract_seed_items(
    records: list[dict],
    context_field: str | None,
    strip_history_labels: bool,
    require_last_speaker: str | None,
    min_turns: int,
    max_context_chars: int,
) -> list[dict]:
    items: list[dict] = []
    seen_contexts: set[str] = set()
    for index, item in enumerate(records):
        context = ""
        if context_field:
            value = item.get(context_field)
            if isinstance(value, str):
                context = value.strip()
        else:
            context = extract_history(item).strip()
            if not context:
                prompt = item.get("prompt")
                if isinstance(prompt, str):
                    context = prompt.strip()
        if not context:
            continue
        if strip_history_labels:
            context = strip_annotation_labels(context)
        if not context or context in seen_contexts:
            continue
        if not is_usable_context(
            context=context,
            require_last_speaker=require_last_speaker,
            min_turns=min_turns,
            max_context_chars=max_context_chars,
        ):
            continue
        seen_contexts.add(context)
        seed_record_index = item.get("original_index", index)
        items.append(
            {
                "seed_id": f"seed_{seed_record_index}_{len(items)}",
                "seed_record_index": seed_record_index,
                "persona_name": item.get("persona_name"),
                "target_label": item.get("target_label"),
                "context": context,
            }
        )
    return items


def strip_annotation_labels(history: str) -> str:
    cleaned_lines: list[str] = []
    for raw_line in history.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        labeled = TURN_WITH_LABEL_RE.match(line)
        if labeled:
            speaker, utterance = labeled.groups()
            cleaned_lines.append(f"{normalize_speaker(speaker)}: {utterance.strip()}")
            continue

        plain = TURN_WITHOUT_LABEL_RE.match(line)
        if plain:
            speaker, utterance = plain.groups()
            cleaned_lines.append(f"{normalize_speaker(speaker)}: {utterance.strip()}")
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def normalize_speaker(raw: str) -> str:
    return SPEAKER_MAP.get(raw.strip(), raw.strip().title())


def extract_line_speaker(line: str) -> str | None:
    labeled = TURN_WITHOUT_LABEL_RE.match(line.strip())
    if not labeled:
        return None
    speaker, _ = labeled.groups()
    return normalize_speaker(speaker)


def is_usable_context(
    context: str,
    require_last_speaker: str | None,
    min_turns: int,
    max_context_chars: int,
) -> bool:
    lines = [line.strip() for line in context.splitlines() if line.strip()]
    if len(lines) < min_turns:
        return False
    if max_context_chars and len(context) > max_context_chars:
        return False
    if require_last_speaker:
        last_speaker = extract_line_speaker(lines[-1])
        if last_speaker != require_last_speaker:
            return False
    return True


def generate_pair(
    client: OpenAI,
    model: str,
    reasoning_effort: str,
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
            axis_spec, shared_rules, instr_pair["pos"], instr_pair["neg"]
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
            negative_reply = clean_reply(payload.get(negative_field, ""))
            positive_reply = clean_reply(payload.get(positive_field, ""))
            if not negative_reply or not positive_reply:
                raise ValueError(f"Missing reply in model output: {response.output_text}")
            return {
                negative_field: negative_reply,
                positive_field: positive_reply,
            }
        except (JSONDecodeError, ValueError) as exc:
            last_error = exc
            if attempt == max_retries:
                break
            time.sleep(retry_sleep_seconds)
        except Exception as exc:  # pragma: no cover - network/API failure path
            last_error = exc
            if attempt == max_retries:
                break
            time.sleep(retry_sleep_seconds)

    raise RuntimeError(f"Failed to generate a valid pair after {max_retries} attempts.") from last_error


def build_system_prompt(
    axis_spec: dict,
    shared_rules: list[str],
    pos_instruction: str,
    neg_instruction: str,
) -> str:
    negative_field = axis_spec["negative_field"]
    positive_field = axis_spec["positive_field"]
    rules = list(shared_rules)
    if axis_spec.get("negative_extra_rule"):
        rules.append(axis_spec["negative_extra_rule"])
    if axis_spec.get("positive_extra_rule"):
        rules.append(axis_spec["positive_extra_rule"])
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

{research_block}Instruktionen für diese Generierung:
- {negative_field}: {neg_instruction}
- {positive_field}: {pos_instruction}

Wichtige Regeln:
{rule_block}

Ausgabeformat:
- Gib nur gültiges JSON mit den Schlüsseln {negative_field} und {positive_field} zurück.
"""


def build_user_prompt(context: str) -> str:
    return f"""Gesprächsverlauf:
{context}

Aufgabe:
1. Erzeuge die nächste Klientenäußerung für den negativen Pol der Achse.
2. Erzeuge die nächste Klientenäußerung für den positiven Pol der Achse.
3. Beide Antworten sollen auf denselben Gesprächsverlauf folgen und sich primär in dieser Persona-Achse unterscheiden.
"""


def parse_json_object(text: str) -> dict:
    raw = text.strip()
    if raw.startswith("{") and raw.endswith("}"):
        return json.loads(raw)

    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        return json.loads(raw[start : end + 1])

    raise ValueError(f"Could not parse JSON object from output: {text}")


def clean_reply(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^\s*(Klient|Client)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.split(r"\n{2,}", cleaned, maxsplit=1)[0].strip()
    cleaned = re.split(r"\n\s*(Berater|Klient|Therapeut)\s*:", cleaned, maxsplit=1)[0].strip()
    return cleaned


if __name__ == "__main__":
    main()
