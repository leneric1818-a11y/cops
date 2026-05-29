#!/usr/bin/env python3
"""Build COPS context dataset from ncp_eval.jsonl.

Reads the rolling-window counseling dataset, filters out organisational turns
(Anrede, Gesprächseröffnung, Moderation), parses the personality_condition field
to extract structured case metadata (Hauptanliegen, Nebenanliegen, Steckbrief),
and writes an enriched JSONL suitable for COPS contrast-pair generation and
content-probe training.

Output fields (compatible with generate_steering_contrast_dataset.py):
  seed_id            str   unique context identifier
  original_index     int   source record index
  persona_name       str   client name
  target_label       str   ViKl-F2F category of the next client turn
  context            str   cleaned dialogue history ending in Berater turn
  hauptanliegen      str   case narrative (invariance set for COPS)
  nebenanliegen      list  secondary concerns
  steckbrief         dict  demographic facts
  personality_condition str raw personality_condition field (kept for reference)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Org-block target_label prefixes to exclude as TARGETS
# (turns where only administrative content is being generated)
ORG_TARGET_PREFIXES = {"K-FA-", "B-FA-", "B-Mod-", "B-A-*-", "K-A-*-"}

# Minimum number of substantive (non-org) client turns required in context
MIN_CLIENT_CONTENT_TURNS = 1

# Max context length in characters (0 = disabled)
MAX_CONTEXT_CHARS = 0

# Matches both plain "Speaker: text" and annotated "Speaker (label | category): text"
# Uses a lazy match for the annotation group to handle nested parentheses
TURN_RE = re.compile(r"^\s*([A-Za-zÄÖÜäöüß]+)\s*(?:\(.*?\))?\s*:\s*(.+?)\s*$", re.DOTALL)
SPEAKER_MAP = {"K": "Klient", "B": "Berater"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        default=str(ROOT / "data/processed/ncp_eval.jsonl"),
        help="Input JSONL (default: data/processed/ncp_eval.jsonl)",
    )
    parser.add_argument(
        "--output-path",
        default=str(ROOT / "data/processed/cops_contexts.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--target-speaker",
        default="Klient",
        choices=("Klient", "both"),
        help="Only keep contexts whose TARGET is a Klient turn (default) or both.",
    )
    parser.add_argument(
        "--skip-org-targets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip organisational/admin target turns (K-FA-*, B-FA-*, B-Mod-*, ...).",
    )
    parser.add_argument(
        "--min-client-content-turns",
        type=int,
        default=MIN_CLIENT_CONTENT_TURNS,
        help="Minimum substantive client turns in context before accepting it.",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=MAX_CONTEXT_CHARS,
        help="Max context length in chars (0 = disabled).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats and first 5 examples without writing output.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# personality_condition parser
# ---------------------------------------------------------------------------

def parse_personality_condition(text: str) -> dict:
    """Extract structured fields from the personality_condition free-text block."""
    result: dict = {
        "hauptanliegen": "",
        "nebenanliegen": [],
        "steckbrief": {},
        "sprachliche_merkmale": "",
        "emotionale_merkmale": "",
    }
    if not text:
        return result

    # Split on known section headers
    sections = re.split(
        r"\n(?=Name:|Steckbrief:|Hauptanliegen:|Nebenanliegen:|Sprachliche Merkmale:|Emotionale Merkmale:|Prinzipien:)",
        text.strip(),
    )
    for section in sections:
        section = section.strip()
        if section.startswith("Hauptanliegen:"):
            result["hauptanliegen"] = section[len("Hauptanliegen:"):].strip()
        elif section.startswith("Nebenanliegen:"):
            raw = section[len("Nebenanliegen:"):].strip()
            result["nebenanliegen"] = [
                item.strip().lstrip(";").strip()
                for item in re.split(r"[;\n]", raw)
                if item.strip()
            ]
        elif section.startswith("Steckbrief:"):
            raw = section[len("Steckbrief:"):].strip()
            steckbrief: dict = {}
            for part in re.split(r";", raw):
                part = part.strip()
                if ":" in part:
                    k, _, v = part.partition(":")
                    steckbrief[k.strip()] = v.strip()
            result["steckbrief"] = steckbrief
        elif section.startswith("Sprachliche Merkmale:"):
            result["sprachliche_merkmale"] = section[len("Sprachliche Merkmale:"):].strip()
        elif section.startswith("Emotionale Merkmale:"):
            result["emotionale_merkmale"] = section[len("Emotionale Merkmale:"):].strip()

    return result


# ---------------------------------------------------------------------------
# Context cleaning
# ---------------------------------------------------------------------------

def clean_history(raw: str) -> str:
    """Strip annotation labels from conversation history lines."""
    cleaned: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        m = TURN_RE.match(line)
        if m:
            speaker, utterance = m.groups()
            speaker = SPEAKER_MAP.get(speaker.strip(), speaker.strip().title())
            cleaned.append(f"{speaker}: {utterance.strip()}")
        else:
            cleaned.append(line)
    return "\n".join(cleaned)


def last_speaker(context: str) -> str | None:
    for line in reversed(context.splitlines()):
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^([A-Za-zÄÖÜäöüß]+)\s*:", line)
        if m:
            return m.group(1)
    return None


def count_substantive_client_turns(raw_history: str) -> int:
    """Count client turns that are NOT org-block (K-FA-* label)."""
    count = 0
    for line in raw_history.splitlines():
        line = line.strip()
        if not line:
            continue
        # Client turn with non-org label
        m = re.match(r"^K\s*\(([^)]*)\)\s*:", line)
        if m:
            label = m.group(1).split("|")[0].strip()
            if not any(label.startswith(p.rstrip("*-")) for p in ["K-FA-", "K-A-*"]):
                count += 1
    return count


def is_org_target(target_label: str) -> bool:
    return any(target_label.startswith(p) for p in ORG_TARGET_PREFIXES)


def target_speaker_from_label(target_label: str) -> str:
    if target_label.startswith("K"):
        return "Klient"
    if target_label.startswith("B"):
        return "Berater"
    return "unknown"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    output_path = Path(args.output_path)

    records = []
    with data_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    print(f"Loaded {len(records)} records from {data_path}")

    accepted: list[dict] = []
    stats = {"total": len(records), "org_target": 0, "wrong_speaker": 0,
             "too_short": 0, "too_long": 0, "accepted": 0}

    seen_contexts: set[str] = set()

    for rec in records:
        target_label = rec.get("target_label", "")
        raw_history = rec.get("conversation_history", "")
        persona_name = rec.get("persona_name", "")
        original_index = rec.get("original_index", -1)
        personality_condition = rec.get("personality_condition", "")

        # Filter org targets
        if args.skip_org_targets and is_org_target(target_label):
            stats["org_target"] += 1
            continue

        # Filter by target speaker
        tspeaker = target_speaker_from_label(target_label)
        if args.target_speaker == "Klient" and tspeaker != "Klient":
            stats["wrong_speaker"] += 1
            continue

        # Context must end in Berater turn (model generates next client response)
        context = clean_history(raw_history)
        if last_speaker(context) != "Berater":
            stats["wrong_speaker"] += 1
            continue

        # Minimum substantive client content in context
        if count_substantive_client_turns(raw_history) < args.min_client_content_turns:
            stats["too_short"] += 1
            continue

        # Max length
        if args.max_context_chars and len(context) > args.max_context_chars:
            stats["too_long"] += 1
            continue

        # Deduplicate
        if context in seen_contexts:
            continue
        seen_contexts.add(context)

        # Parse case metadata
        case = parse_personality_condition(personality_condition)

        row = {
            "seed_id": f"cops_{original_index}_{len(accepted)}",
            "original_index": original_index,
            "persona_name": persona_name,
            "target_label": target_label,
            "context": context,                  # clean — for prompting
            "context_labeled": raw_history,      # with annotation labels — for analysis
            "hauptanliegen": case["hauptanliegen"],
            "nebenanliegen": case["nebenanliegen"],
            "steckbrief": case["steckbrief"],
            "personality_condition": personality_condition,
        }
        accepted.append(row)
        stats["accepted"] += 1

    print(f"\nFiltering stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    if args.dry_run:
        print("\n--- First 5 accepted contexts ---")
        for row in accepted[:5]:
            print(f"\nseed_id: {row['seed_id']}")
            print(f"target_label: {row['target_label']}")
            print(f"persona_name: {row['persona_name']}")
            print(f"hauptanliegen: {row['hauptanliegen'][:150]}...")
            print(f"context:\n{row['context']}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in accepted:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(accepted)} contexts to {output_path}")


if __name__ == "__main__":
    main()
