"""Label description helpers."""

from __future__ import annotations

import csv
import os
from typing import Dict

from llm_ft_comparison.utils.label_utils import extract_label_code


FILE_ENCODINGS = ["utf-8", "latin1", "cp1252", "iso-8859-1"]


def load_label_descriptions(descriptions_file: str) -> Dict[str, str]:
    """Load label descriptions from a CSV file."""
    if not descriptions_file or not os.path.exists(descriptions_file):
        print(f"Warning: label descriptions file not found at {descriptions_file}")
        return {}

    label_descriptions: Dict[str, str] = {}

    for encoding in FILE_ENCODINGS:
        try:
            with open(descriptions_file, "r", encoding=encoding, newline="") as handle:
                reader = csv.reader(handle)
                try:
                    header = next(reader)
                except StopIteration:
                    return label_descriptions

                has_header = any(
                    "kategorie" in cell.lower() or "category" in cell.lower()
                    for cell in header
                )
                if not has_header:
                    _process_description_row(header, label_descriptions)

                for row in reader:
                    _process_description_row(row, label_descriptions)

            print(
                f"OK: loaded {len(label_descriptions)} label descriptions using {encoding}"
            )
            break
        except UnicodeDecodeError:
            continue
        except Exception as exc:
            print(f"Warning: error loading label descriptions: {exc}")
            break

    return label_descriptions


def get_label_description_for_label(
    label: str | None,
    label_descriptions: Dict[str, str],
    warn_missing: bool = False,
) -> str:
    """Return a description for a label, falling back to the label text."""
    if not isinstance(label, str) or not label:
        return "Self-explanatory counseling category"

    desc = label_descriptions.get(label)
    if desc:
        return desc

    code_key = extract_label_code(label)
    desc = label_descriptions.get(code_key)
    if desc:
        return desc

    if "|" in label:
        try:
            _, desc_part = label.split("|", 1)
            desc = desc_part.strip() or label
        except Exception:
            desc = label
    else:
        desc = label

    if warn_missing:
        print(f"Warning: no description found for label {label}. Using: {desc}")
    return desc


def _process_description_row(row: list[str], label_descriptions: Dict[str, str]) -> None:
    if not row or not row[0].strip():
        return
    full_label = row[0].strip()
    additional_desc = row[1].strip() if len(row) > 1 and row[1].strip() else ""
    code_key = extract_label_code(full_label)

    if additional_desc:
        label_descriptions[full_label] = additional_desc
        if code_key and code_key not in label_descriptions:
            label_descriptions[code_key] = additional_desc
    else:
        if "|" in full_label:
            try:
                _, desc_part = full_label.split("|", 1)
            except Exception:
                desc_part = ""
            desc_part = desc_part.strip()
            if desc_part:
                label_descriptions[full_label] = desc_part
                if code_key and code_key not in label_descriptions:
                    label_descriptions[code_key] = desc_part
            else:
                _set_self_explanatory(label_descriptions, full_label, code_key)
        else:
            _set_self_explanatory(label_descriptions, full_label, code_key)


def _set_self_explanatory(
    label_descriptions: Dict[str, str], full_label: str, code_key: str | None
) -> None:
    label_descriptions[full_label] = "Self-explanatory counseling category"
    if code_key and code_key not in label_descriptions:
        label_descriptions[code_key] = "Self-explanatory counseling category"
