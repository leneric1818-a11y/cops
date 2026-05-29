# Data

## `processed_anonymized/cops_contexts.jsonl`

400 counseling roleplay contexts derived from a German social-work
course corpus. Each line is one JSON record with fields:

| Field | Description |
|---|---|
| `seed_id` | Stable ID, `cops_<original_index>` |
| `original_index` | Position in the source corpus |
| `persona_name` | Anonymized client name (e.g. *Frau Schuster*) |
| `target_label` | OnCoCo-compatible utterance-category label |
| `context` | Raw dialogue prefix (Klient/Berater turns) |
| `context_labeled` | Same context with per-turn OnCoCo labels |

**Splits:**
- Indices `0–199`: *held-in* — used to fit per-persona and global steering
  vectors.
- Indices `200–399`: *held-out* — seed-fixed evaluation split. All numbers
  reported in the paper use this split.

**PII handling:** The corpus has been processed with
`openai/privacy-filter` (revision `7ffa9a04`, 1.5B-token ONNX classifier)
prior to release. 581 substitutions were applied across the source corpus.
Personal names, locations, organisations, contact details, and other
identifiers are replaced with consistent surrogates.

**Consent & license:** Original participants (German social-work students)
consented to research use of the roleplay transcripts. The released data
are licensed CC BY-NC 4.0 (see top-level `LICENSE-DATA`).

## Files NOT included in this release

- `data/raw/` — un-anonymized transcripts; held by the original authors.
- `data/processed/` — pre-anonymization intermediate; superseded by
  `processed_anonymized/`.
- Internal preference databases and per-scenario raw exports.
