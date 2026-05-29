#!/usr/bin/env python3
"""Classify steered and baseline responses with the OnCoCo sentence classifier.

Each response is first split into individual sentences; every sentence is then
classified independently.  The output has one row per sentence, with a
response_idx field to re-group sentences back to their source response.

Usage (on cluster):
    python scripts/classify_responses_oncoco.py \
        --model-path models/xlm-roberta-large-OnCoCo-DE-EN \
        --input-dirs outputs/steering_eval/paper_v2 \
                     outputs/steering_eval/paper_v2_negalpha \
        --output outputs/steering_eval/oncoco_labels.jsonl \
        --batch-size 256 \
        --skip-glob "*_all.jsonl"
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import time
from pathlib import Path

# Abbreviations that must not trigger a sentence split
_ABBREVS = re.compile(
    r"\b(?:Dr|Prof|Hr|Fr|bzw|z\.B|d\.h|usw|etc|ca|ggf|evtl|inkl|bzgl|Tel|Nr|Str)\."
)

def split_sentences(text: str) -> list[str]:
    """Split German text into sentences, robust to common abbreviations."""
    # Temporarily mask abbreviation periods
    masked = _ABBREVS.sub(lambda m: m.group().replace(".", "\x00"), text)
    parts = re.split(r"(?<=[.!?…])\s+(?=[A-ZÄÖÜ\"\'\(])", masked)
    sentences = [p.replace("\x00", ".").strip() for p in parts if p.strip()]
    return sentences or [text.strip()]

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default="models/xlm-roberta-large-OnCoCo-DE-EN",
                   help="Path to OnCoCo model directory (absolute or relative to repo root)")
    p.add_argument("--input-dirs", nargs="+",
                   default=["outputs/steering_eval/paper_v2",
                            "outputs/steering_eval/paper_v2_negalpha"])
    p.add_argument("--output", default="outputs/steering_eval/oncoco_labels.jsonl")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-length", type=int, default=256,
                   help="Max token length; counseling turns are short, 256 is safe")
    p.add_argument("--top-k", type=int, default=5,
                   help="Store top-k label probabilities per response (default 5)")
    p.add_argument("--skip-glob", default="*_all.jsonl",
                   help="Glob pattern for filenames to skip (avoids double-counting supersets)")
    # Role prefix tokens added during OnCoCo fine-tuning ([CL] for client, [CO] for counselor).
    # Always use [CL] here since we classify generated client responses.
    p.add_argument("--role-prefix", default="[CL]",
                   help="Special role token prepended to each utterance ([CL] or [CO])")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return p.parse_args()


def load_sentences(input_dirs: list[str], skip_glob: str) -> list[dict]:
    """Load responses and expand into one record per sentence."""
    sentences = []
    for d in input_dirs:
        dpath = ROOT / d if not Path(d).is_absolute() else Path(d)
        for f in sorted(dpath.glob("*.jsonl")):
            if fnmatch.fnmatch(f.name, skip_glob):
                print(f"  skip {f.name}")
                continue
            n_resp = 0
            with f.open(encoding="utf-8") as fh:
                for resp_idx, line in enumerate(fh):
                    row = json.loads(line)
                    src = str(f.relative_to(ROOT))
                    for sent_idx, sent in enumerate(split_sentences(row["response"])):
                        sentences.append({
                            "seed_id":      row["seed_id"],
                            "run_seed":     row.get("run_seed"),
                            "variant":      row["variant"],
                            "alpha":        row.get("alpha"),
                            "axis":         row.get("axis"),
                            "model_path":   row.get("model_path"),
                            "target_label": row.get("target_label"),
                            "resp_idx":     resp_idx,
                            "sent_idx":     sent_idx,
                            "n_sents":      None,   # filled below
                            "sentence":     sent,
                            "_source_file": src,
                        })
                    n_resp += 1
            print(f"  {n_resp:5d} responses → {len(sentences):7d} sentences total  ← {f.name}")
    # fill n_sents: count sentences per (source_file, resp_idx)
    counts: dict[tuple, int] = {}
    for s in sentences:
        key = (s["_source_file"], s["resp_idx"])
        counts[key] = counts.get(key, 0) + 1
    for s in sentences:
        s["n_sents"] = counts[(s["_source_file"], s["resp_idx"])]
    return sentences


def batched(lst: list, n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def main() -> None:
    args = parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_path = ROOT / args.model_path if not Path(args.model_path).is_absolute() else Path(args.model_path)
    print(f"Loading model from {model_path} …")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model.eval().to(device)
    print(f"Model on {device}. Labels: {len(model.config.id2label)}")
    id2label: dict[int, str] = model.config.id2label

    # Build role mask: suppress opposite-role logits before softmax (per paper)
    import torch as _torch
    role_prefix = args.role_prefix.strip("[] ").upper()   # "CL" or "CO"
    suppress_prefix = "CO-" if role_prefix == "CL" else "CL-"
    suppressed = _torch.tensor(
        [lbl.startswith(suppress_prefix) for lbl in id2label.values()],
        dtype=_torch.bool, device=device,
    )
    n_suppressed = suppressed.sum().item()
    print(f"Role prefix: '[{role_prefix}]'  —  suppressing {n_suppressed} {suppress_prefix}* logits (output masking)")

    print("\nLoading generation files and splitting sentences …")
    rows = load_sentences(args.input_dirs, args.skip_glob)
    print(f"\n{len(rows)} sentences to classify.")

    out_path = ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    n_done = 0
    with out_path.open("w", encoding="utf-8") as out_fh:
        for batch in batched(rows, args.batch_size):
            texts = [f"{args.role_prefix} {r['sentence']}" for r in batch]
            enc = tokenizer(
                texts,
                max_length=args.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            with torch.no_grad():
                logits = model(**enc).logits
            logits[:, suppressed] = float("-inf")   # output masking (paper §4)
            probs = torch.softmax(logits, dim=-1).cpu().float().numpy()

            for row, prob_vec in zip(batch, probs):
                top_idx = prob_vec.argsort()[::-1][: args.top_k]
                result = {
                    "seed_id":        row["seed_id"],
                    "run_seed":       row.get("run_seed"),
                    "variant":        row["variant"],
                    "alpha":          row.get("alpha"),
                    "axis":           row.get("axis"),
                    "model_path":     row.get("model_path"),
                    "target_label":   row.get("target_label"),
                    "resp_idx":       row.get("resp_idx"),
                    "sent_idx":       row.get("sent_idx"),
                    "n_sents":        row.get("n_sents"),
                    "sentence":       row["sentence"],
                    "predicted_label":  id2label[int(top_idx[0])],
                    "predicted_prob":   float(prob_vec[top_idx[0]]),
                    "top_k_labels": [
                        {"label": id2label[int(i)], "prob": float(prob_vec[i])}
                        for i in top_idx
                    ],
                    "_source_file":   row.get("_source_file"),
                }
                out_fh.write(json.dumps(result, ensure_ascii=False) + "\n")

            n_done += len(batch)
            elapsed = time.time() - t0
            rate = n_done / elapsed
            remaining = (len(rows) - n_done) / rate if rate > 0 else 0
            print(
                f"  {n_done:6d}/{len(rows)}  "
                f"{rate:6.0f} resp/s  "
                f"ETA {remaining:5.0f}s",
                end="\r",
            )

    print(f"\n\nDone. {n_done} sentences written to {out_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
