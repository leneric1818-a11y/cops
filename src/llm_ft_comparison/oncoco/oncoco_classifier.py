"""OnCoCo classifier helpers for label-conditioned evaluation."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from llm_ft_comparison.oncoco.config import (
    DEFAULT_ONCOCO_MAPPING_PATH,
    DEFAULT_ONCOCO_MODEL_PATH,
    FILE_ENCODINGS,
)

try:
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        XLMRobertaTokenizerFast,
    )

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


def load_oncoco_classifier(
    model_path: str | None = None,
    strict: bool = True,
) -> Tuple[Any, Any]:
    """Load the OnCoCo classifier model and tokenizer."""
    if model_path is None:
        model_path = DEFAULT_ONCOCO_MODEL_PATH

    if not TRANSFORMERS_AVAILABLE:
        error_msg = (
            "transformers library not available for OnCoCo classifier. "
            "Install with: pip install transformers torch"
        )
        if strict:
            raise RuntimeError(error_msg)
        print(f"Error: {error_msg}")
        return None, None

    if not os.path.exists(model_path):
        error_msg = f"OnCoCo model path does not exist: {model_path}"
        if strict:
            raise FileNotFoundError(error_msg)
        print(f"Error: {error_msg}")
        return None, None

    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            device_name = "CUDA GPU"
        else:
            device = torch.device("cpu")
            device_name = "CPU"

        print(f"Loading OnCoCo classifier from {model_path}")
        print(f"Using device: {device_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, trust_remote_code=True
        )
        model = model.to(device)
        model.eval()

        print(f"OK: OnCoCo classifier loaded on {device_name}")
        return tokenizer, model
    except Exception as exc:
        print(f"Warning: error loading OnCoCo classifier: {exc}")
        try:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                device_name = "CUDA GPU"
            else:
                device = torch.device("cpu")
                device_name = "CPU"

            tok_file = os.path.join(model_path, "tokenizer.json")
            if os.path.exists(tok_file):
                print("Retrying tokenizer.json fallback with XLMRobertaTokenizerFast.")
                tokenizer = XLMRobertaTokenizerFast(tokenizer_file=tok_file)
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_path, trust_remote_code=True
                )
                model = model.to(device)
                model.eval()
                print(f"OK: OnCoCo classifier loaded (fallback) on {device_name}")
                return tokenizer, model
        except Exception as exc2:
            print(f"Warning: fallback failed: {exc2}")

        error_msg = (
            f"Failed to load OnCoCo classifier from {model_path}. "
            f"Primary error: {exc}. "
            "Check that the model path is correct and contains valid model files."
        )
        if strict:
            raise RuntimeError(error_msg)
        print(f"Error: {error_msg}")
        return None, None


def load_oncoco_code_mapping(mapping_file: str | None = None) -> Dict[str, str]:
    """Load mapping from English OnCoCo codes to German codes + K5 names."""
    if mapping_file is None:
        mapping_file = DEFAULT_ONCOCO_MAPPING_PATH

    import csv

    code_mapping: Dict[str, str] = {}

    if not os.path.exists(mapping_file):
        print(f"Warning: OnCoCo mapping file not found at {mapping_file}")
        return code_mapping

    try:
        for encoding in FILE_ENCODINGS:
            try:
                with open(mapping_file, "r", encoding=encoding, newline="") as handle:
                    reader = csv.reader(handle)
                    header = next(reader)
                    header_lut = {h.strip(): idx for idx, h in enumerate(header)}

                    def idx_for(name_candidates, default=None):
                        for name in name_candidates:
                            if name in header_lut:
                                return header_lut[name]
                        return default

                    idx_de = idx_for(["Code DE", "code de", "DE", "Code_DE"], 0)
                    idx_en = idx_for(["Code EN", "code en", "EN", "Code_EN"], 1)
                    idx_k5 = idx_for(["K5 Name", "K5", "K5_Name"], 19)

                    for row in reader:
                        if idx_de is None or idx_en is None or idx_k5 is None:
                            continue
                        if max(idx_de, idx_en, idx_k5) >= len(row):
                            continue
                        code_de = (row[idx_de] or "").strip()
                        code_en = (row[idx_en] or "").strip()
                        k5_name = (row[idx_k5] or "").strip()

                        if code_de and code_en and k5_name:
                            code_mapping[code_en] = f"{code_de} | {k5_name}"

                print(
                    f"OK: Loaded {len(code_mapping)} OnCoCo code mappings using {encoding}"
                )
                break

            except UnicodeDecodeError:
                if encoding == FILE_ENCODINGS[-1]:
                    print("Warning: OnCoCo mapping could not be decoded.")
                continue

    except Exception as exc:
        print(f"Warning: error loading OnCoCo mapping: {exc}")

    return code_mapping


def _build_allowed_indices(
    model: Any, code_mapping: Optional[Dict[str, str]], speaker: str
) -> set:
    """Build set of allowed class indices for given speaker (client/counselor)."""
    desired_prefix = "K-" if speaker == "client" else "B-"
    allowed: set[int] = set()
    id2label = getattr(model.config, "id2label", {}) or {}
    if not isinstance(id2label, dict):
        return allowed

    for key, eng_label in id2label.items():
        try:
            idx = key if isinstance(key, int) else int(key)
        except Exception:
            continue
        de_label = code_mapping.get(eng_label) if code_mapping else None
        if isinstance(de_label, str) and de_label.strip().startswith(desired_prefix):
            allowed.add(idx)
        elif isinstance(eng_label, str) and eng_label.strip().startswith(desired_prefix):
            allowed.add(idx)

    return allowed


def classify_message_with_oncoco(
    message: str,
    speaker: str,
    tokenizer: Any,
    model: Any,
    code_mapping: Optional[Dict[str, str]] = None,
    strict: bool = True,
) -> str:
    """Classify a single message using the OnCoCo classifier."""
    if not tokenizer or not model:
        error_msg = (
            "OnCoCo classifier (tokenizer or model) not loaded. "
            "Ensure load_oncoco_classifier() succeeded before calling this function."
        )
        if strict:
            raise RuntimeError(error_msg)
        print(f"Warning: {error_msg}")
        return (
            "K-WF-AKP-*-PDar-* | Problemdarstellung"
            if speaker == "client"
            else "B-WF-AK-RS-ERx-* | Einfache Reflexion"
        )

    try:
        role_prefix = "Klient" if speaker == "client" else "Berater"
        input_msg = f"{role_prefix}: {message}"

        inputs = tokenizer(
            input_msg,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.squeeze(0)

            allowed = _build_allowed_indices(model, code_mapping, speaker)
            if allowed:
                mask = torch.ones_like(logits, dtype=torch.bool)
                for idx in allowed:
                    if 0 <= idx < logits.numel():
                        mask[idx] = False
                logits = logits.masked_fill(mask, float("-inf"))

            predicted_class_id = torch.argmax(logits).item()

        id2label = getattr(model.config, "id2label", {}) or {}
        english_label = None
        if isinstance(id2label, dict):
            english_label = id2label.get(predicted_class_id) or id2label.get(
                str(predicted_class_id)
            )

        de_label = code_mapping.get(english_label) if english_label and code_mapping else None
        if not de_label and isinstance(english_label, str) and (
            english_label.startswith("B-") or english_label.startswith("K-")
        ):
            de_label = english_label
        if not de_label:
            de_label = (
                "K-WF-AKP-*-PDar-* | Problemdarstellung"
                if speaker == "client"
                else "B-WF-AK-RS-ERx-* | Einfache Reflexion"
            )

        desired_prefix = "K" if speaker == "client" else "B"
        if not de_label.strip().startswith(desired_prefix):
            de_label = (
                "K-WF-AKP-*-PDar-* | Problemdarstellung"
                if speaker == "client"
                else "B-WF-AK-RS-ERx-* | Einfache Reflexion"
            )

        return de_label

    except Exception as exc:
        print(f"Warning: error classifying message with OnCoCo: {exc}")
        return (
            "K-WF-AKP-*-PDar-* | Problemdarstellung"
            if speaker == "client"
            else "B-WF-AK-RS-ERx-* | Einfache Reflexion"
        )


def classify_messages_batch_oncoco(
    messages_data: List[Tuple[str, str]],
    tokenizer: Any,
    model: Any,
    code_mapping: Optional[Dict[str, str]] = None,
    batch_size: int = 32,
    strict: bool = True,
) -> List[str]:
    """Classify multiple messages in batches with speaker-role constraints."""
    if not tokenizer or not model:
        error_msg = (
            "OnCoCo classifier (tokenizer or model) not loaded. "
            "Ensure load_oncoco_classifier() succeeded before calling this function."
        )
        if strict:
            raise RuntimeError(error_msg)
        print(f"Warning: {error_msg}")
        return [
            (
                "K-WF-AKP-*-PDar-* | Problemdarstellung"
                if speaker == "client"
                else "B-WF-AK-RS-ERx-* | Einfache Reflexion"
            )
            for _, speaker in messages_data
        ]

    def map_idx_to_de(idx: int, speaker: str) -> str:
        id2label = getattr(model.config, "id2label", {}) or {}
        eng = None
        if isinstance(id2label, dict):
            eng = id2label.get(idx) or id2label.get(str(idx))
        de = code_mapping.get(eng) if code_mapping and eng else None
        if not de and isinstance(eng, str) and (eng.startswith("B-") or eng.startswith("K-")):
            de = eng
        if not de:
            de = (
                "K-WF-AKP-*-PDar-* | Problemdarstellung"
                if speaker == "client"
                else "B-WF-AK-RS-ERx-* | Einfache Reflexion"
            )
        desired_prefix = "K" if speaker == "client" else "B"
        if not de.strip().startswith(desired_prefix):
            de = (
                "K-WF-AKP-*-PDar-* | Problemdarstellung"
                if speaker == "client"
                else "B-WF-AK-RS-ERx-* | Einfache Reflexion"
            )
        return de

    results: List[str] = []
    try:
        for i in range(0, len(messages_data), batch_size):
            batch = messages_data[i : i + batch_size]
            texts = []
            speakers = []
            for message, speaker in batch:
                role_prefix = "Klient" if speaker == "client" else "Berater"
                texts.append(f"{role_prefix}: {message}")
                speakers.append(speaker)

            inputs = tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            )

            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits

            for row_logits, speaker in zip(logits, speakers):
                row_logits = row_logits.clone()
                allowed = _build_allowed_indices(model, code_mapping, speaker)
                if allowed:
                    mask = torch.ones_like(row_logits, dtype=torch.bool)
                    for idx in allowed:
                        if 0 <= idx < row_logits.numel():
                            mask[idx] = False
                    row_logits = row_logits.masked_fill(mask, float("-inf"))
                pred_id = int(torch.argmax(row_logits).item())
                results.append(map_idx_to_de(pred_id, speaker))

    except Exception as exc:
        print(f"Warning: error in batch classification with OnCoCo: {exc}")
        results = [
            (
                "K-WF-AKP-*-PDar-* | Problemdarstellung"
                if speaker == "client"
                else "B-WF-AK-RS-ERx-* | Einfache Reflexion"
            )
            for _, speaker in messages_data
        ]

    return results


def classify_messages_batch_with_scores(
    messages_data: List[Tuple[str, str]],
    tokenizer: Any,
    model: Any,
    code_mapping: Optional[Dict[str, str]] = None,
    topk: int = 3,
    batch_size: int = 32,
    strict: bool = False,
) -> List[Dict[str, Any]]:
    """Classify multiple messages in batches and return detailed scores."""
    if not tokenizer or not model:
        if strict:
            raise RuntimeError("OnCoCo classifier not loaded")
        return [
            {
                "label": (
                    "K-WF-AKP-*-PDar-* | Problemdarstellung"
                    if speaker == "client"
                    else "B-WF-AK-RS-ERx-* | Einfache Reflexion"
                ),
                "confidence": 1.0,
                "topk": [
                    (
                        "K-WF-AKP-*-PDar-* | Problemdarstellung"
                        if speaker == "client"
                        else "B-WF-AK-RS-ERx-* | Einfache Reflexion",
                        1.0,
                    )
                ],
                "entropy": 0.0,
            }
            for _, speaker in messages_data
        ]

    def map_idx_to_de(idx: int, speaker: str) -> str:
        id2label = getattr(model.config, "id2label", {}) or {}
        eng = None
        if isinstance(id2label, dict):
            eng = id2label.get(idx) or id2label.get(str(idx))
        de = code_mapping.get(eng) if code_mapping and eng else None
        if not de and isinstance(eng, str) and (eng.startswith("B-") or eng.startswith("K-")):
            de = eng
        if not de:
            de = (
                "K-WF-AKP-*-PDar-* | Problemdarstellung"
                if speaker == "client"
                else "B-WF-AK-RS-ERx-* | Einfache Reflexion"
            )
        desired_prefix = "K" if speaker == "client" else "B"
        if not de.strip().startswith(desired_prefix):
            de = (
                "K-WF-AKP-*-PDar-* | Problemdarstellung"
                if speaker == "client"
                else "B-WF-AK-RS-ERx-* | Einfache Reflexion"
            )
        return de

    results: List[Dict[str, Any]] = []

    try:
        for i in range(0, len(messages_data), batch_size):
            batch = messages_data[i : i + batch_size]
            texts = []
            speakers = []
            for message, speaker in batch:
                role_prefix = "Klient" if speaker == "client" else "Berater"
                texts.append(f"{role_prefix}: {message}")
                speakers.append(speaker)

            inputs = tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            )

            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits

            for row_logits, speaker in zip(logits, speakers):
                row_logits = row_logits.clone()
                allowed = _build_allowed_indices(model, code_mapping, speaker)
                if allowed:
                    mask = torch.ones_like(row_logits, dtype=torch.bool)
                    for idx in allowed:
                        if 0 <= idx < row_logits.numel():
                            mask[idx] = False
                    row_logits = row_logits.masked_fill(mask, float("-inf"))

                probs = torch.softmax(row_logits, dim=-1)
                k = min(topk, probs.numel())
                top_probs, top_indices = torch.topk(probs, k)

                topk_list = [
                    (map_idx_to_de(int(idx), speaker), float(prob))
                    for prob, idx in zip(top_probs.tolist(), top_indices.tolist())
                ]
                top_label, top_conf = topk_list[0]

                probs_clamped = torch.clamp(probs, min=1e-12)
                entropy = float(-torch.sum(probs_clamped * torch.log(probs_clamped)).item())

                results.append(
                    {
                        "label": top_label,
                        "confidence": top_conf,
                        "topk": topk_list,
                        "entropy": entropy,
                    }
                )

    except Exception as exc:
        print(f"Warning: batch scoring failed: {exc}")
        for _, speaker in messages_data:
            fallback_label = (
                "K-WF-AKP-*-PDar-* | Problemdarstellung"
                if speaker == "client"
                else "B-WF-AK-RS-ERx-* | Einfache Reflexion"
            )
            results.append(
                {
                    "label": fallback_label,
                    "confidence": 1.0,
                    "topk": [(fallback_label, 1.0)],
                    "entropy": 0.0,
                }
            )

    return results


def classify_message_with_oncoco_scores(
    message: str,
    speaker: str,
    tokenizer: Any,
    model: Any,
    code_mapping: Optional[Dict[str, str]] = None,
    topk: int = 3,
    strict: bool = True,
) -> Dict[str, Any]:
    """Classify a single message and return scoring details."""
    if not tokenizer or not model:
        error_msg = (
            "OnCoCo classifier (tokenizer or model) not loaded. "
            "Ensure load_oncoco_classifier() succeeded before calling this function."
        )
        if strict:
            raise RuntimeError(error_msg)
        print(f"Warning: {error_msg}")
        base_label = classify_message_with_oncoco(
            message, speaker, tokenizer, model, code_mapping, strict=False
        )
        return {
            "label": base_label,
            "confidence": 1.0,
            "topk": [(base_label, 1.0)],
            "entropy": 0.0,
        }

    try:
        role_prefix = "Klient" if speaker == "client" else "Berater"
        input_msg = f"{role_prefix}: {message}"
        inputs = tokenizer(
            input_msg,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )

        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.squeeze(0)

            allowed = _build_allowed_indices(model, code_mapping, speaker)
            if allowed:
                mask = torch.ones_like(logits, dtype=torch.bool)
                for idx in allowed:
                    if 0 <= idx < logits.numel():
                        mask[idx] = False
                logits = logits.masked_fill(mask, float("-inf"))

            probs = torch.softmax(logits, dim=-1)

            k = min(int(topk) if isinstance(topk, int) and topk > 0 else 3, probs.numel())
            top_probs, top_indices = torch.topk(probs, k)

            id2label = getattr(model.config, "id2label", {}) if hasattr(model, "config") else {}

            def map_idx_to_de(idx: int) -> str:
                eng = None
                if isinstance(id2label, dict):
                    eng = id2label.get(idx) or id2label.get(str(idx))
                de = code_mapping.get(eng) if code_mapping and eng else None
                if not de and isinstance(eng, str) and (eng.startswith("B-") or eng.startswith("K-")):
                    de = eng
                if not de:
                    de = (
                        "K-WF-AKP-*-PDar-* | Problemdarstellung"
                        if speaker == "client"
                        else "B-WF-AK-RS-ERx-* | Einfache Reflexion"
                    )
                desired_prefix = "K" if speaker == "client" else "B"
                if not de.strip().startswith(desired_prefix):
                    de = (
                        "K-WF-AKP-*-PDar-* | Problemdarstellung"
                        if speaker == "client"
                        else "B-WF-AK-RS-ERx-* | Einfache Reflexion"
                    )
                return de

            topk_list = [
                (map_idx_to_de(int(idx)), float(prob))
                for prob, idx in zip(top_probs.tolist(), top_indices.tolist())
            ]
            top_label, top_conf = topk_list[0]

            probs_clamped = torch.clamp(probs, min=1e-12)
            entropy = float(-torch.sum(probs_clamped * torch.log(probs_clamped)).item())

            return {
                "label": top_label,
                "confidence": float(top_conf),
                "topk": topk_list,
                "entropy": entropy,
            }

    except Exception as exc:
        print(f"Warning: error scoring with OnCoCo: {exc}")
        base_label = classify_message_with_oncoco(
            message, speaker, tokenizer, model, code_mapping
        )
        return {
            "label": base_label,
            "confidence": 0.0,
            "topk": [(base_label, 0.0)],
            "entropy": 0.0,
        }
