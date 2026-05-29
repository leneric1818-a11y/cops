"""Label-conditioned generation with OnCoCo scoring."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_ft_comparison.data import (
    extract_history,
    extract_last_utterance,
    extract_prev_category_from_history,
    extract_target_category,
    get_label_description_for_label,
    load_label_descriptions,
    load_records,
)
from llm_ft_comparison.models import CategoryPredictor, TransitionMatrix
from llm_ft_comparison.oncoco import (
    classify_message_with_oncoco_scores,
    load_oncoco_classifier,
    load_oncoco_code_mapping,
)
from llm_ft_comparison.training.unsloth_helpers import load_unsloth_model
from llm_ft_comparison.utils.label_utils import labels_equal, normalize_to_code, topk_contains


class LabelConditionedGenerationPipeline:
    def run(self, config: dict) -> str:
        dataset_cfg = config.get("dataset", {})
        label_cfg = config.get("label_descriptions", {})
        few_shot_cfg = config.get("few_shot", {})
        cat_cfg = config.get("ndap") or config.get("category_predictor", {})
        tm_cfg = config.get("transition_matrix", {})
        llm_cfg = config.get("llm", {})
        oncoco_cfg = config.get("oncoco", {})

        root = Path(__file__).resolve().parents[3]
        dataset_path = _resolve_path(dataset_cfg.get("path"), root)
        records = load_records(dataset_path)

        prompt_template = _load_prompt_template(llm_cfg, root)
        label_descs = _load_label_descriptions(label_cfg, root)
        few_shot_index = _build_few_shot_index(few_shot_cfg, root)

        target_source = dataset_cfg.get("target_source", "dataset")
        if target_source == "predicted" and not tm_cfg.get("path"):
            raise ValueError(
                "target_source=predicted requires transition_matrix.path. "
                "NDAP is a placeholder; set a transition matrix or use target_source=dataset."
            )

        if (
            target_source == "predicted"
            and not cat_cfg.get("model_dir")
            and not cat_cfg.get("use_history_label", False)
        ):
            print(
                "Warning: NDAP model not configured and use_history_label=false. "
                "Current category may be empty for some records."
            )

        cat_predictor = _load_category_predictor(cat_cfg, root)
        transition_matrix = _load_transition_matrix(tm_cfg, root)

        llm_model, llm_tokenizer = _load_llm(llm_cfg)

        oncoco_model_path = oncoco_cfg.get("model_path")
        oncoco_model_path = str(_resolve_path(oncoco_model_path, root)) if oncoco_model_path else None
        oncoco_tokenizer, oncoco_model = load_oncoco_classifier(
            model_path=oncoco_model_path, strict=True
        )
        mapping_path = oncoco_cfg.get("mapping_path")
        mapping_path = str(_resolve_path(mapping_path, root)) if mapping_path else None
        code_mapping = load_oncoco_code_mapping(mapping_path)
        topk = int(oncoco_cfg.get("topk", 3))

        output_path = _resolve_path(
            config.get("output_path", "outputs/metrics/label_conditioned_generation.jsonl"),
            root,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        summary_path = _resolve_path(
            config.get("summary_path", "outputs/metrics/label_conditioned_generation_summary.json"),
            root,
        )

        store_prompt = bool(config.get("store_prompt", False))

        metrics = _Metrics()
        with output_path.open("w", encoding="utf-8") as handle:
            for item in records:
                history = _extract_history(item, dataset_cfg)
                current_category = _get_current_category(
                    item=item,
                    history=history,
                    predictor=cat_predictor,
                    cat_cfg=cat_cfg,
                )
                target_label = _get_target_label(
                    item=item,
                    current_category=current_category,
                    transition_matrix=transition_matrix,
                    dataset_cfg=dataset_cfg,
                    target_source=target_source,
                )
                if not target_label:
                    continue

                persona_profile = _get_field(item, dataset_cfg.get("persona_field"))
                persona_name = _get_field(item, dataset_cfg.get("persona_name_field"))
                label_description = get_label_description_for_label(
                    target_label, label_descs, warn_missing=False
                )
                required_speaker = _required_speaker(target_label)
                required_speaker_de = "Klient" if required_speaker == "client" else "Berater"
                if not persona_name:
                    persona_name = required_speaker_de

                few_shot_examples = _format_few_shot_examples(
                    target_label=target_label,
                    few_shot_index=few_shot_index,
                    max_examples=int(few_shot_cfg.get("max_examples", 0) or 0),
                    seed=few_shot_cfg.get("seed"),
                )

                prompt_vars = {
                    "dialogue_history": history,
                    "current_category": current_category,
                    "next_category": target_label,
                    "target_label": target_label,
                    "label_description": label_description,
                    "personality_condition": persona_profile,
                    "persona_profile": persona_profile,
                    "name": persona_name,
                    "required_speaker": required_speaker,
                    "required_speaker_de": required_speaker_de,
                    "few_shot_examples": few_shot_examples,
                }
                prompt = prompt_template.format_map(_SafeDict(prompt_vars))

                response = _generate(llm_model, llm_tokenizer, prompt, llm_cfg)
                scored = classify_message_with_oncoco_scores(
                    response,
                    required_speaker,
                    oncoco_tokenizer,
                    oncoco_model,
                    code_mapping,
                    topk=topk,
                    strict=True,
                )

                predicted_label = scored.get("label")
                match = labels_equal(predicted_label, target_label)
                in_topk = topk_contains(scored.get("topk"), target_label)

                payload = {
                    "history": history,
                    "current_category": current_category,
                    "target_label": target_label,
                    "target_label_code": normalize_to_code(target_label),
                    "label_description": label_description,
                    "required_speaker": required_speaker,
                    "generated": response,
                    "predicted_label": predicted_label,
                    "predicted_label_code": normalize_to_code(predicted_label),
                    "label_matches_target": match,
                    "is_in_oncoco_topk": in_topk,
                    "oncoco_confidence": scored.get("confidence"),
                    "oncoco_entropy": scored.get("entropy"),
                    "oncoco_topk": scored.get("topk"),
                }
                if persona_profile:
                    payload["persona_profile"] = persona_profile
                if persona_name:
                    payload["persona_name"] = persona_name
                if store_prompt:
                    payload["prompt"] = prompt

                metrics.update(payload)
                handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

        if summary_path:
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                json.dumps(metrics.summary(), indent=2, ensure_ascii=True),
                encoding="utf-8",
            )

        return str(output_path)


class _Metrics:
    def __init__(self) -> None:
        self.count = 0
        self.match_count = 0
        self.topk_count = 0
        self.confidence_total = 0.0
        self.entropy_total = 0.0

    def update(self, payload: dict) -> None:
        self.count += 1
        if payload.get("label_matches_target"):
            self.match_count += 1
        if payload.get("is_in_oncoco_topk"):
            self.topk_count += 1
        if payload.get("oncoco_confidence") is not None:
            self.confidence_total += float(payload["oncoco_confidence"])
        if payload.get("oncoco_entropy") is not None:
            self.entropy_total += float(payload["oncoco_entropy"])

    def summary(self) -> dict:
        if self.count == 0:
            return {
                "count": 0,
                "exact_match": 0.0,
                "topk_accuracy": 0.0,
                "mean_confidence": 0.0,
                "mean_entropy": 0.0,
            }
        return {
            "count": self.count,
            "exact_match": self.match_count / self.count,
            "topk_accuracy": self.topk_count / self.count,
            "mean_confidence": self.confidence_total / self.count,
            "mean_entropy": self.entropy_total / self.count,
        }


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return ""


def _resolve_path(path_value: str | None, root: Path) -> Path:
    if not path_value:
        raise ValueError("Missing required path in config")
    expanded = os.path.expandvars(os.path.expanduser(path_value))
    path = Path(expanded)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def _load_prompt_template(llm_cfg: dict, root: Path) -> str:
    template_path = llm_cfg.get("prompt_template")
    if not template_path:
        return "{dialogue_history}\n{target_label}"
    path = _resolve_path(template_path, root)
    return path.read_text(encoding="utf-8")


def _load_label_descriptions(label_cfg: dict, root: Path) -> dict:
    path_value = label_cfg.get("path")
    if not path_value:
        return {}
    path = _resolve_path(path_value, root)
    return load_label_descriptions(str(path))


def _load_category_predictor(cat_cfg: dict, root: Path) -> CategoryPredictor | None:
    if not cat_cfg.get("model_dir"):
        return None
    model_dir = _resolve_path(cat_cfg["model_dir"], root)
    return CategoryPredictor(
        model_dir=model_dir,
        device=cat_cfg.get("device", "cuda"),
        max_length=cat_cfg.get("max_length", 256),
        top_k=cat_cfg.get("top_k", 5),
    )


def _load_transition_matrix(tm_cfg: dict, root: Path) -> TransitionMatrix | None:
    if not tm_cfg.get("path"):
        return None
    path = _resolve_path(tm_cfg["path"], root)
    return TransitionMatrix.from_csv(str(path))


def _extract_history(item: dict, dataset_cfg: dict) -> str:
    history_field = dataset_cfg.get("history_field")
    if history_field:
        value = _get_field(item, history_field)
        return _format_history(value)
    return _format_history(extract_history(item))


def _format_history(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        lines = []
        for entry in value:
            if isinstance(entry, dict):
                speaker = entry.get("speaker") or entry.get("sender") or ""
                msg = entry.get("message") or entry.get("text") or ""
                if speaker and msg:
                    lines.append(f"{speaker}: {msg}")
                elif msg:
                    lines.append(str(msg))
            else:
                lines.append(str(entry))
        return "\n".join(lines)
    return str(value)


def _get_field(item: dict, field_path: str | None) -> str:
    if not field_path:
        return ""
    value: Any = item
    for part in field_path.split("."):
        if not isinstance(value, dict):
            return ""
        value = value.get(part)
    return value or ""


def _get_current_category(
    item: dict,
    history: str,
    predictor: CategoryPredictor | None,
    cat_cfg: dict,
) -> str:
    if "current_category" in item:
        return item["current_category"]

    if cat_cfg.get("use_history_label", False):
        from_history = extract_prev_category_from_history(history)
        if from_history:
            return from_history

    if predictor:
        text = extract_last_utterance(history) if cat_cfg.get("use_last_utterance", True) else history
        predictions = predictor.predict([text])
        return predictions[0] if predictions else ""

    return ""


def _get_target_label(
    item: dict,
    current_category: str,
    transition_matrix: TransitionMatrix | None,
    dataset_cfg: dict,
    target_source: str,
) -> str:
    if target_source != "predicted":
        target_field = dataset_cfg.get("target_field")
        if target_field:
            target_value = _get_field(item, target_field)
            if target_value:
                return target_value

        target_value = extract_target_category(item)
        if target_value:
            return target_value

    return _predict_next_category(current_category, transition_matrix)


def _predict_next_category(current_category: str, matrix: TransitionMatrix | None) -> str:
    if not current_category or not matrix:
        return ""
    if current_category in matrix.probabilities:
        dist = matrix.next_distribution(current_category)
        return max(dist, key=dist.get)
    code = current_category.split("|")[0].strip()
    for label in matrix.probabilities:
        if label.split("|")[0].strip() == code:
            dist = matrix.next_distribution(label)
            return max(dist, key=dist.get)
    return ""


def _required_speaker(label: str) -> str:
    label_code = normalize_to_code(label) or ""
    if label_code.startswith("K-"):
        return "client"
    if label_code.startswith("B-"):
        return "counselor"
    return "client"


def _build_few_shot_index(few_shot_cfg: dict, root: Path) -> dict[str, list[tuple[str, str]]]:
    data_path = few_shot_cfg.get("data_path")
    if not data_path:
        return {}
    path = _resolve_path(data_path, root)
    label_field = few_shot_cfg.get("label_field", "label")
    utterance_field = few_shot_cfg.get("utterance_field", "final_utterance")
    speaker_field = few_shot_cfg.get("speaker_field")

    index: dict[str, list[tuple[str, str]]] = {}
    for row in load_records(path):
        label = normalize_to_code(row.get(label_field))
        if not label:
            continue
        utterance = row.get(utterance_field)
        if not utterance:
            continue
        speaker = row.get(speaker_field) if speaker_field else row.get("speaker") or row.get("sender")
        if not speaker:
            speaker = "client" if label.startswith("K-") else "counselor"
        index.setdefault(label, []).append((speaker, utterance))
    return index


def _format_few_shot_examples(
    target_label: str,
    few_shot_index: dict[str, list[tuple[str, str]]],
    max_examples: int,
    seed: int | None,
) -> str:
    if max_examples <= 0 or not few_shot_index:
        return ""
    label_code = normalize_to_code(target_label)
    if not label_code or label_code not in few_shot_index:
        return ""

    examples = few_shot_index[label_code]
    rng = random.Random(seed)
    selected = (
        rng.sample(examples, k=min(max_examples, len(examples)))
        if len(examples) > max_examples
        else list(examples)
    )
    lines = []
    for idx, (speaker, message) in enumerate(selected, start=1):
        speaker_de = "Klient" if speaker == "client" else "Berater"
        lines.append(f"{idx}. {speaker_de}: \"{message}\"")
    return "\n".join(lines)


def _load_llm(llm_cfg: dict):
    base_model = llm_cfg.get("base_model")
    if not base_model:
        raise ValueError("llm.base_model is required")

    use_unsloth = llm_cfg.get("use_unsloth", True)
    max_seq_len = llm_cfg.get("max_seq_len", 2048)
    if use_unsloth:
        model, tokenizer = load_unsloth_model(
            base_model=base_model,
            max_seq_len=max_seq_len,
            load_in_4bit=llm_cfg.get("load_in_4bit", True),
            dtype=llm_cfg.get("dtype"),
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        return model, tokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
    model.eval()
    return model, tokenizer


def _generate(model, tokenizer, prompt: str, llm_cfg: dict) -> str:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    gen_cfg = llm_cfg.get("generation", {})
    max_new_tokens = gen_cfg.get("max_new_tokens", 128)
    temperature = gen_cfg.get("temperature", 0.7)
    top_p = gen_cfg.get("top_p", 0.9)
    top_k = gen_cfg.get("top_k", 50)
    do_sample = gen_cfg.get("do_sample", True)
    repetition_penalty = gen_cfg.get("repetition_penalty", 1.05)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
        )

    generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if generated.startswith(prompt):
        return generated[len(prompt) :].strip()
    return generated.strip()
