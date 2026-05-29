"""Category -> transition matrix -> LLM generation pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_ft_comparison.data import (
    extract_history,
    extract_last_utterance,
    extract_prev_category_from_history,
    load_records,
)
from llm_ft_comparison.models import CategoryPredictor, TransitionMatrix
from llm_ft_comparison.training.unsloth_helpers import load_unsloth_model


class CategoryChainPipeline:
    def run(self, config: dict) -> str:
        dataset_cfg = config.get("dataset", {})
        cat_cfg = config.get("category_predictor", {})
        tm_cfg = config.get("transition_matrix", {})
        llm_cfg = config.get("llm", {})

        root = Path(__file__).resolve().parents[3]
        dataset_path = _resolve_path(dataset_cfg.get("path"), root)
        records = load_records(dataset_path)

        prompt_template_path = _resolve_path(llm_cfg.get("prompt_template"), root)
        if prompt_template_path:
            prompt_template = prompt_template_path.read_text(encoding="utf-8")
        else:
            prompt_template = "{dialogue_history}\n{next_category}"

        cat_predictor = None
        if cat_cfg.get("model_dir"):
            cat_predictor = CategoryPredictor(
                model_dir=_resolve_path(cat_cfg["model_dir"], root),
                device=cat_cfg.get("device", "cuda"),
                max_length=cat_cfg.get("max_length", 256),
                top_k=cat_cfg.get("top_k", 5),
            )

        transition_path = _resolve_path(tm_cfg.get("path"), root)
        transition_matrix = TransitionMatrix.from_csv(str(transition_path)) if transition_path else None

        llm_model, llm_tokenizer = self._load_llm(llm_cfg)

        output_path = _resolve_path(
            config.get("output_path", "outputs/metrics/category_chain.jsonl"),
            root,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as handle:
            for item in records:
                history = extract_history(item)
                current_category = _get_current_category(item, history, cat_predictor, cat_cfg)
                next_category = _predict_next_category(current_category, transition_matrix)

                prompt = prompt_template.format(
                    dialogue_history=history,
                    current_category=current_category,
                    next_category=next_category,
                )

                response = _generate(llm_model, llm_tokenizer, prompt, llm_cfg)
                payload = {
                    "history": history,
                    "current_category": current_category,
                    "next_category": next_category,
                    "generated": response,
                }
                handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

        return str(output_path)

    def _load_llm(self, llm_cfg: dict):
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


def _resolve_path(path_value: str | None, root: Path) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


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
        return generated[len(prompt):].strip()
    return generated.strip()
