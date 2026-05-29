"""Category predictor training pipeline."""

from __future__ import annotations

from pathlib import Path

from llm_ft_comparison.data import extract_history, extract_last_utterance, extract_target_category, load_records
from llm_ft_comparison.models.category_predictor import CategoryPredictor


class CategoryPredictorPipeline:
    def run(self, config: dict) -> str:
        dataset_cfg = config.get("dataset", {})
        model_cfg = config.get("model", {})
        training_cfg = config.get("training", {})

        root = Path(__file__).resolve().parents[3]
        dataset_path = _resolve_path(dataset_cfg.get("path"), root)
        records = load_records(dataset_path)

        texts, labels = _extract_texts_and_labels(records, dataset_cfg)

        output_dir = _resolve_path(
            config.get("output_dir", "outputs/checkpoints/category_predictor"),
            root,
        )
        CategoryPredictor.train(
            base_model=model_cfg["base_model"],
            train_texts=texts,
            train_labels=labels,
            output_dir=output_dir,
            max_length=training_cfg.get("max_length", 256),
            num_train_epochs=training_cfg.get("epochs", 3),
            learning_rate=training_cfg.get("lr", 2e-5),
            batch_size=training_cfg.get("batch_size", 8),
            gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 1),
            seed=training_cfg.get("seed", 42),
        )

        return str(output_dir)


def _resolve_path(path_value: str | None, root: Path) -> Path:
    if not path_value:
        raise ValueError("Missing required path in config")
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def _extract_texts_and_labels(records: list[dict], dataset_cfg: dict) -> tuple[list[str], list[str]]:
    text_field = dataset_cfg.get("text_field")
    label_field = dataset_cfg.get("label_field")
    history_field = dataset_cfg.get("history_field")
    use_last_utterance = dataset_cfg.get("use_last_utterance", True)

    texts = []
    labels = []

    for item in records:
        if text_field and text_field in item:
            text = item[text_field]
        else:
            history = item.get(history_field) if history_field else extract_history(item)
            text = extract_last_utterance(history) if use_last_utterance else history

        label = item.get(label_field) if label_field else extract_target_category(item)

        if not text or not label:
            continue

        texts.append(text)
        labels.append(label)

    if not texts:
        raise ValueError("No training examples found after filtering. Check dataset fields.")

    return texts, labels
