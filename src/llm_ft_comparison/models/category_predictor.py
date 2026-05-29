"""Category prediction model wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pickle
import torch
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments


class CategoryPredictor:
    """Category predictor using a HuggingFace sequence classification model."""

    def __init__(
        self,
        model_dir: str | Path,
        device: str = "cuda",
        max_length: int = 256,
        top_k: int = 5,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.top_k = top_k

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model.to(self.device)
        self.model.eval()

        label_path = self.model_dir / "label_encoder.pkl"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label encoder at {label_path}")
        with label_path.open("rb") as handle:
            self.label_encoder = pickle.load(handle)

    @staticmethod
    def train(
        base_model: str,
        train_texts: list[str],
        train_labels: list[str],
        output_dir: str | Path,
        max_length: int = 256,
        num_train_epochs: int = 3,
        learning_rate: float = 2e-5,
        batch_size: int = 8,
        gradient_accumulation_steps: int = 1,
        eval_texts: list[str] | None = None,
        eval_labels: list[str] | None = None,
        seed: int = 42,
    ) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        label_encoder = LabelEncoder()
        label_encoder.fit(train_labels)

        label2id = {label: idx for idx, label in enumerate(label_encoder.classes_)}
        id2label = {idx: label for label, idx in label2id.items()}

        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=len(label2id),
            label2id=label2id,
            id2label=id2label,
        )

        def tokenize(texts: list[str]) -> dict:
            return tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )

        train_encodings = tokenize(train_texts)
        train_labels_idx = label_encoder.transform(train_labels)
        train_dataset = _ArrayDataset(train_encodings, train_labels_idx)

        eval_dataset = None
        if eval_texts and eval_labels:
            eval_encodings = tokenize(eval_texts)
            eval_labels_idx = label_encoder.transform(eval_labels)
            eval_dataset = _ArrayDataset(eval_encodings, eval_labels_idx)

        args = TrainingArguments(
            output_dir=str(output_path),
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=200 if eval_dataset else None,
            logging_steps=50,
            save_steps=200,
            save_total_limit=2,
            load_best_model_at_end=bool(eval_dataset),
            seed=seed,
            report_to=[],
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=_compute_accuracy,
        )
        trainer.train()

        trainer.save_model(str(output_path))
        tokenizer.save_pretrained(str(output_path))
        with (output_path / "label_encoder.pkl").open("wb") as handle:
            pickle.dump(label_encoder, handle)

    def predict_top_k(self, texts: Iterable[str]) -> list[list[dict]]:
        results = []
        for text in texts:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

            top_indices = np.argsort(probs)[-self.top_k:][::-1]
            predictions = []
            for idx in top_indices:
                category = self.label_encoder.inverse_transform([idx])[0]
                predictions.append({"category": category, "probability": float(probs[idx])})
            results.append(predictions)
        return results

    def predict(self, texts: Iterable[str]) -> list[str]:
        top_k = self.predict_top_k(texts)
        return [preds[0]["category"] if preds else "" for preds in top_k]


class _ArrayDataset(torch.utils.data.Dataset):
    def __init__(self, encodings: dict, labels: list[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:  # noqa: D401
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def _compute_accuracy(eval_pred) -> dict:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean() if len(labels) else 0.0
    return {"accuracy": accuracy}
