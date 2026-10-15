"""Evaluate the saved models on the untouched test split and create figures."""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset

from config import MAX_LENGTH, PROCESSED_DIR, RESULTS_DIR, TRANSFORMER_DIR

ROOT = Path(__file__).resolve().parents[1]
BASELINE_MODEL = ROOT / "models" / "baseline.joblib"


class NewsDataset(Dataset):
    def __init__(self, frame, tokenizer):
        self.texts = frame["text"].tolist()
        self.labels = frame["label"].astype(int).tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        tokens = self.tokenizer(
            self.texts[index],
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in tokens.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def evaluate_transformer(frame):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    loader = DataLoader(NewsDataset(frame, tokenizer), batch_size=16, shuffle=False)
    preds = []

    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            del labels
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            preds.extend(logits.argmax(dim=-1).cpu().tolist())

    return np.array(preds)


def save_confusion_matrix(y_true, y_pred, filename, title):
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(5.2, 4.2))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cbar=False,
        xticklabels=["Fake", "Real"],
        yticklabels=["Fake", "Real"],
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=220)
    plt.close()


def save_model_comparison(results):
    rows = []
    for model_name, key in [
        ("TF-IDF + Logistic Regression", "logistic_regression_tfidf"),
        ("DistilBERT", "distilbert"),
    ]:
        scores = results.get(key)
        if scores:
            rows.append({"model": model_name, **scores})

    if not rows:
        return

    frame = pd.DataFrame(rows).set_index("model")
    frame[["accuracy", "precision", "recall", "f1"]].plot(
        kind="bar", figsize=(8.8, 5.0), ylim=(0, 1.05)
    )
    plt.ylabel("Score")
    plt.title("Test-set model comparison")
    plt.xticks(rotation=0)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "model_comparison.png", dpi=220)
    plt.close()


def save_training_curves():
    history_path = TRANSFORMER_DIR / "training_history.json"
    if not history_path.exists():
        return

    history = json.loads(history_path.read_text(encoding="utf-8"))
    if not history:
        return

    history_df = pd.DataFrame(history)
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(history_df["epoch"], history_df["train_loss"], marker="o", label="Train loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], marker="o", label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("DistilBERT training and validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "training_curves.png", dpi=220)
    plt.close()


def save_error_analysis(test, predictions_by_model):
    rows = []
    for model_name, predictions in predictions_by_model.items():
        for index, row in test.iterrows():
            true_label = int(row["label"])
            pred_label = int(predictions[index])
            if true_label != pred_label:
                rows.append(
                    {
                        "model": model_name,
                        "true_label": "REAL" if true_label == 1 else "FAKE",
                        "predicted_label": "REAL" if pred_label == 1 else "FAKE",
                        "text": row["text"],
                    }
                )

    if rows:
        pd.DataFrame(rows).to_csv(RESULTS_DIR / "error_analysis.csv", index=False)


def main(sample=False):
    del sample  # Kept for CLI compatibility with the smoke-test workflow.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    y_true = test["label"].astype(int).to_numpy()

    if not BASELINE_MODEL.exists():
        raise FileNotFoundError("Baseline model not found. Run src/train_baseline.py first.")

    baseline = joblib.load(BASELINE_MODEL)
    baseline_preds = baseline.predict(test["text"])

    results = {
        "dataset": {
            "test_size": int(len(test)),
            "fake": int((test["label"] == 0).sum()),
            "real": int((test["label"] == 1).sum()),
        },
        "logistic_regression_tfidf": metrics(y_true, baseline_preds),
    }

    predictions_by_model = {"TF-IDF + Logistic Regression": baseline_preds}

    transformer_exists = TRANSFORMER_DIR.exists() and (TRANSFORMER_DIR / "config.json").exists()
    if transformer_exists:
        transformer_preds = evaluate_transformer(test)
        results["distilbert"] = metrics(y_true, transformer_preds)
        results["confusion_matrix_distilbert"] = confusion_matrix(
            y_true, transformer_preds, labels=[0, 1]
        ).tolist()
        predictions_by_model["DistilBERT"] = transformer_preds
        test.assign(prediction=transformer_preds).to_csv(
            RESULTS_DIR / "test_predictions.csv", index=False
        )
        save_confusion_matrix(
            y_true, transformer_preds, "confusion_matrix.png", "DistilBERT confusion matrix"
        )
        save_confusion_matrix(
            y_true, transformer_preds, "confusion_matrix_distilbert.png", "DistilBERT confusion matrix"
        )
    else:
        results["distilbert"] = None
        print("DistilBERT model not found. Baseline metrics and figures were still generated.")

    results["confusion_matrix_baseline"] = confusion_matrix(
        y_true, baseline_preds, labels=[0, 1]
    ).tolist()
    if not transformer_exists:
        save_confusion_matrix(
            y_true, baseline_preds, "confusion_matrix.png", "TF-IDF + Logistic Regression confusion matrix"
        )

    save_confusion_matrix(
        y_true, baseline_preds, "confusion_matrix_baseline.png", "TF-IDF + Logistic Regression confusion matrix"
    )
    save_model_comparison(results)
    save_training_curves()
    save_error_analysis(test, predictions_by_model)

    with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    rows = []
    for model_name, model_scores in (
        ("TF-IDF + Logistic Regression", results["logistic_regression_tfidf"]),
        ("DistilBERT", results["distilbert"]),
    ):
        if model_scores:
            rows.append({"model": model_name, **model_scores})
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "metrics.csv", index=False)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true")
    main(sample=parser.parse_args().sample)
