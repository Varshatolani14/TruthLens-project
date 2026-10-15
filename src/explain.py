"""Create a LIME explanation for one held-out test example."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from lime.lime_text import LimeTextExplainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import MAX_LENGTH, PROCESSED_DIR, RESULTS_DIR, TRANSFORMER_DIR


def main(sample=False):
    del sample
    if not TRANSFORMER_DIR.exists():
        raise FileNotFoundError("Train the DistilBERT model first.")

    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    if test.empty:
        raise ValueError("The test split is empty.")

    text = test.iloc[0]["text"]

    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    def predict_proba(texts):
        encoded = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(**encoded).logits, dim=-1)
        return probs.cpu().numpy()

    probs = predict_proba([text])[0]
    predicted_label = int(probs.argmax())
    names = ["FAKE", "REAL"]

    explainer = LimeTextExplainer(
        class_names=names,
        random_state=42,
    )
    explanation = explainer.explain_instance(
        text,
        predict_proba,
        labels=[predicted_label],
        num_features=12,
        num_samples=500,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    html_path = RESULTS_DIR / "lime_example.html"
    explanation.save_to_file(str(html_path))

    values = explanation.as_list(label=predicted_label)
    terms = [term for term, _ in values]
    weights = [weight for _, weight in values]

    plt.figure(figsize=(8.2, 5.0))
    positions = list(range(len(terms)))
    plt.barh(positions, weights)
    plt.yticks(positions, terms)
    plt.axvline(0, linewidth=0.8)
    plt.xlabel("LIME weight")
    plt.title(f"Local explanation for predicted class: {names[predicted_label]}")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "lime_example.png", dpi=220)
    plt.close()

    print(f"Predicted class for explanation: {names[predicted_label]}")
    print(f"Saved explanation to {html_path}")
    print(f"Saved explanation figure to {RESULTS_DIR / 'lime_example.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true")
    main(sample=parser.parse_args().sample)
