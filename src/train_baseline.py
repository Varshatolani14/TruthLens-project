"""Train the traditional TF-IDF + Logistic Regression baseline."""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from config import BASELINE_MODEL, PROCESSED_DIR, RESULTS_DIR, SEED, TFIDF_MODEL


def main(sample=False):
    train_path = PROCESSED_DIR / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError("Run preprocess.py first.")

    train = pd.read_csv(train_path)

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=20000,
                    ngram_range=(1, 2),
                    stop_words="english",
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=SEED,
                ),
            ),
        ]
    )

    model.fit(train["text"], train["label"])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    BASELINE_MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, BASELINE_MODEL)

    # Saving a named vectorizer is handy for inspection and viva questions.
    vectorizer = model.named_steps["tfidf"]
    joblib.dump(vectorizer, TFIDF_MODEL)

    print(f"Saved baseline model to {BASELINE_MODEL}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_):,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true")
    main(sample=parser.parse_args().sample)
