"""Clean the ISOT files and create reproducible train/val/test CSV files."""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from config import PROCESSED_DIR, RAW_DIR, SEED, SAMPLES_PER_CLASS


def clean_text(value: str) -> str:
    text = str(value)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"^\s*[^\n]{1,100}\(Reuters\)\s*[-:–]?\s*", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_data(sample=False):
    if sample:
        path = Path(__file__).resolve().parents[1] / "data" / "sample" / "sample_news.csv"
        df = pd.read_csv(path)
    else:
        real = pd.read_csv(RAW_DIR / "True.csv")
        fake = pd.read_csv(RAW_DIR / "Fake.csv")
        real = real.assign(label=1)
        fake = fake.assign(label=0)
        df = pd.concat([real, fake], ignore_index=True)

    required = {"text", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns {sorted(required)}")

    df = df[["text", "label"]].dropna().copy()
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.len() >= 100]
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    if df["label"].nunique() != 2:
        raise ValueError("The dataset must contain both label classes.")

    if not sample:
        available = df.groupby("label").size()
        n = min(SAMPLES_PER_CLASS, int(available.min()))
        if n < SAMPLES_PER_CLASS:
            print(
                f"Only {n:,} usable examples per class are available after cleaning; "
                f"requested {SAMPLES_PER_CLASS:,}."
            )
        df = (
            df.groupby("label", group_keys=False)
            .sample(n=n, random_state=SEED)
            .sample(frac=1.0, random_state=SEED)
            .reset_index(drop=True)
        )

    return df


def main(sample=False):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data(sample=sample)

    train, temp = train_test_split(
        df,
        test_size=0.30,
        random_state=SEED,
        stratify=df["label"],
    )
    val, test = train_test_split(
        temp,
        test_size=0.50,
        random_state=SEED,
        stratify=temp["label"],
    )

    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)

    info = {
        "sample_run": sample,
        "seed": SEED,
        "total": int(len(df)),
        "train": int(len(train)),
        "validation": int(len(val)),
        "test": int(len(test)),
        "samples_per_class": None if sample else int(df.groupby("label").size().min()),
    }
    (PROCESSED_DIR / "data_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    print(f"Total: {len(df)}")
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print(df["label"].value_counts().sort_index())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()
    main(sample=args.sample)
