"""Fine-tune DistilBERT on the processed ISOT split."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    MAX_LENGTH,
    PROCESSED_DIR,
    SEED,
    TRANSFORMER_DIR,
)

MODEL_NAME = "distilbert-base-uncased"


class NewsDataset(Dataset):
    def __init__(self, frame, tokenizer):
        self.texts = frame["text"].tolist()
        self.labels = frame["label"].astype(int).tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoded = self.tokenizer(
            self.texts[index],
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item() * batch["labels"].size(0)
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def main(sample=False):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    val = pd.read_csv(PROCESSED_DIR / "val.csv")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "FAKE", 1: "REAL"},
        label2id={"FAKE": 0, "REAL": 1},
    ).to(device)

    train_loader = DataLoader(
        NewsDataset(train, tokenizer),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        NewsDataset(val, tokenizer),
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    best_val_acc = -1.0
    history = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        seen = 0

        for step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            size = batch["labels"].size(0)
            running_loss += loss.item() * size
            seen += size

            if step % 100 == 0:
                print(f"epoch={epoch + 1} step={step}/{len(train_loader)} loss={loss.item():.4f}")

        train_loss = running_loss / max(seen, 1)
        val_loss, val_acc = evaluate(model, val_loader, device)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
        )
        print(
            f"epoch={epoch + 1} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_accuracy={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            TRANSFORMER_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(TRANSFORMER_DIR)
            tokenizer.save_pretrained(TRANSFORMER_DIR)

    with open(TRANSFORMER_DIR / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Saved model to {TRANSFORMER_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true")
    main(sample=parser.parse_args().sample)
