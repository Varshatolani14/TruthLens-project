from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
TRANSFORMER_DIR = ROOT / "models" / "truthlens_distilbert"
BASELINE_PATH = ROOT / "models" / "baseline.joblib"

app = FastAPI(title="TruthLens AI API", version="1.0")


class Article(BaseModel):
    text: str = Field(min_length=50)


class Prediction(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]
    model: str


def predict_with_transformer(text):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    encoded = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(**encoded).logits, dim=-1)[0].cpu().tolist()
    labels = ["FAKE", "REAL"]
    idx = int(max(range(len(probs)), key=lambda i: probs[i]))
    return Prediction(
        label=labels[idx],
        confidence=float(probs[idx]),
        probabilities=dict(zip(labels, map(float, probs))),
        model="DistilBERT",
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_available": bool(
            (TRANSFORMER_DIR / "config.json").exists() or BASELINE_PATH.exists()
        ),
    }


@app.post("/predict", response_model=Prediction)
def predict(article: Article):
    try:
        if TRANSFORMER_DIR.exists() and (TRANSFORMER_DIR / "config.json").exists():
            try:
                return predict_with_transformer(article.text)
            except ModuleNotFoundError:
                pass

        if BASELINE_PATH.exists():
            model = joblib.load(BASELINE_PATH)
            probabilities = model.predict_proba([article.text])[0]
            classes = ["FAKE" if int(x) == 0 else "REAL" for x in model.classes_]
            idx = int(probabilities.argmax())
            return Prediction(
                label=classes[idx],
                confidence=float(probabilities[idx]),
                probabilities=dict(zip(classes, map(float, probabilities))),
                model="TF-IDF + Logistic Regression",
            )

        raise HTTPException(status_code=503, detail="No trained model is available")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
