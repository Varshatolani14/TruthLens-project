from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

SEED = 42
SAMPLES_PER_CLASS = 5000
MAX_LENGTH = 256
BATCH_SIZE = 8
EPOCHS = 2
LEARNING_RATE = 2e-5

TRANSFORMER_DIR = MODELS_DIR / "truthlens_distilbert"
BASELINE_MODEL = MODELS_DIR / "baseline.joblib"
TFIDF_MODEL = MODELS_DIR / "tfidf.joblib"
