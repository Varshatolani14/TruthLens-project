"""Run the parts of the project that can be checked without downloading a transformer model."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    import sys
    sys.path.insert(0, str(ROOT / "src"))

    from preprocess import main as preprocess_main
    from train_baseline import main as baseline_main
    from evaluate import main as evaluate_main

    preprocess_main(sample=True)
    baseline_main(sample=True)
    evaluate_main(sample=True)

    metrics = json.loads((ROOT / "results" / "metrics.json").read_text(encoding="utf-8"))
    assert "logistic_regression_tfidf" in metrics
    print("Smoke test passed. Baseline pipeline is wired correctly.")


if __name__ == "__main__":
    main()
