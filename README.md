# TruthLens AI - NLP AAT Project

**Title:** TruthLens AI: A Comparative Study of Traditional and Transformer-Based NLP Models for Explainable Fake News Detection

This project is built to match the AAT activity sheet:

1. Topic exploration and conceptual understanding
2. Real-world application and architecture study
3. Dataset-based NLP implementation
4. Pre-processing, training/testing and evaluation
5. Comparison of approaches
6. Interpretation using explainability
7. Report + presentation

## What is implemented

- ISOT Fake News Dataset ingestion
- Text cleaning and duplicate removal
- Stratified train/validation/test split
- TF-IDF + Logistic Regression baseline
- DistilBERT fine-tuning for binary classification
- Accuracy, precision, recall and F1 evaluation
- Confusion matrix and model comparison figures
- LIME-based local explanation
- Streamlit application
- Optional FastAPI endpoint
- Report and PPT templates
- Colab notebook that runs the same pipeline

## Important note about the claim

The classifier learns patterns associated with labels in the training corpus. It is **not a fact-checking engine** and does not independently verify the truth of every sentence in a new article. The report deliberately states this limitation.

## 1. Dataset

Use the original ISOT Fake News Dataset from the University of Victoria ISOT Research Lab:

https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/

The official download is a ZIP containing `True.csv` and `Fake.csv`.

Run:

```bash
python scripts/download_data.py
```

The script checks that both CSV files exist and contain a `text` column.

If the official download is unavailable from your network, download the same two files manually and place them here:

```text
data/raw/True.csv
data/raw/Fake.csv
```

## 2. Environment

Python 3.10 or 3.11 is recommended.

Create a virtual environment:

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For the transformer stage, use Google Colab with a GPU if possible. Local CPU execution is supported but can be slow.

## 3. Run a complete pipeline

After the dataset is in `data/raw`:

```bash
python src/preprocess.py
python src/train_baseline.py
python src/train_transformer.py
python src/evaluate.py
python src/explain.py
```

The transformer script defaults to 10,000 articles: 5,000 fake + 5,000 real. The split is 70/15/15, stratified.

For a quick smoke test, use the included small sample dataset:

```bash
python src/preprocess.py --sample
python src/train_baseline.py --sample
python src/evaluate.py --sample
```

The sample data is only for checking that the pipeline wiring works. **Do not use the sample results in the college report.**

## 4. Streamlit app

After the transformer model is trained:

```bash
streamlit run app.py
```

The app loads `models/truthlens_distilbert` and displays:

- predicted label
- model confidence
- class probabilities
- an explanation panel when LIME is available

If the transformer model is not present, the app can fall back to the saved TF-IDF baseline for a basic demo.

## 5. FastAPI API

```bash
uvicorn api:app --reload
```

Then POST JSON like:

```json
{"text": "Paste an article here"}
```

to `/predict`.

## 6. Files produced

```text
results/
  metrics.json
  metrics.csv
  confusion_matrix.png
  model_comparison.png
  lime_example.html
figures/
  architecture.png
  workflow.png
models/
  baseline.joblib
  tfidf.joblib
  truthlens_distilbert/
```

## 7. Report and PPT

The DOCX report and PPT are deliberately written around the exact AAT requirements. They use placeholders such as `[[BERT_F1]]` where measured values must be inserted after the real run.

Run:

```bash
python scripts/build_submission_docs.py
```

after `results/metrics.json` exists. This replaces the metric placeholders with the actual results from your experiment.

## 8. Recommended presentation order

1. Problem statement
2. Why fake news detection is difficult
3. Literature review
4. Dataset and preprocessing
5. TF-IDF baseline
6. DistilBERT architecture
7. End-to-end workflow
8. Experimental setup
9. Results
10. Confusion matrix
11. LIME explanation
12. Application + limitations + conclusion

## 9. What not to do

- Do not invent accuracy or F1 numbers.
- Do not call the system a universal fact checker.
- Do not claim that high confidence means an article is objectively true.
- Do not remove the limitations section from the report.
