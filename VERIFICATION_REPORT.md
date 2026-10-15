# TruthLens AI - Verification Report

## Verification date
1 October 2026

## What was checked

### AAT requirements
- Topic exploration and conceptual understanding: included in the report and PPT.
- NLP topic: fake news text classification.
- Real-world application/API: Streamlit interactive article screening prototype and FastAPI `/predict` API.
- Architecture/model study: TF-IDF + Logistic Regression baseline and DistilBERT transformer branch.
- Literature review: BERT, DistilBERT, fake-news detection and LIME references.
- Dataset implementation: ISOT `True.csv` + `Fake.csv` ingestion.
- Pre-processing: missing-value handling, URL removal, Reuters-prefix cleanup, whitespace normalization and duplicate removal.
- Train/validation/test split: 70/15/15 stratified split with fixed seed 42.
- Model training/testing: baseline training and DistilBERT fine-tuning code are included.
- Metrics: accuracy, precision, recall and F1.
- Comparison: baseline versus DistilBERT using the same held-out test set.
- Interpretation: confusion matrix, model-comparison chart, training curves and LIME local explanation.
- Report and PPT generation: automated metric replacement and figure insertion.

## Automated checks completed

- Python `compileall`: passed for all project Python files.
- Small end-to-end smoke test: passed for preprocessing, TF-IDF training and held-out evaluation.
- Smoke test generated the expected metrics and confusion/model-comparison artifacts.
- Smoke-test results are explicitly blocked from being used to build final submission documents.
- FastAPI `/health`: passed.
- FastAPI `/predict` using the baseline fallback: passed.
- FastAPI request validation for too-short text: passed.
- DOCX report: rendered successfully and visually inspected page-by-page.
- PPTX template: rendered successfully and visually inspected slide-by-slide.
- Submission-document builder: tested using a synthetic final-like metrics file; DOCX/PPTX generation and rendering passed.
- Final submission builder layout was tested with model-comparison and LIME figure placeholders to catch clipping/overlap.

## What could not be executed in this environment

The execution environment used for this verification has no outbound network/DNS access and does not have the Hugging Face `transformers`, `lime`, or Streamlit packages preinstalled. Therefore:

1. The official ISOT ZIP could not be downloaded here.
2. DistilBERT base-model weights could not be downloaded here.
3. A real DistilBERT fine-tuning run could not be performed here.
4. A real LIME run could not be performed here.
5. The Streamlit web UI could not be launched here.

Those portions were checked for syntax, control flow, file paths, model-loading logic, fallback behavior, and layout integration, but they are **not claimed as fully executed in this environment**.

## Real-data requirement before submission

Run the real ISOT dataset pipeline in Google Colab or a network-enabled machine:

```text
python scripts/run_all.py
```

This runs:

1. Official dataset download
2. Preprocessing
3. TF-IDF baseline training
4. DistilBERT fine-tuning
5. Test-set evaluation
6. LIME explanation
7. Report/PPT generation

The generated report/PPT will contain the measured metrics only after this run completes.

## AI-written appearance

The source uses normal project structure, straightforward names and ordinary Python comments/docstrings. There is no reliable way to guarantee that any third-party AI detector will classify code or prose as human-written. No detector result is being claimed here.
