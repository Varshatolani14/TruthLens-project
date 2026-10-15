# Exact AAT Checklist

## Part 1 - Topic Exploration and Conceptual Understanding

[ ] Form a team of two.

[ ] NLP topic selected: fake news text classification.

[ ] Real-world application identified: article screening / moderation / newsroom triage.

[ ] Architecture studied: preprocessing -> TF-IDF baseline / DistilBERT -> classification -> explanation.

[ ] Literature reviewed: BERT, DistilBERT, fake-news detection, LIME.

[ ] Preliminary report prepared: use `docs/TruthLens_AAT_Report_Template.docx`.

## Part 2 - Implementation, Validation and Presentation

[ ] Download `True.csv` and `Fake.csv`.

[ ] Run preprocessing.

[ ] Train TF-IDF + Logistic Regression.

[ ] Fine-tune DistilBERT.

[ ] Evaluate on held-out test split.

[ ] Generate accuracy, precision, recall and F1.

[ ] Generate confusion matrix.

[ ] Compare both models.

[ ] Generate one LIME explanation.

[ ] Run Streamlit demo.

[ ] Optionally run FastAPI demo.

[ ] Run `scripts/build_submission_docs.py`.

[ ] Replace student names, USNs, class, semester and faculty name.

[ ] Check that all report metrics match `results/metrics.json`.

[ ] Add screenshots from the real application and real LIME explanation to the report/PPT.

[ ] Submit the final report and PPT.
