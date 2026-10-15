# TruthLens AI

## Abstract

This project develops an explainable NLP-based classifier for identifying fake-news-like text patterns in a labelled news corpus. A TF-IDF + Logistic Regression model is used as a traditional baseline and a fine-tuned DistilBERT classifier is used as the transformer model. Both approaches are evaluated on the same held-out test set using accuracy, precision, recall and F1-score. LIME is then used to inspect token-level evidence for an individual prediction. A Streamlit application provides a simple real-world screening interface.

## Results table

| Model | Accuracy | Precision | Recall | F1 | Test size |
|---|---:|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | [[LR_ACC]] | [[LR_PREC]] | [[LR_REC]] | [[LR_F1]] | [[TEST_SIZE]] |
| DistilBERT | [[BERT_ACC]] | [[BERT_PREC]] | [[BERT_REC]] | [[BERT_F1]] | [[TEST_SIZE]] |

## Limitation

The model is a statistical classifier trained on a labelled corpus. Its output should be interpreted as a prediction of label-associated textual patterns, not as independent proof that a real-world claim is true or false.
