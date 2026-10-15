# Viva questions and answers

## 1. Why did you choose fake-news detection?
It is a real-world text classification problem where NLP is used to screen large volumes of articles. The same pipeline covers preprocessing, feature extraction, supervised learning and interpretation.

## 2. Why TF-IDF as a baseline?
TF-IDF is simple, fast and interpretable. It creates a reference point before using a much heavier transformer model.

## 3. Why DistilBERT?
It keeps the transformer architecture of BERT while being smaller and faster, making it practical for an academic prototype.

## 4. Why is a validation set needed?
The validation set is used during model development and checkpoint selection. The test set remains untouched until final evaluation.

## 5. Why use F1-score?
F1 combines precision and recall. It is useful when both false positives and false negatives matter.

## 6. What does LIME do?
LIME perturbs the input text and builds a local surrogate explanation around one prediction. It helps show which words contributed to that particular decision.

## 7. Is this a fact-checker?
No. It predicts labels learned from the dataset. It does not retrieve evidence from trusted sources and verify the underlying claim.

## 8. Why use a random seed?
A fixed seed makes the split and experiment reproducible so another run can be compared fairly.

## 9. What is the biggest limitation?
Dataset bias and source leakage can make performance look better than performance on truly new sources. The system therefore needs cross-dataset evaluation before any high-stakes deployment.

## 10. What would you improve next?
Add a source-independent external test set, calibrate probabilities, retrieve evidence from trusted sources, and evaluate robustness to paraphrasing and adversarial wording.
