# TruthLens AI - 12-slide presentation outline

1. **Title** - TruthLens AI: Explainable Fake News Detection with NLP
2. **Problem** - misinformation screening is difficult to scale manually
3. **Objectives** - implement, compare, evaluate and explain two NLP approaches
4. **Literature Review** - BERT, DistilBERT, fake-news detection, LIME
5. **Dataset** - ISOT Fake News Dataset; cleaning and stratified split
6. **System Architecture** - preprocessing -> baseline / transformer -> evaluation -> explanation
7. **Baseline** - TF-IDF + Logistic Regression
8. **Transformer** - DistilBERT fine-tuning and classification head
9. **Experimental Setup** - seed 42, 70/15/15 split, 256 tokens, 2 epochs
10. **Results** - LR accuracy [[LR_ACC]], F1 [[LR_F1]]; DistilBERT accuracy [[BERT_ACC]], F1 [[BERT_F1]]
11. **Explainability + Demo** - LIME and Streamlit application
12. **Conclusion + Limitations** - comparative findings and responsible interpretation
