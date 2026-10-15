# Google Colab quick start

Upload or clone the `TruthLens_AAT` folder into Colab.

1. Runtime -> Change runtime type -> choose a GPU if available.
2. Change directory into the project folder.
3. Run the notebook `notebooks/TruthLens_AAT.ipynb` from top to bottom.
4. Do not use results until the final `evaluate.py` cell completes.
5. Run `scripts/build_submission_docs.py` to create the submission versions of the report and PPT.

The full DistilBERT run can take a while. That is expected. The quick smoke test is only for debugging and must not be used for the final report.
