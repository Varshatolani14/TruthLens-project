from pathlib import Path

import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parent
TRANSFORMER_DIR = ROOT / "models" / "truthlens_distilbert"
BASELINE_PATH = ROOT / "models" / "baseline.joblib"

st.set_page_config(page_title="TruthLens AI", page_icon="🔎", layout="centered")
st.title("TruthLens AI")
st.subheader("Explainable News Text Classification")
st.caption(
    "Research prototype - the model detects learned label patterns; "
    "it does not independently fact-check claims."
)


@st.cache_resource
def load_transformer():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_DIR)
    model.eval()
    return tokenizer, model


@st.cache_resource
def load_baseline():
    return joblib.load(BASELINE_PATH)


def transformer_predict(texts, tokenizer, model):
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    encoded = tokenizer(
        list(texts),
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(**encoded).logits, dim=-1)
    return probabilities.cpu().numpy()


article = st.text_area(
    "Paste a news article",
    height=260,
    placeholder="Paste the article text here...",
)

if st.button("Analyze Article", type="primary"):
    text = article.strip()
    if len(text) < 50:
        st.warning("Please paste a longer article so the classifier has useful context.")
        st.stop()

    transformer_available = TRANSFORMER_DIR.exists() and (TRANSFORMER_DIR / "config.json").exists()

    if transformer_available:
        try:
            tokenizer, model = load_transformer()
            probabilities = transformer_predict([text], tokenizer, model)[0]
            predictor = lambda texts: transformer_predict(texts, tokenizer, model)
            names = ["FAKE", "REAL"]
            mode = "DistilBERT"
        except ModuleNotFoundError:
            transformer_available = False

    if not transformer_available and BASELINE_PATH.exists():
        model = load_baseline()
        probabilities = model.predict_proba([text])[0]
        predictor = model.predict_proba
        names = ["FAKE" if int(x) == 0 else "REAL" for x in model.classes_]
        mode = "TF-IDF + Logistic Regression"
    elif not transformer_available:
        st.error("No trained model is available. Run the training scripts first.")
        st.stop()

    index = int(probabilities.argmax())
    label = names[index]
    confidence = float(probabilities[index])

    st.success(f"Prediction: {label}")
    st.metric("Confidence", f"{confidence * 100:.2f}%")
    st.write(f"Model used: **{mode}**")
    st.progress(confidence)

    st.write("### Class probabilities")
    st.write({names[i]: round(float(probabilities[i]) * 100, 2) for i in range(len(names))})

    try:
        from lime.lime_text import LimeTextExplainer

        with st.spinner("Generating local explanation..."):
            explainer = LimeTextExplainer(class_names=names, random_state=42)
            explanation = explainer.explain_instance(
                text,
                predictor,
                num_features=10,
                num_samples=250,
            )

        st.write("### Local explanation (LIME)")
        explanation_rows = [
            {"term": term, "weight": round(float(weight), 4)}
            for term, weight in explanation.as_list()
        ]
        st.dataframe(explanation_rows, use_container_width=True, hide_index=True)
        st.caption(
            "LIME is a local, approximate explanation of this prediction. "
            "It does not prove that a highlighted term is factually important."
        )
    except Exception as exc:
        st.info(f"LIME explanation is unavailable in this environment: {exc}")


st.divider()
st.header("Model Evaluation & Evidence Dashboard")
st.caption(
    "Benchmark results below come from the saved held-out test-set evaluation. "
    "They describe dataset performance and do not independently verify a pasted article."
)

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

METRICS_PATH = ROOT / "results" / "metrics.json"
if METRICS_PATH.exists():
    with METRICS_PATH.open(encoding="utf-8") as metrics_file:
        evaluation = json.load(metrics_file)

    dataset = evaluation.get("dataset", {})
    baseline_scores = evaluation.get("logistic_regression_tfidf")
    transformer_scores = evaluation.get("distilbert")

    overview = st.container(border=True)
    with overview:
        summary_columns = st.columns(4)
        summary_columns[0].metric("Held-out test articles", f"{dataset.get('test_size', 0):,}")
        summary_columns[1].metric("FAKE articles", f"{dataset.get('fake', 0):,}")
        summary_columns[2].metric("REAL articles", f"{dataset.get('real', 0):,}")
        if transformer_scores:
            summary_columns[3].metric(
                "DistilBERT test accuracy",
                f"{transformer_scores['accuracy'] * 100:.2f}%",
            )
        elif baseline_scores:
            summary_columns[3].metric(
                "Baseline test accuracy",
                f"{baseline_scores['accuracy'] * 100:.2f}%",
            )

    overview_tab, matrices_tab, formulas_tab, visuals_tab = st.tabs(
        ["📊 Model comparison", "🧮 Confusion matrices", "📐 Metrics & formulas", "🧬 Project visuals"]
    )

    with overview_tab:
        model_rows = []
        for model_name, scores in (
            ("TF-IDF + Logistic Regression", baseline_scores),
            ("DistilBERT", transformer_scores),
        ):
            if scores:
                model_rows.append(
                    {
                        "Model": model_name,
                        "Accuracy": scores["accuracy"],
                        "Precision": scores["precision"],
                        "Recall": scores["recall"],
                        "F1": scores["f1"],
                    }
                )

        if model_rows:
            comparison = pd.DataFrame(model_rows).set_index("Model")
            chart_column, table_column = st.columns([1.35, 1])
            with chart_column:
                st.subheader("Held-out performance")
                metric_names = ["Accuracy", "Precision", "Recall", "F1"]
                positions = np.arange(len(metric_names))
                bar_width = 0.8 / len(model_rows)
                chart_figure, chart_axis = plt.subplots(figsize=(8, 4.4))
                for model_index, row in enumerate(model_rows):
                    offsets = positions + (model_index - (len(model_rows) - 1) / 2) * bar_width
                    values = [row[metric] for metric in metric_names]
                    chart_axis.bar(
                        offsets,
                        values,
                        bar_width,
                        label=row["Model"],
                    )
                chart_axis.set_xticks(positions, metric_names)
                chart_axis.set_ylim(0, 1.05)
                chart_axis.set_yticks(
                    np.linspace(0, 1, 6),
                    [f"{tick:.0%}" for tick in np.linspace(0, 1, 6)],
                )
                chart_axis.set_ylabel("Test-set score")
                chart_axis.set_axisbelow(True)
                chart_axis.grid(axis="y", alpha=0.25)
                chart_axis.legend(loc="lower right", fontsize="small")
                chart_figure.tight_layout()
                st.pyplot(chart_figure, clear_figure=True)
                plt.close(chart_figure)
            with table_column:
                st.subheader("Exact scores")
                st.dataframe(
                    comparison.style.format("{:.3%}"),
                    use_container_width=True,
                )

            if baseline_scores and transformer_scores:
                f1_gain = (transformer_scores["f1"] - baseline_scores["f1"]) * 100
                st.info(
                    f"DistilBERT's F1 score is {f1_gain:.2f} percentage points "
                    "higher than the TF-IDF baseline on this saved test set."
                )

        fake_count = int(dataset.get("fake", 0))
        real_count = int(dataset.get("real", 0))
        if fake_count + real_count:
            st.subheader("Held-out dataset class balance")
            balance_figure, balance_axis = plt.subplots(figsize=(5.4, 3.4))
            balance_axis.pie(
                [fake_count, real_count],
                labels=["FAKE", "REAL"],
                autopct="%1.1f%%",
                startangle=90,
                colors=["#ef6a6a", "#4b91e2"],
                wedgeprops={"width": 0.42, "edgecolor": "white"},
                pctdistance=0.78,
            )
            balance_axis.set_title(f"{fake_count + real_count:,} test articles")
            st.pyplot(balance_figure, clear_figure=True)
            plt.close(balance_figure)

        comparison_image = ROOT / "results" / "model_comparison.png"
        if comparison_image.exists():
            st.image(comparison_image, caption="Saved test-set model comparison")

    with matrices_tab:
        st.markdown(
            "Rows are the **actual** class; columns are the **predicted** class. "
            "A correct prediction appears on the diagonal."
        )
        matrix_specs = [
            ("TF-IDF + Logistic Regression", evaluation.get("confusion_matrix_baseline")),
            ("DistilBERT", evaluation.get("confusion_matrix_distilbert")),
        ]
        available_matrices = [(name, matrix) for name, matrix in matrix_specs if matrix]
        matrix_columns = st.columns(max(1, len(available_matrices)))
        for column, (model_name, matrix) in zip(matrix_columns, available_matrices):
            with column:
                st.subheader(model_name)
                matrix_values = np.asarray(matrix, dtype=int)
                row_totals = matrix_values.sum(axis=1, keepdims=True)
                normalized_matrix = np.divide(
                    matrix_values,
                    row_totals,
                    out=np.zeros_like(matrix_values, dtype=float),
                    where=row_totals != 0,
                )
                figure, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))
                sns.heatmap(
                    matrix_values,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    cbar=False,
                    xticklabels=["FAKE", "REAL"],
                    yticklabels=["FAKE", "REAL"],
                    ax=axes[0],
                )
                axes[0].set_title("Counts")
                axes[0].set_xlabel("Predicted")
                axes[0].set_ylabel("Actual")
                sns.heatmap(
                    normalized_matrix,
                    annot=True,
                    fmt=".1%",
                    cmap="Greens",
                    vmin=0,
                    vmax=1,
                    cbar=False,
                    xticklabels=["FAKE", "REAL"],
                    yticklabels=["FAKE", "REAL"],
                    ax=axes[1],
                )
                axes[1].set_title("Row-normalized")
                axes[1].set_xlabel("Predicted")
                axes[1].set_ylabel("Actual")
                figure.tight_layout()
                st.pyplot(figure, clear_figure=True)
                plt.close(figure)
                true_negative, false_positive = matrix[0]
                false_negative, true_positive = matrix[1]
                st.caption(
                    f"Correct: {true_negative + true_positive:,} · "
                    f"False alarms: {false_positive:,} · "
                    f"Missed REAL: {false_negative:,}"
                )

        st.caption(
            "Class order is FAKE (0), REAL (1). The matrices summarize the held-out "
            "evaluation set, not a confidence or truth score for the current input."
        )

    with formulas_tab:
        st.markdown("### How the evaluation metrics are calculated")
        st.latex(r"\mathrm{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}")
        st.latex(r"\mathrm{Precision} = \frac{TP}{TP + FP}")
        st.latex(r"\mathrm{Recall} = \frac{TP}{TP + FN}")
        st.latex(r"\mathrm{F1} = 2 \cdot \frac{\mathrm{Precision}\cdot\mathrm{Recall}}{\mathrm{Precision}+\mathrm{Recall}}")
        st.markdown(
            """
            **For this dashboard:** REAL is the positive class.  
            **TP** = REAL correctly predicted REAL · **TN** = FAKE correctly predicted FAKE  
            **FP** = FAKE incorrectly predicted REAL · **FN** = REAL incorrectly predicted FAKE
            """
        )
        if baseline_scores or transformer_scores:
            score_rows = []
            for model_name, scores in (
                ("TF-IDF + Logistic Regression", baseline_scores),
                ("DistilBERT", transformer_scores),
            ):
                if scores:
                    score_rows.append(
                        {
                            "Model": model_name,
                            "Accuracy": scores["accuracy"],
                            "Precision": scores["precision"],
                            "Recall": scores["recall"],
                            "F1 score": scores["f1"],
                        }
                    )
            st.dataframe(
                pd.DataFrame(score_rows).set_index("Model").style.format("{:.3%}"),
                use_container_width=True,
            )

    with visuals_tab:
        left_visual, right_visual = st.columns(2)
        project_visuals = (
            ("figures/architecture.png", "System architecture"),
            ("figures/workflow.png", "End-to-end workflow"),
            ("results/training_curves.png", "DistilBERT training and validation loss"),
            ("results/confusion_matrix.png", "Saved evaluation confusion matrix"),
        )
        for index, (relative_path, caption) in enumerate(project_visuals):
            visual_path = ROOT / relative_path
            if visual_path.exists():
                target_column = left_visual if index % 2 == 0 else right_visual
                with target_column:
                    st.image(visual_path, caption=caption, use_container_width=True)
else:
    st.info(
        "Saved evaluation metrics are not available yet. Run the evaluation pipeline "
        "to populate results/metrics.json and show the benchmark dashboard."
    )
