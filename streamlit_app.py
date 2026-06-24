import base64
import gzip
import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RANDOM_STATE = 42

LABEL_DISPLAY = {
    "neutral_discussion": "Neutral Discussion",
    "friendly_banter": "Friendly Banter",
    "harassment": "Harassment",
    "malicious_sarcasm": "Malicious Sarcasm",
    "trolling": "Trolling",
    "prejudicial": "Prejudicial",
}

LABEL_GROUP = {
    "neutral_discussion": "Aman",
    "friendly_banter": "Aman",
    "harassment": "Toksik",
    "malicious_sarcasm": "Toksik",
    "trolling": "Toksik",
    "prejudicial": "Toksik",
}

LABEL_NOTES = {
    "neutral_discussion": "Diskusi umum tanpa serangan personal.",
    "friendly_banter": "Candaan ringan atau komentar santai.",
    "harassment": "Serangan verbal langsung atau hinaan.",
    "malicious_sarcasm": "Sarkasme yang merendahkan atau mengejek.",
    "trolling": "Komentar provokatif, bait, atau mengganggu.",
    "prejudicial": "Serangan berbasis identitas atau stereotip kelompok.",
}


st.set_page_config(
    page_title="Cyberbullying Comment Classifier",
    page_icon="C",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
    .metric-card {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 14px 16px;
        background: #ffffff;
    }
    .small-muted {color: #64748b; font-size: 0.92rem;}
    .label-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        background: #eef2ff;
        color: #3730a3;
        font-weight: 700;
        font-size: 0.86rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name)


@st.cache_data
def read_prepared_dataset() -> pd.DataFrame:
    csv_path = DATA_DIR / "prepared_dataset.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)

    chunk_paths = sorted(DATA_DIR.glob("prepared_dataset.csv.gz.b64.*"))
    if not chunk_paths:
        raise FileNotFoundError("Missing prepared dataset CSV or compressed chunk files.")

    encoded = "".join(path.read_text(encoding="ascii").strip() for path in chunk_paths)
    decoded = gzip.decompress(base64.b64decode(encoded)).decode("utf-8")
    return pd.read_csv(io.StringIO(decoded))


@st.cache_data
def load_tables() -> dict[str, pd.DataFrame]:
    return {
        "prepared": read_prepared_dataset(),
        "model_comparison": read_csv("model_comparison.csv"),
        "tuning": read_csv("tuning_comparison_all_methods.csv"),
        "classification": read_csv("classification_report.csv"),
        "decision_offsets": read_csv("decision_offsets.csv"),
        "label_distribution": read_csv("label_distribution.csv"),
    }


@st.cache_resource
def train_final_model() -> tuple[Pipeline, dict[str, float], pd.DataFrame]:
    tables = load_tables()
    dataset = tables["prepared"].copy()
    offsets = tables["decision_offsets"].set_index("Label")["Decision offset"].to_dict()

    text_column = "raw_text" if "raw_text" in dataset.columns else "text"
    x_train = dataset[text_column].fillna("").astype(str)
    y_train = dataset["label"].astype(str)

    pipeline = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "word",
                            TfidfVectorizer(
                                lowercase=True,
                                analyzer="word",
                                token_pattern=r"(?u)\b\w+\b",
                                ngram_range=(1, 1),
                                min_df=2,
                                max_df=0.9999629727063646,
                                max_features=None,
                                sublinear_tf=False,
                            ),
                        ),
                        (
                            "char",
                            TfidfVectorizer(
                                lowercase=True,
                                analyzer="char_wb",
                                ngram_range=(2, 4),
                                min_df=3,
                                max_features=8000,
                                sublinear_tf=True,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "model",
                LinearSVC(
                    C=3.234148724269344,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    return pipeline, offsets, dataset


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum()


def predict_comment(text: str) -> tuple[str, pd.DataFrame]:
    pipeline, offsets, _ = train_final_model()
    model = pipeline.named_steps["model"]
    classes = model.classes_

    raw_scores = pipeline.decision_function([text])[0]
    offset_values = np.array([offsets.get(label, 0.0) for label in classes])
    adjusted_scores = raw_scores + offset_values
    confidence = softmax(adjusted_scores)
    prediction = classes[int(np.argmax(adjusted_scores))]

    score_table = pd.DataFrame(
        {
            "Label": classes,
            "Display": [LABEL_DISPLAY.get(label, label) for label in classes],
            "Group": [LABEL_GROUP.get(label, "-") for label in classes],
            "Raw score": raw_scores,
            "Decision offset": offset_values,
            "Adjusted score": adjusted_scores,
            "Relative confidence": confidence,
        }
    ).sort_values("Adjusted score", ascending=False)

    return prediction, score_table


def show_metric_card(title: str, value: str, caption: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="small-muted">{title}</div>
            <h2 style="margin: 0.2rem 0 0.15rem 0;">{value}</h2>
            <div class="small-muted">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_bar_chart(df: pd.DataFrame, index_col: str, value_col: str) -> None:
    chart_data = df[[index_col, value_col]].set_index(index_col)
    st.bar_chart(chart_data)


def overview_page(tables: dict[str, pd.DataFrame]) -> None:
    tuning = tables["tuning"]
    best = tuning.sort_values(["Macro F1", "Accuracy"], ascending=False).iloc[0]
    baseline = tuning[tuning["Skenario"] == "Baseline Cleaned Word TF-IDF"].iloc[0]
    dataset = tables["prepared"]

    st.title("Cyberbullying Comment Classifier")
    st.write(
        "A Streamlit app for classifying social-media comments into six cyberbullying-related labels. "
        "The app includes an interactive classifier, dataset overview, model evaluation, and deployment notes."
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        show_metric_card("Training-ready comments", f"{len(dataset):,}", "Rows used by the app model")
    with col2:
        show_metric_card("Final Accuracy", f"{best['Accuracy']:.4f}", "Overall correct predictions")
    with col3:
        show_metric_card("Final Macro F1", f"{best['Macro F1']:.4f}", "Balanced score across labels")
    with col4:
        show_metric_card(
            "Macro F1 gain",
            f"+{best['Delta Macro F1 vs Baseline']:.4f}",
            f"Baseline Macro F1: {baseline['Macro F1']:.4f}",
        )

    st.subheader("Best model")
    st.markdown(
        f"""
        <span class="label-pill">{best['Skenario']}</span>

        This model combines raw word TF-IDF, character TF-IDF, Linear SVM, Optuna tuning, and decision-offset tuning.
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Label distribution")
        st.dataframe(tables["label_distribution"], use_container_width=True, hide_index=True)
    with right:
        st.subheader("Label chart")
        show_bar_chart(tables["label_distribution"], "Label", "Jumlah")


def classifier_page() -> None:
    st.title("Try the Classifier")
    st.write(
        "Enter a comment and the app will predict one of six labels. "
        "The model uses decision scores, so the confidence values are relative, not calibrated probabilities."
    )

    examples = [
        "Kasian gua lihat anak anak nya kehilangan ibu di umur segitu",
        "Lau siape mpruy",
        "Wkwkwk kocak banget ini orang",
        "Dasar goblok, komentar lu sampah banget",
    ]
    selected_example = st.selectbox("Example comment", [""] + examples)
    default_text = selected_example or "Tulis komentar di sini..."
    text = st.text_area("Comment text", value=default_text, height=130)

    if st.button("Predict label", type="primary"):
        cleaned = text.strip()
        if not cleaned or cleaned == "Tulis komentar di sini...":
            st.warning("Please enter a comment first.")
            return

        prediction, score_table = predict_comment(cleaned)
        display_label = LABEL_DISPLAY.get(prediction, prediction)
        group = LABEL_GROUP.get(prediction, "-")

        if group == "Toksik":
            st.error(f"Prediction: {display_label} ({group})")
        else:
            st.success(f"Prediction: {display_label} ({group})")

        st.caption(LABEL_NOTES.get(prediction, ""))

        st.subheader("Decision score details")
        shown = score_table.copy()
        shown["Raw score"] = shown["Raw score"].round(4)
        shown["Decision offset"] = shown["Decision offset"].round(4)
        shown["Adjusted score"] = shown["Adjusted score"].round(4)
        shown["Relative confidence"] = (shown["Relative confidence"] * 100).round(2).astype(str) + "%"
        st.dataframe(shown, use_container_width=True, hide_index=True)

        st.info(
            "The final label is chosen from the highest adjusted decision score. "
            "Offsets help balance the model for smaller labels."
        )


def evaluation_page(tables: dict[str, pd.DataFrame]) -> None:
    st.title("Model Evaluation")
    st.write(
        "This page shows how the final model compares with baseline and tuning methods. "
        "Macro F1 is the main metric because the labels are imbalanced."
    )

    st.subheader("All-method comparison")
    st.dataframe(tables["tuning"], use_container_width=True, hide_index=True)

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.subheader("Macro F1")
        show_bar_chart(tables["tuning"], "Skenario", "Macro F1")
    with chart_col2:
        st.subheader("Accuracy")
        show_bar_chart(tables["tuning"], "Skenario", "Accuracy")

    st.subheader("Main metrics")
    metric_plot = tables["tuning"].set_index("Skenario")[
        ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]
    ]
    st.bar_chart(metric_plot)

    st.subheader("Classification report")
    st.dataframe(tables["classification"], use_container_width=True, hide_index=True)
    st.info(
        "The full notebook contains the saved confusion matrix image. "
        "This deployed app focuses on interactive prediction and metric tables."
    )

    with st.expander("Baseline model comparison"):
        st.dataframe(tables["model_comparison"], use_container_width=True, hide_index=True)


def workflow_page() -> None:
    st.title("Workflow Explanation")
    st.write("This is the end-to-end process behind the application.")

    steps = [
        (
            "1. Load the CSV dataset",
            "The app uses a prepared dataset of comments and labels exported from the notebook.",
        ),
        (
            "2. Convert text into numeric features",
            "Text is transformed with TF-IDF. Word TF-IDF captures terms, while character TF-IDF captures short patterns, slang, and typos.",
        ),
        (
            "3. Train the classifier",
            "The final model is a Linear SVM trained on raw comment text using the same tuned parameters from the notebook.",
        ),
        (
            "4. Apply decision offsets",
            "Class-level offsets adjust decision scores so minority classes are not ignored too easily.",
        ),
        (
            "5. Predict a label",
            "For a new comment, the app calculates decision scores for all labels and returns the highest adjusted score.",
        ),
        (
            "6. Explain the evaluation",
            "The Evaluation page shows Accuracy, Macro Precision, Macro Recall, Macro F1, classification report, and confusion matrix.",
        ),
    ]

    for title, body in steps:
        st.subheader(title)
        st.write(body)

    st.info(
        "The notebook used 5-fold stratified cross-validation. Each fold is approximately 80% training data "
        "and 20% validation data, repeated five times."
    )


def deployment_page() -> None:
    st.title("How to Use and Deploy")

    st.subheader("Run locally")
    st.code(
        "pip install -r requirements.txt\nstreamlit run streamlit_app.py",
        language="bash",
    )

    st.subheader("Deploy on Streamlit Community Cloud")
    st.write(
        "Put these files in a GitHub repository, connect the repository to Streamlit Community Cloud, "
        "and set the main app file to `streamlit_app.py`."
    )
    st.markdown(
        """
        Required files:

        - `streamlit_app.py`
        - `requirements.txt`
        - `data/`
        - `.streamlit/config.toml`
        """
    )

    st.subheader("Public or private repository")
    st.write(
        "A public repository is easiest if the app link needs to open without login. "
        "A private repository can also work, but viewers may need explicit access depending on the sharing settings."
    )

    st.subheader("Why Streamlit")
    st.write(
        "Streamlit turns the notebook result into a small web app. "
        "That makes the model easier to demonstrate because users can type a comment, see a prediction, "
        "and inspect the evaluation results without opening the notebook."
    )


def main() -> None:
    tables = load_tables()
    page = st.sidebar.radio(
        "Navigation",
        [
            "Overview",
            "Try classifier",
            "Evaluation",
            "Workflow explanation",
            "How to use and deploy",
        ],
    )
    st.sidebar.divider()
    st.sidebar.caption("Final model: Optuna + Decision Offset Raw Word+Char Linear SVM")

    if page == "Overview":
        overview_page(tables)
    elif page == "Try classifier":
        classifier_page()
    elif page == "Evaluation":
        evaluation_page(tables)
    elif page == "Workflow explanation":
        workflow_page()
    else:
        deployment_page()


if __name__ == "__main__":
    main()
