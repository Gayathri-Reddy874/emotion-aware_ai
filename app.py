"""
Emotion AI App — Streamlit front end.

This module is intentionally "thin": it only handles page layout, user
input, and rendering. All model logic lives in core/engine.py, all
validation/formatting in core/utils.py, and all charting in
core/visualization.py. This separation keeps the UI testable-by-hand
and the logic testable-by-pytest.
"""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from core.config import APP_CONFIG
from core.engine import EmotionEngine, InferenceError, ModelLoadError
from core.utils import (
    ValidationError,
    format_percentage,
    get_emotion_style,
    is_low_confidence,
    predictions_to_rows,
    validate_text,
)
from core.visualization import build_emotion_bar_chart

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title=APP_CONFIG.page_title,
    page_icon=APP_CONFIG.page_icon,
    layout="centered",
)


@st.cache_resource(show_spinner=False)
def get_engine() -> EmotionEngine:
    """Load and cache the emotion engine once per session/process."""
    return EmotionEngine().load()


def render_single_result(text: str, predictions: list[dict]) -> None:
    """Render the top emotion, a confidence chart, and a low-confidence warning."""
    top = predictions[0]
    style = get_emotion_style(top["label"])

    st.success(f"{style['emoji']} **Top Emotion: {top['label'].title()}** — {format_percentage(top['score'])}")

    if is_low_confidence(top["score"]):
        st.warning(
            "The model isn't very confident about this prediction — the text may be "
            "ambiguous, too short, or express mixed emotions."
        )

    st.subheader("Confidence Breakdown")
    st.plotly_chart(build_emotion_bar_chart(predictions), use_container_width=True)

    with st.expander("Raw scores"):
        df = pd.DataFrame(predictions_to_rows(text, predictions)).drop(columns=["text"])
        st.dataframe(df, hide_index=True, use_container_width=True)


def render_single_mode(engine: EmotionEngine) -> None:
    st.write("Enter text below and the model will detect the underlying emotion(s).")

    user_text = st.text_area(
        "Enter your text:",
        max_chars=APP_CONFIG.max_chars,
        height=140,
        placeholder="e.g. I can't believe I finally finished this project!",
    )

    if st.button("Analyze Emotion", type="primary"):
        try:
            cleaned = validate_text(user_text)
            with st.spinner("Analyzing..."):
                predictions = engine.predict(cleaned)
            render_single_result(cleaned, predictions)

            # Keep a running history for this session.
            st.session_state.setdefault("history", [])
            st.session_state["history"].append(
                {"text": cleaned, "top_emotion": predictions[0]["label"], "confidence": predictions[0]["score"]}
            )

        except ValidationError as exc:
            st.warning(str(exc))
        except InferenceError as exc:
            st.error(str(exc))

    history = st.session_state.get("history", [])
    if history:
        with st.expander(f"Session history ({len(history)})"):
            hist_df = pd.DataFrame(history)
            hist_df["confidence"] = hist_df["confidence"].apply(format_percentage)
            st.dataframe(hist_df, hide_index=True, use_container_width=True)


def render_batch_mode(engine: EmotionEngine) -> None:
    st.write(
        f"Upload a CSV with a `text` column to analyze up to {APP_CONFIG.max_batch_rows} "
        "rows at once — useful for reviews, survey responses, or support tickets."
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is None:
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception:
        st.error("Could not read that file. Please upload a valid CSV.")
        return

    if "text" not in df.columns:
        st.error("The CSV must contain a column named `text`.")
        return

    if len(df) > APP_CONFIG.max_batch_rows:
        st.warning(f"File has {len(df)} rows; only the first {APP_CONFIG.max_batch_rows} will be analyzed.")
        df = df.head(APP_CONFIG.max_batch_rows)

    if st.button("Analyze Batch", type="primary"):
        texts = df["text"].astype(str).tolist()
        with st.spinner(f"Analyzing {len(texts)} rows..."):
            results = engine.predict_batch(texts)

        all_rows = []
        for text, predictions in zip(texts, results):
            if not predictions:
                all_rows.append({"text": text, "emotion": "ERROR", "confidence": None})
            else:
                top = predictions[0]
                all_rows.append({"text": text, "emotion": top["label"], "confidence": round(top["score"], 4)})

        result_df = pd.DataFrame(all_rows)
        st.success(f"Analyzed {len(result_df)} rows.")
        st.dataframe(result_df, hide_index=True, use_container_width=True)

        st.download_button(
            "Download results as CSV",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="emotion_analysis_results.csv",
            mime="text/csv",
        )


def main() -> None:
    st.title(f"{APP_CONFIG.page_icon} Emotion-Aware AI App")
    st.caption("Detect emotions from text using a fine-tuned DistilRoBERTa transformer model.")

    try:
        with st.spinner("Loading model (first run may take a minute)..."):
            engine = get_engine()
    except ModelLoadError as exc:
        st.error(str(exc))
        st.stop()
        return

    tab_single, tab_batch = st.tabs(["Single Text", "Batch (CSV)"])
    with tab_single:
        render_single_mode(engine)
    with tab_batch:
        render_batch_mode(engine)

    st.divider()
    st.caption(
        "Model: [j-hartmann/emotion-english-distilroberta-base]"
        "(https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) · "
        "7 emotions: anger, disgust, fear, joy, neutral, sadness, surprise"
    )


if __name__ == "__main__":
    main()
