from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from src.config import METRICS_JSON
from src.predict import (
    StartupInput,
    StartupSuccessPredictor,
    probability_band,
)


st.set_page_config(
    page_title="Startup App Success Analyzer",
    page_icon="📱",
    layout="wide",
)

st.title("Startup App Success Analyzer")
st.caption(
    "Semantic competitor discovery + machine-learning success-profile prediction"
)

st.info(
    "This is a comparative ML estimate, not a guarantee of future commercial "
    "success. The target is derived from current Google Play installs and reviews."
)


@st.cache_resource
def load_predictor() -> StartupSuccessPredictor:
    return StartupSuccessPredictor()


try:
    predictor = load_predictor()
except Exception as exc:
    st.error(
        "Model artifacts are not ready. Run the pipeline first:\n\n"
        "`python run_pipeline.py --skip-scrape` if you already have data, "
        "or `python run_pipeline.py` to collect live data."
    )
    st.exception(exc)
    st.stop()


with st.sidebar:
    st.header("Startup settings")
    genre = st.text_input(
        "Expected Play Store category",
        value="Unknown",
    )
    content_rating = st.selectbox(
        "Content rating",
        ["Everyone", "Teen", "Mature 17+", "Everyone 10+", "Unknown"],
    )
    free = st.checkbox("Free app", value=True)
    offers_iap = st.checkbox("Will offer in-app purchases", value=False)
    contains_ads = st.checkbox("Will contain ads", value=False)
    price = st.number_input(
        "Launch price",
        min_value=0.0,
        value=0.0,
        step=0.99,
        disabled=free,
    )
    top_k = st.slider(
        "Comparable apps",
        min_value=10,
        max_value=100,
        value=int(predictor.bundle.get("top_k", 30)),
        step=5,
    )

summary = st.text_input(
    "Short Play Store summary",
    placeholder="Example: Your AI personal workout and nutrition coach",
)
description = st.text_area(
    "Full startup app description",
    height=220,
    placeholder=(
        "Example: An AI-powered fitness application that generates personalized "
        "workouts, tracks nutrition, adapts routines from progress, and helps "
        "users build long-term exercise habits."
    ),
)

if st.button("Analyze startup idea", type="primary"):
    if len(description.strip()) < 40:
        st.warning("Please enter a fuller app description.")
    else:
        with st.spinner("Finding semantic competitors and running the model..."):
            result = predictor.predict(
                StartupInput(
                    description=description.strip(),
                    summary=summary.strip(),
                    genre=genre.strip() or "Unknown",
                    content_rating=content_rating,
                    free=free,
                    price=0.0 if free else float(price),
                    offers_iap=offers_iap,
                    contains_ads=contains_ads,
                ),
                top_k=top_k,
            )

        probability = result["probability"]
        market = result["market"]

        st.subheader("Prediction")
        col1, col2, col3 = st.columns(3)
        col1.metric("Success-profile score", f"{probability:.1%}")
        col2.metric("Model", result["model_name"].replace("_", " ").title())
        col3.metric("Assessment", probability_band(probability))

        st.subheader("Comparable-market signals")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Median installs",
            f"{market['market_median_installs']:,.0f}",
        )
        c2.metric(
            "Median reviews",
            f"{market['market_median_reviews']:,.0f}",
        )
        c3.metric(
            "Mean rating",
            f"{market['market_mean_rating']:.2f}",
        )
        c4.metric(
            "Successful-neighbor ratio",
            f"{market['market_success_ratio']:.1%}",
        )

        st.caption(
            f"Mean semantic similarity: "
            f"{market['market_mean_similarity']:.3f} · "
            f"Close competitors (similarity ≥ 0.70): "
            f"{market['market_close_competitors']}"
        )

        st.subheader("Closest Google Play apps")
        st.dataframe(
            result["neighbors"],
            use_container_width=True,
            hide_index=True,
        )

with st.expander("Model evaluation"):
    if Path(METRICS_JSON).exists():
        metrics = json.loads(
            Path(METRICS_JSON).read_text(encoding="utf-8")
        )
        st.json(metrics)
    else:
        st.write("No metrics file found.")
