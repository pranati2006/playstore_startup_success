from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from src.config import (
    CV_FOLD_RESULTS_CSV,
    FEATURE_IMPORTANCE_CSV,
    HYPERPARAMETER_RESULTS_CSV,
    METRICS_JSON,
    MODEL_COMPARISON_CSV,
    MODEL_PREDICTIONS_CSV,
    PREPARED_CSV,
)


st.set_page_config(
    page_title="Model Analytics Dashboard",
    page_icon="📊",
    layout="wide",
)


METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1 Score",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
    "balanced_accuracy": "Balanced Accuracy",
    "specificity": "Specificity",
}


CV_METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1 Score",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
    "balanced_accuracy": "Balanced Accuracy",
    "specificity": "Specificity",
}


def display_model_name(
    value: str,
) -> str:
    return (
        str(value)
        .replace(
            "_",
            " ",
        )
        .title()
    )


@st.cache_data
def load_csv(
    path,
) -> pd.DataFrame:
    return pd.read_csv(
        path
    )


@st.cache_data
def load_json(
    path,
) -> dict:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def required_files_exist() -> bool:
    required = [
        MODEL_COMPARISON_CSV,
        HYPERPARAMETER_RESULTS_CSV,
        CV_FOLD_RESULTS_CSV,
        MODEL_PREDICTIONS_CSV,
        FEATURE_IMPORTANCE_CSV,
        METRICS_JSON,
        PREPARED_CSV,
    ]

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:
        st.error(
            "The dashboard data has not been generated yet."
        )

        st.write(
            "Run the updated training step first. "
            "Missing files:"
        )

        for path in missing:
            st.code(
                str(path)
            )

        return False

    return True


def show_overview(
    comparison_df: pd.DataFrame,
    metrics: dict,
    prepared_df: pd.DataFrame,
) -> None:
    selected = metrics[
        "selected_model"
    ]

    test_metrics = selected[
        "final_test_metrics"
    ]

    best_name = display_model_name(
        selected[
            "name"
        ]
    )

    col1, col2, col3, col4, col5 = (
        st.columns(5)
    )

    col1.metric(
        "Selected Model",
        best_name,
    )

    col2.metric(
        "Final ROC-AUC",
        f"{test_metrics['roc_auc']:.3f}",
    )

    col3.metric(
        "Final F1",
        f"{test_metrics['f1']:.3f}",
    )

    col4.metric(
        "Final Recall",
        f"{test_metrics['recall']:.3f}",
    )

    col5.metric(
        "Prepared Apps",
        f"{len(prepared_df):,}",
    )

    st.caption(
        "Final metrics are from the untouched test set. "
        "The comparison charts below use the separate "
        "model-comparison set."
    )

    metric = st.selectbox(
        "Metric for model ranking",
        options=list(
            METRIC_LABELS.keys()
        ),
        format_func=lambda value: (
            METRIC_LABELS[
                value
            ]
        ),
        index=list(
            METRIC_LABELS.keys()
        ).index(
            "roc_auc"
        ),
        key="overview_metric",
    )

    chart_df = (
        comparison_df
        .copy()
        .sort_values(
            metric,
            ascending=True,
        )
    )

    chart_df[
        "display_model"
    ] = (
        chart_df[
            "model"
        ]
        .map(
            display_model_name
        )
    )

    fig = px.bar(
        chart_df,
        x=metric,
        y="display_model",
        orientation="h",
        text=metric,
        title=(
            "Comparative Performance - "
            f"{METRIC_LABELS[metric]}"
        ),
        labels={
            metric: (
                METRIC_LABELS[
                    metric
                ]
            ),
            "display_model": (
                "Model"
            ),
        },
    )

    fig.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside",
    )

    fig.update_xaxes(
        range=[0, 1.05]
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    with st.expander(
        "View complete model comparison table"
    ):
        display_df = (
            comparison_df
            .copy()
        )

        display_df[
            "model"
        ] = (
            display_df[
                "model"
            ]
            .map(
                display_model_name
            )
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )


def show_roc_pr(
    predictions_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
) -> None:
    models = (
        comparison_df[
            "model"
        ]
        .tolist()
    )

    selected_models = st.multiselect(
        "Models to display",
        options=models,
        default=models,
        format_func=(
            display_model_name
        ),
        key="curve_models",
    )

    if not selected_models:
        st.info(
            "Select at least one model."
        )
        return

    roc_fig = go.Figure()

    for model_name in (
        selected_models
    ):
        model_data = (
            predictions_df[
                predictions_df[
                    "model"
                ]
                == model_name
            ]
        )

        fpr, tpr, _ = roc_curve(
            model_data[
                "y_true"
            ],
            model_data[
                "probability"
            ],
        )

        auc = float(
            comparison_df.loc[
                comparison_df[
                    "model"
                ]
                == model_name,
                "roc_auc",
            ].iloc[0]
        )

        roc_fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=(
                    f"{display_model_name(model_name)} "
                    f"(AUC={auc:.3f})"
                ),
            )
        )

    roc_fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random classifier",
            line={
                "dash": "dash",
            },
        )
    )

    roc_fig.update_layout(
        title=(
            "Interactive ROC Curves"
        ),
        xaxis_title=(
            "False Positive Rate"
        ),
        yaxis_title=(
            "True Positive Rate"
        ),
        xaxis={
            "range": [0, 1],
        },
        yaxis={
            "range": [0, 1],
        },
    )

    st.plotly_chart(
        roc_fig,
        use_container_width=True,
    )

    pr_fig = go.Figure()

    for model_name in (
        selected_models
    ):
        model_data = (
            predictions_df[
                predictions_df[
                    "model"
                ]
                == model_name
            ]
        )

        precision, recall, _ = (
            precision_recall_curve(
                model_data[
                    "y_true"
                ],
                model_data[
                    "probability"
                ],
            )
        )

        pr_auc = float(
            comparison_df.loc[
                comparison_df[
                    "model"
                ]
                == model_name,
                "pr_auc",
            ].iloc[0]
        )

        pr_fig.add_trace(
            go.Scatter(
                x=recall,
                y=precision,
                mode="lines",
                name=(
                    f"{display_model_name(model_name)} "
                    f"(AP={pr_auc:.3f})"
                ),
            )
        )

    positive_rate = float(
        predictions_df[
            "y_true"
        ].mean()
    )

    pr_fig.add_hline(
        y=positive_rate,
        line_dash="dash",
        annotation_text=(
            "Positive-rate baseline"
        ),
    )

    pr_fig.update_layout(
        title=(
            "Interactive Precision-Recall Curves"
        ),
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis={
            "range": [0, 1],
        },
        yaxis={
            "range": [0, 1],
        },
    )

    st.plotly_chart(
        pr_fig,
        use_container_width=True,
    )


def show_confusion_matrix(
    predictions_df: pd.DataFrame,
) -> None:
    models = sorted(
        predictions_df[
            "model"
        ]
        .unique()
        .tolist()
    )

    model_name = st.selectbox(
        "Select model",
        options=models,
        format_func=(
            display_model_name
        ),
        key="confusion_model",
    )

    model_data = (
        predictions_df[
            predictions_df[
                "model"
            ]
            == model_name
        ]
    )

    cm = confusion_matrix(
        model_data[
            "y_true"
        ],
        model_data[
            "y_pred"
        ],
        labels=[0, 1],
    )

    fig = px.imshow(
        cm,
        text_auto=True,
        x=[
            "Predicted Not Successful",
            "Predicted Successful",
        ],
        y=[
            "Actual Not Successful",
            "Actual Successful",
        ],
        labels={
            "x": "Prediction",
            "y": "Actual",
            "color": "Count",
        },
        title=(
            f"{display_model_name(model_name)} "
            "- Confusion Matrix"
        ),
        aspect="auto",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    tn, fp, fn, tp = (
        cm.ravel()
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    c1.metric(
        "True Negative",
        int(tn),
    )
    c2.metric(
        "False Positive",
        int(fp),
    )
    c3.metric(
        "False Negative",
        int(fn),
    )
    c4.metric(
        "True Positive",
        int(tp),
    )


def show_feature_importance(
    importance_df: pd.DataFrame,
) -> None:
    models = sorted(
        importance_df[
            "model"
        ]
        .unique()
        .tolist()
    )

    col1, col2 = (
        st.columns(2)
    )

    model_name = col1.selectbox(
        "Select model",
        options=models,
        format_func=(
            display_model_name
        ),
        key="importance_model",
    )

    top_n = col2.selectbox(
        "Number of features",
        options=[
            10,
            15,
            20,
            25,
        ],
        index=2,
    )

    model_df = (
        importance_df[
            importance_df[
                "model"
            ]
            == model_name
        ]
        .sort_values(
            "absolute_importance",
            ascending=False,
        )
        .head(
            top_n
        )
        .sort_values(
            "absolute_importance",
            ascending=True,
        )
    )

    method = (
        model_df[
            "method"
        ]
        .iloc[0]
        if not model_df.empty
        else ""
    )

    fig = px.bar(
        model_df,
        x="absolute_importance",
        y="feature",
        orientation="h",
        hover_data=[
            "importance",
            "method",
        ],
        title=(
            f"{display_model_name(model_name)} "
            f"- Top {len(model_df)} Features"
        ),
        labels={
            "absolute_importance": (
                "Importance"
            ),
            "feature": (
                "Feature"
            ),
        },
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.caption(
        f"Importance method: {method}. "
        "Importance indicates predictive usefulness, "
        "not causation."
    )


def show_hyperparameter_analysis(
    hyper_df: pd.DataFrame,
) -> None:
    models = sorted(
        hyper_df[
            "model"
        ]
        .unique()
        .tolist()
    )

    col1, col2 = (
        st.columns(2)
    )

    model_name = col1.selectbox(
        "Select model",
        options=models,
        format_func=(
            display_model_name
        ),
        key="hyper_model",
    )

    metric = col2.selectbox(
        "Cross-validation metric",
        options=[
            "cv_roc_auc_mean",
            "cv_f1_mean",
            "cv_precision_mean",
            "cv_recall_mean",
            "cv_accuracy_mean",
            "cv_pr_auc_mean",
        ],
        format_func=lambda value: (
            value
            .replace(
                "cv_",
                "",
            )
            .replace(
                "_mean",
                "",
            )
            .replace(
                "_",
                " ",
            )
            .title()
        ),
    )

    model_df = (
        hyper_df[
            hyper_df[
                "model"
            ]
            == model_name
        ]
        .copy()
        .sort_values(
            "config_id"
        )
    )

    model_df[
        "config_label"
    ] = (
        "Config "
        + model_df[
            "config_id"
        ]
        .astype(str)
    )

    fig = px.bar(
        model_df,
        x="config_label",
        y=metric,
        text=metric,
        hover_data=[
            "parameters",
        ],
        title=(
            f"{display_model_name(model_name)} "
            "- Hyperparameter Comparison"
        ),
        labels={
            metric: (
                "Cross-validation score"
            ),
            "config_label": (
                "Configuration"
            ),
        },
    )

    fig.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside",
    )

    fig.update_yaxes(
        range=[0, 1.05]
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.dataframe(
        model_df[
            [
                "config_id",
                "parameters",
                "cv_accuracy_mean",
                "cv_precision_mean",
                "cv_recall_mean",
                "cv_f1_mean",
                "cv_roc_auc_mean",
                "cv_pr_auc_mean",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


def show_cv_stability(
    cv_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
) -> None:
    best_configs = (
        comparison_df[
            [
                "model",
                "config_id",
            ]
        ]
        .copy()
    )

    best_cv = (
        cv_df.merge(
            best_configs,
            on=[
                "model",
                "config_id",
            ],
            how="inner",
        )
    )

    metric = st.selectbox(
        "Cross-validation stability metric",
        options=list(
            CV_METRIC_LABELS.keys()
        ),
        format_func=lambda value: (
            CV_METRIC_LABELS[
                value
            ]
        ),
        index=list(
            CV_METRIC_LABELS.keys()
        ).index(
            "roc_auc"
        ),
    )

    best_cv[
        "display_model"
    ] = (
        best_cv[
            "model"
        ]
        .map(
            display_model_name
        )
    )

    fig = px.box(
        best_cv,
        x="display_model",
        y=metric,
        points="all",
        title=(
            "Cross-Validation Stability - "
            f"{CV_METRIC_LABELS[metric]}"
        ),
        labels={
            "display_model": (
                "Model"
            ),
            metric: (
                CV_METRIC_LABELS[
                    metric
                ]
            ),
        },
    )

    fig.update_yaxes(
        range=[0, 1.05]
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    summary = (
        best_cv
        .groupby(
            "display_model"
        )[metric]
        .agg(
            [
                "mean",
                "std",
                "min",
                "max",
            ]
        )
        .sort_values(
            "mean",
            ascending=False,
        )
        .reset_index()
    )

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True,
    )


def bool_to_label(
    series: pd.Series,
) -> pd.Series:
    if series.dtype == bool:
        return series.map(
            {
                True: "Yes",
                False: "No",
            }
        )

    return (
        series
        .fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(
            [
                "true",
                "1",
                "yes",
                "y",
                "t",
            ]
        )
        .map(
            {
                True: "Yes",
                False: "No",
            }
        )
    )


def show_dataset_explorer(
    prepared_df: pd.DataFrame,
) -> None:
    working = (
        prepared_df
        .copy()
    )

    genres = [
        "All",
        *sorted(
            working[
                "genre"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
    ]

    col1, col2, col3 = (
        st.columns(3)
    )

    genre = col1.selectbox(
        "Genre",
        options=genres,
    )

    success_filter = (
        col2.selectbox(
            "Success class",
            options=[
                "All",
                "Successful",
                "Not Successful",
            ],
        )
    )

    free_filter = (
        col3.selectbox(
            "Free/Paid",
            options=[
                "All",
                "Free",
                "Paid",
            ],
        )
    )

    if genre != "All":
        working = working[
            working[
                "genre"
            ]
            .astype(str)
            == genre
        ]

    if (
        success_filter
        == "Successful"
    ):
        working = working[
            working[
                "success"
            ]
            .astype(int)
            == 1
        ]
    elif (
        success_filter
        == "Not Successful"
    ):
        working = working[
            working[
                "success"
            ]
            .astype(int)
            == 0
        ]

    if "free" in working.columns:
        free_values = (
            bool_to_label(
                working[
                    "free"
                ]
            )
        )

        if free_filter == "Free":
            working = working[
                free_values
                == "Yes"
            ]
        elif free_filter == "Paid":
            working = working[
                free_values
                == "No"
            ]

    st.metric(
        "Apps after filters",
        f"{len(working):,}",
    )

    if working.empty:
        st.info(
            "No apps match the selected filters."
        )
        return

    success_counts = (
        working[
            "success"
        ]
        .astype(int)
        .map(
            {
                0: "Not Successful",
                1: "Successful",
            }
        )
        .value_counts()
        .rename_axis(
            "class"
        )
        .reset_index(
            name="count"
        )
    )

    fig_success = px.bar(
        success_counts,
        x="class",
        y="count",
        text="count",
        title=(
            "Success-Class Distribution"
        ),
    )

    st.plotly_chart(
        fig_success,
        use_container_width=True,
    )

    top_genres = (
        working[
            "genre"
        ]
        .fillna(
            "Unknown"
        )
        .astype(str)
        .value_counts()
        .head(20)
        .rename_axis(
            "genre"
        )
        .reset_index(
            name="count"
        )
        .sort_values(
            "count",
            ascending=True,
        )
    )

    fig_genre = px.bar(
        top_genres,
        x="count",
        y="genre",
        orientation="h",
        title=(
            "Top Genres"
        ),
    )

    st.plotly_chart(
        fig_genre,
        use_container_width=True,
    )

    left, right = (
        st.columns(2)
    )

    if "score" in working.columns:
        score_values = (
            pd.to_numeric(
                working[
                    "score"
                ],
                errors="coerce",
            )
            .dropna()
        )

        fig_rating = px.histogram(
            x=score_values,
            nbins=30,
            labels={
                "x": "Rating",
                "y": "Apps",
            },
            title=(
                "Rating Distribution"
            ),
        )

        left.plotly_chart(
            fig_rating,
            use_container_width=True,
        )

    if (
        "realInstalls"
        in working.columns
    ):
        install_values = (
            pd.to_numeric(
                working[
                    "realInstalls"
                ],
                errors="coerce",
            )
            .fillna(0)
        )

        log_installs = (
            np.log1p(
                install_values
            )
        )

        fig_installs = px.histogram(
            x=log_installs,
            nbins=40,
            labels={
                "x": (
                    "log(1 + real installs)"
                ),
                "y": "Apps",
            },
            title=(
                "Install Distribution "
                "(Log Scale)"
            ),
        )

        right.plotly_chart(
            fig_installs,
            use_container_width=True,
        )


def main() -> None:
    st.title(
        "Model Analytics Dashboard"
    )

    st.write(
        "Interactive analysis of the ten machine-learning "
        "algorithms, their hyperparameter experiments, "
        "cross-validation stability, error patterns, "
        "feature importance and dataset characteristics."
    )

    if not required_files_exist():
        st.stop()

    comparison_df = load_csv(
        MODEL_COMPARISON_CSV
    )

    hyper_df = load_csv(
        HYPERPARAMETER_RESULTS_CSV
    )

    cv_df = load_csv(
        CV_FOLD_RESULTS_CSV
    )

    predictions_df = load_csv(
        MODEL_PREDICTIONS_CSV
    )

    importance_df = load_csv(
        FEATURE_IMPORTANCE_CSV
    )

    prepared_df = load_csv(
        PREPARED_CSV
    )

    metrics = load_json(
        METRICS_JSON
    )

    (
        tab_overview,
        tab_curves,
        tab_confusion,
        tab_importance,
        tab_hyper,
        tab_cv,
        tab_dataset,
    ) = st.tabs(
        [
            "Overview",
            "ROC & Precision-Recall",
            "Confusion Matrix",
            "Feature Importance",
            "Hyperparameters",
            "CV Stability",
            "Dataset Explorer",
        ]
    )

    with tab_overview:
        show_overview(
            comparison_df=(
                comparison_df
            ),
            metrics=metrics,
            prepared_df=(
                prepared_df
            ),
        )

    with tab_curves:
        show_roc_pr(
            predictions_df=(
                predictions_df
            ),
            comparison_df=(
                comparison_df
            ),
        )

    with tab_confusion:
        show_confusion_matrix(
            predictions_df=(
                predictions_df
            )
        )

    with tab_importance:
        show_feature_importance(
            importance_df=(
                importance_df
            )
        )

    with tab_hyper:
        show_hyperparameter_analysis(
            hyper_df=(
                hyper_df
            )
        )

    with tab_cv:
        show_cv_stability(
            cv_df=(
                cv_df
            ),
            comparison_df=(
                comparison_df
            ),
        )

    with tab_dataset:
        show_dataset_explorer(
            prepared_df=(
                prepared_df
            )
        )


if __name__ == "__main__":
    main()
