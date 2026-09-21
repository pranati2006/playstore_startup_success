from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    DEFAULT_TOP_K,
    EMBEDDING_MODEL,
    FEATURES_CSV,
    METRICS_JSON,
    MODEL_BUNDLE,
    RANDOM_STATE,
)
from .features import MARKET_COLUMNS, market_features_against_reference


BASE_NUMERIC_COLUMNS = [
    "price",
    "free",
    "offersIAP",
    "containsAds",
    "description_chars",
    "description_words",
    "summary_chars",
    *MARKET_COLUMNS,
]

CATEGORICAL_COLUMNS = [
    "genre",
    "contentRating",
]


def make_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, numeric_columns),
            ("categorical", categorical_pipe, categorical_columns),
        ],
        remainder="drop",
    )


def make_pipeline(
    estimator,
    numeric_columns: list[str],
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "preprocessor",
                make_preprocessor(
                    numeric_columns,
                    CATEGORICAL_COLUMNS,
                ),
            ),
            ("classifier", estimator),
        ]
    )


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(
            precision_score(y_test, pred, zero_division=0)
        ),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }


def attach_market_features(
    base_df: pd.DataFrame,
    market_df: pd.DataFrame,
) -> pd.DataFrame:
    output = base_df.copy().reset_index(drop=True)
    for column in MARKET_COLUMNS:
        if column in output.columns:
            output = output.drop(columns=[column])

    return pd.concat(
        [output, market_df.reset_index(drop=True)],
        axis=1,
    )


def train_models(
    feature_path: Path,
    model_output: Path,
    metrics_output: Path,
    top_k: int = DEFAULT_TOP_K,
) -> tuple[dict, dict]:
    df = pd.read_csv(feature_path)

    if len(df) < 100:
        raise ValueError(
            "Use at least 100 apps for a meaningful demo. "
            "A few thousand is strongly recommended."
        )

    embedding_columns = sorted(
        column for column in df.columns if column.startswith("emb_")
    )
    if not embedding_columns:
        raise ValueError("No embedding columns found.")

    numeric_columns = BASE_NUMERIC_COLUMNS + embedding_columns

    needed = {
        "appId",
        "success",
        *embedding_columns,
        *CATEGORICAL_COLUMNS,
        "realInstalls",
        "reviews",
        "score",
        "free",
        "offersIAP",
        "containsAds",
        "price",
        "description_chars",
        "description_words",
        "summary_chars",
    }
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing model columns: {sorted(missing)}")

    y = df["success"].astype(int)
    all_indices = np.arange(len(df))

    train_idx, test_idx = train_test_split(
        all_indices,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    train_embeddings = train_df[embedding_columns].to_numpy(np.float32)
    test_embeddings = test_df[embedding_columns].to_numpy(np.float32)

    # Leakage-safe evaluation:
    # - train apps use only other train apps as market references
    # - test apps use only train apps as market references
    train_market, _, _ = market_features_against_reference(
        query_embeddings=train_embeddings,
        reference_df=train_df,
        reference_embeddings=train_embeddings,
        top_k=top_k,
        query_app_ids=train_df["appId"].astype(str).tolist(),
    )
    test_market, _, _ = market_features_against_reference(
        query_embeddings=test_embeddings,
        reference_df=train_df,
        reference_embeddings=train_embeddings,
        top_k=top_k,
        query_app_ids=None,
    )

    train_features = attach_market_features(train_df, train_market)
    test_features = attach_market_features(test_df, test_market)

    model_columns = numeric_columns + CATEGORICAL_COLUMNS
    X_train = train_features[model_columns]
    y_train = train_features["success"].astype(int)
    X_test = test_features[model_columns]
    y_test = test_features["success"].astype(int)

    estimators = {
        "logistic_regression": LogisticRegression(
            max_iter=2500,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=350,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=350,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }

    metrics: dict[str, dict] = {}

    for name, estimator in estimators.items():
        print(f"Training {name}...")
        model = make_pipeline(
            estimator=clone(estimator),
            numeric_columns=numeric_columns,
        )
        model.fit(X_train, y_train)
        metrics[name] = evaluate_model(model, X_test, y_test)

    best_name = max(
        metrics,
        key=lambda name: (
            metrics[name]["roc_auc"],
            metrics[name]["f1"],
        ),
    )

    # Retrain the chosen model on all historical apps.
    all_embeddings = df[embedding_columns].to_numpy(np.float32)
    full_market, _, _ = market_features_against_reference(
        query_embeddings=all_embeddings,
        reference_df=df,
        reference_embeddings=all_embeddings,
        top_k=top_k,
        query_app_ids=df["appId"].astype(str).tolist(),
    )
    full_features = attach_market_features(df, full_market)
    X_full = full_features[model_columns]
    y_full = full_features["success"].astype(int)

    final_model = make_pipeline(
        estimator=clone(estimators[best_name]),
        numeric_columns=numeric_columns,
    )
    final_model.fit(X_full, y_full)

    metrics["selected_model"] = {
        "name": best_name,
        "selection_metric": "roc_auc_then_f1",
        "test_rows": int(len(X_test)),
        "train_rows": int(len(X_train)),
        "full_rows_for_final_fit": int(len(X_full)),
        "positive_rate_full": float(y_full.mean()),
        "top_k": int(top_k),
        "evaluation_market_reference": "training_pool_only",
    }

    bundle = {
        "model": final_model,
        "model_name": best_name,
        "numeric_columns": numeric_columns,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "embedding_columns": embedding_columns,
        "embedding_model": EMBEDDING_MODEL,
        "top_k": int(top_k),
    }

    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_output)

    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print(f"Selected model: {best_name}")
    print(json.dumps(metrics[best_name], indent=2))

    return bundle, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=str(FEATURES_CSV))
    parser.add_argument("--model-output", default=str(MODEL_BUNDLE))
    parser.add_argument("--metrics-output", default=str(METRICS_JSON))
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    train_models(
        feature_path=Path(args.features),
        model_output=Path(args.model_output),
        metrics_output=Path(args.metrics_output),
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
