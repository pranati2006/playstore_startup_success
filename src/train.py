from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .config import (
    COMPARISON_CHECKPOINT_DIR,
    CONFUSION_MATRIX_DIR,
    CV_CACHE_DIR,
    CV_FOLD_RESULTS_CSV,
    DEFAULT_CV_FOLDS,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL,
    EVALUATION_DIR,
    FEATURE_CACHE_DIR,
    FEATURE_IMPORTANCE_CSV,
    FEATURE_IMPORTANCE_DIR,
    FEATURES_CSV,
    HYPERPARAMETER_RESULTS_CSV,
    METRICS_JSON,
    MODEL_BUNDLE,
    MODEL_COMPARISON_CSV,
    MODEL_COMPARISON_PNG,
    MODEL_EVALUATION_REPORT,
    MODEL_PREDICTIONS_CSV,
    FINAL_MODEL_DONE_JSON,
    FINAL_TEST_CHECKPOINT,
    PR_CURVES_PNG,
    RANDOM_STATE,
    ROC_CURVES_PNG,
    SEARCH_CHECKPOINT_DIR,
    SPLIT_INDICES_NPZ,
    TRAINING_CHECKPOINT_DIR,
    TRAINING_STATE_JSON,
)
from .evaluation import (
    calculate_metrics,
    ensure_evaluation_dirs,
    feature_importance_frame,
    generate_html_report,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_model_comparison,
    plot_precision_recall_curves,
    plot_roc_curves,
    save_results_csv,
)
from .features import (
    MARKET_COLUMNS,
    market_features_against_reference,
)


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


CHECKPOINT_SCHEMA_VERSION = 1


def atomic_write_json(
    path: Path,
    data: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        path.name + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            data,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    temporary.replace(
        path
    )


def atomic_joblib_dump(
    value,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        path.name + ".tmp"
    )

    joblib.dump(
        value,
        temporary,
    )

    temporary.replace(
        path
    )


def safe_load_json(
    path: Path,
):
    if not path.exists():
        return None

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        print(
            "Ignoring unreadable checkpoint: "
            f"{path}"
        )

        try:
            path.unlink()
        except OSError:
            pass

        return None


def safe_joblib_load(
    path: Path,
):
    if not path.exists():
        return None

    try:
        return joblib.load(
            path
        )
    except Exception:
        print(
            "Ignoring unreadable checkpoint: "
            f"{path}"
        )

        try:
            path.unlink()
        except OSError:
            pass

        return None


def file_sha256(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(
                chunk_size
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def estimator_signature(
    estimator,
) -> dict:
    output = {}

    for key, value in sorted(
        estimator
        .get_params(
            deep=True
        )
        .items()
    ):
        if isinstance(
            value,
            (
                str,
                int,
                float,
                bool,
                type(None),
            ),
        ):
            output[key] = value
        else:
            output[key] = repr(
                value
            )

    return output


def model_configs_signature(
    model_configs: dict[str, list],
) -> dict:
    return {
        model_name: [
            estimator_signature(
                estimator
            )
            for estimator
            in estimators
        ]
        for model_name, estimators
        in model_configs.items()
    }


def build_training_state(
    feature_path: Path,
    df: pd.DataFrame,
    embedding_columns: list[str],
    top_k: int,
    cv_folds: int,
    model_configs: dict[str, list],
) -> dict:
    resolved_path = (
        feature_path
        .resolve()
    )

    return {
        "schema_version": (
            CHECKPOINT_SCHEMA_VERSION
        ),
        "feature_path": str(
            resolved_path
        ),
        "feature_sha256": (
            file_sha256(
                resolved_path
            )
        ),
        "feature_size_bytes": int(
            resolved_path
            .stat()
            .st_size
        ),
        "dataset_rows": int(
            len(df)
        ),
        "embedding_columns": int(
            len(
                embedding_columns
            )
        ),
        "top_k": int(
            top_k
        ),
        "cv_folds": int(
            cv_folds
        ),
        "random_state": int(
            RANDOM_STATE
        ),
        "model_configs": (
            model_configs_signature(
                model_configs
            )
        ),
    }


def ensure_training_state(
    current_state: dict,
) -> None:
    previous_state = safe_load_json(
        TRAINING_STATE_JSON
    )

    if previous_state == current_state:
        print(
            "\nCompatible training checkpoints found. "
            "Resuming completed work where possible."
        )
        return

    if TRAINING_CHECKPOINT_DIR.exists():
        if previous_state is None:
            print(
                "\nNo valid training-state checkpoint was found. "
                "Starting a fresh training run."
            )
        else:
            print(
                "\nDataset or training settings changed. "
                "Old training checkpoints will not be reused."
            )

        shutil.rmtree(
            TRAINING_CHECKPOINT_DIR,
            ignore_errors=True,
        )

    TRAINING_CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    atomic_write_json(
        TRAINING_STATE_JSON,
        current_state,
    )


def load_or_create_split_indices(
    y: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    if SPLIT_INDICES_NPZ.exists():
        try:
            with np.load(
                SPLIT_INDICES_NPZ
            ) as data:
                development_idx = (
                    data[
                        "development_idx"
                    ]
                    .astype(int)
                )
                comparison_idx = (
                    data[
                        "comparison_idx"
                    ]
                    .astype(int)
                )
                final_test_idx = (
                    data[
                        "final_test_idx"
                    ]
                    .astype(int)
                )

            combined = np.concatenate(
                [
                    development_idx,
                    comparison_idx,
                    final_test_idx,
                ]
            )

            valid = (
                len(combined) == len(y)
                and len(
                    np.unique(
                        combined
                    )
                ) == len(y)
                and combined.min(
                    initial=0
                ) >= 0
                and combined.max(
                    initial=-1
                ) < len(y)
            )

            if valid:
                print(
                    "Using cached train/comparison/test split."
                )

                return (
                    development_idx,
                    comparison_idx,
                    final_test_idx,
                )
        except Exception:
            print(
                "Cached split indices are unreadable. "
                "They will be recreated."
            )

    all_indices = np.arange(
        len(y)
    )

    selection_idx, final_test_idx = (
        train_test_split(
            all_indices,
            test_size=0.15,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    selection_y = y[
        selection_idx
    ]

    comparison_fraction = (
        0.15 / 0.85
    )

    development_idx, comparison_idx = (
        train_test_split(
            selection_idx,
            test_size=(
                comparison_fraction
            ),
            random_state=RANDOM_STATE,
            stratify=selection_y,
        )
    )

    SPLIT_INDICES_NPZ.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = SPLIT_INDICES_NPZ.with_name(
        SPLIT_INDICES_NPZ.name + ".tmp"
    )

    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            development_idx=(
                development_idx
            ),
            comparison_idx=(
                comparison_idx
            ),
            final_test_idx=(
                final_test_idx
            ),
        )

    temporary.replace(
        SPLIT_INDICES_NPZ
    )

    print(
        "Created and cached train/comparison/test split."
    )

    return (
        development_idx,
        comparison_idx,
        final_test_idx,
    )


def load_cached_feature_set(
    path: Path,
    required_keys: set[str],
):
    cached = safe_joblib_load(
        path
    )

    if not isinstance(
        cached,
        dict,
    ):
        return None

    if not required_keys.issubset(
        cached.keys()
    ):
        return None

    return cached


def make_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
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
            (
                "numeric",
                numeric_pipe,
                numeric_columns,
            ),
            (
                "categorical",
                categorical_pipe,
                categorical_columns,
            ),
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
            (
                "classifier",
                estimator,
            ),
        ]
    )


def evaluate_model(
    model: Pipeline,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
) -> tuple[
    dict,
    np.ndarray,
    np.ndarray,
]:
    pred = model.predict(
        X_eval
    )

    proba = (
        model.predict_proba(
            X_eval
        )[:, 1]
    )

    metrics = calculate_metrics(
        y_true=y_eval,
        y_pred=pred,
        y_proba=proba,
    )

    return (
        metrics,
        pred,
        proba,
    )


def attach_market_features(
    base_df: pd.DataFrame,
    market_df: pd.DataFrame,
) -> pd.DataFrame:
    output = (
        base_df
        .copy()
        .reset_index(
            drop=True
        )
    )

    for column in MARKET_COLUMNS:
        if column in output.columns:
            output = output.drop(
                columns=[column]
            )

    return pd.concat(
        [
            output,
            market_df.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )


def get_model_configs() -> dict[str, list]:
    return {
        "logistic_regression": [
            LogisticRegression(
                C=0.5,
                max_iter=3000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            LogisticRegression(
                C=1.0,
                max_iter=3000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            LogisticRegression(
                C=2.0,
                max_iter=3000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ],

        "decision_tree": [
            DecisionTreeClassifier(
                max_depth=6,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            DecisionTreeClassifier(
                max_depth=10,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            DecisionTreeClassifier(
                max_depth=None,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ],

        "random_forest": [
            RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=1,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            RandomForestClassifier(
                n_estimators=350,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            RandomForestClassifier(
                n_estimators=500,
                max_depth=20,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        ],

        "extra_trees": [
            ExtraTreesClassifier(
                n_estimators=200,
                min_samples_leaf=1,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            ExtraTreesClassifier(
                n_estimators=350,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            ExtraTreesClassifier(
                n_estimators=500,
                max_depth=20,
                min_samples_leaf=3,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        ],

        "gradient_boosting": [
            GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=2,
                random_state=RANDOM_STATE,
            ),
            GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=3,
                random_state=RANDOM_STATE,
            ),
            GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.03,
                max_depth=3,
                random_state=RANDOM_STATE,
            ),
        ],

        "hist_gradient_boosting": [
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=150,
                max_leaf_nodes=15,
                random_state=RANDOM_STATE,
            ),
            HistGradientBoostingClassifier(
                learning_rate=0.08,
                max_iter=200,
                max_leaf_nodes=31,
                random_state=RANDOM_STATE,
            ),
            HistGradientBoostingClassifier(
                learning_rate=0.03,
                max_iter=300,
                max_leaf_nodes=31,
                random_state=RANDOM_STATE,
            ),
        ],

        "adaboost": [
            AdaBoostClassifier(
                n_estimators=100,
                learning_rate=0.5,
                random_state=RANDOM_STATE,
            ),
            AdaBoostClassifier(
                n_estimators=200,
                learning_rate=0.5,
                random_state=RANDOM_STATE,
            ),
            AdaBoostClassifier(
                n_estimators=300,
                learning_rate=0.3,
                random_state=RANDOM_STATE,
            ),
        ],

        "svc": [
            CalibratedClassifierCV(
                estimator=SVC(
                    C=0.5,
                    kernel="rbf",
                    gamma="scale",
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
                method="sigmoid",
                cv=3,
                ensemble=False,
            ),
            CalibratedClassifierCV(
                estimator=SVC(
                    C=1.0,
                    kernel="rbf",
                    gamma="scale",
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
                method="sigmoid",
                cv=3,
                ensemble=False,
            ),
            CalibratedClassifierCV(
                estimator=SVC(
                    C=2.0,
                    kernel="rbf",
                    gamma="scale",
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
                method="sigmoid",
                cv=3,
                ensemble=False,
            ),
        ],

        "knn": [
            KNeighborsClassifier(
                n_neighbors=7,
                weights="distance",
                p=2,
                n_jobs=-1,
            ),
            KNeighborsClassifier(
                n_neighbors=15,
                weights="distance",
                p=2,
                n_jobs=-1,
            ),
            KNeighborsClassifier(
                n_neighbors=25,
                weights="distance",
                p=2,
                n_jobs=-1,
            ),
        ],

        "gaussian_nb": [
            GaussianNB(
                var_smoothing=1e-9,
            ),
            GaussianNB(
                var_smoothing=1e-8,
            ),
            GaussianNB(
                var_smoothing=1e-7,
            ),
        ],
    }


def json_safe_params(
    estimator,
) -> dict:
    output = {}

    for key, value in (
        estimator
        .get_params(
            deep=False
        )
        .items()
    ):
        if isinstance(
            value,
            (
                str,
                int,
                float,
                bool,
                type(None),
            ),
        ):
            output[key] = value
        else:
            output[key] = str(
                value
            )

    return output


def build_leakage_safe_features(
    query_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    embedding_columns: list[str],
    top_k: int,
    exclude_self: bool,
) -> pd.DataFrame:
    query_embeddings = (
        query_df[
            embedding_columns
        ]
        .to_numpy(
            np.float32
        )
    )

    reference_embeddings = (
        reference_df[
            embedding_columns
        ]
        .to_numpy(
            np.float32
        )
    )

    query_ids = (
        query_df["appId"]
        .astype(str)
        .tolist()
        if exclude_self
        else None
    )

    market, _, _ = (
        market_features_against_reference(
            query_embeddings=query_embeddings,
            reference_df=reference_df,
            reference_embeddings=reference_embeddings,
            top_k=top_k,
            query_app_ids=query_ids,
        )
    )

    return attach_market_features(
        query_df,
        market,
    )


def prepare_cv_folds(
    development_df: pd.DataFrame,
    embedding_columns: list[str],
    model_columns: list[str],
    top_k: int,
    cv_folds: int,
) -> list[dict]:
    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    y = (
        development_df[
            "success"
        ]
        .astype(int)
        .to_numpy()
    )

    fold_data = []

    CV_CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for fold_number, (
        train_idx,
        val_idx,
    ) in enumerate(
        cv.split(
            development_df,
            y,
        ),
        start=1,
    ):
        cache_path = (
            CV_CACHE_DIR
            / f"fold_{fold_number}.joblib"
        )

        cached = safe_joblib_load(
            cache_path
        )

        if isinstance(
            cached,
            dict,
        ) and {
            "fold",
            "X_train",
            "y_train",
            "X_val",
            "y_val",
        }.issubset(
            cached.keys()
        ):
            print(
                f"  CV fold {fold_number}/{cv_folds}: "
                "loaded from checkpoint"
            )

            fold_data.append(
                cached
            )
            continue

        print(
            f"  CV fold {fold_number}/{cv_folds}: "
            "building leakage-safe features"
        )

        fold_train_df = (
            development_df
            .iloc[
                train_idx
            ]
            .reset_index(
                drop=True
            )
        )

        fold_val_df = (
            development_df
            .iloc[
                val_idx
            ]
            .reset_index(
                drop=True
            )
        )

        fold_train_features = (
            build_leakage_safe_features(
                query_df=fold_train_df,
                reference_df=fold_train_df,
                embedding_columns=embedding_columns,
                top_k=top_k,
                exclude_self=True,
            )
        )

        fold_val_features = (
            build_leakage_safe_features(
                query_df=fold_val_df,
                reference_df=fold_train_df,
                embedding_columns=embedding_columns,
                top_k=top_k,
                exclude_self=False,
            )
        )

        fold = {
            "fold": fold_number,
            "X_train": (
                fold_train_features[
                    model_columns
                ]
            ),
            "y_train": (
                fold_train_features[
                    "success"
                ]
                .astype(int)
            ),
            "X_val": (
                fold_val_features[
                    model_columns
                ]
            ),
            "y_val": (
                fold_val_features[
                    "success"
                ]
                .astype(int)
            ),
        }

        atomic_joblib_dump(
            fold,
            cache_path,
        )

        fold_data.append(
            fold
        )

    return fold_data


def summarize_cv_metrics(
    fold_metrics: list[dict],
) -> dict:
    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "balanced_accuracy",
        "specificity",
    ]

    summary = {}

    for metric in metric_names:
        values = np.asarray(
            [
                row[metric]
                for row
                in fold_metrics
            ],
            dtype=float,
        )

        summary[
            f"cv_{metric}_mean"
        ] = float(
            values.mean()
        )

        summary[
            f"cv_{metric}_std"
        ] = float(
            values.std(
                ddof=0
            )
        )

    return summary


def run_hyperparameter_search(
    model_configs: dict[str, list],
    fold_data: list[dict],
    numeric_columns: list[str],
) -> tuple[
    list[dict],
    dict[str, dict],
    list[dict],
]:
    results = []
    best_configs = {}
    fold_result_rows = []

    SEARCH_CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for model_name, estimators in (
        model_configs.items()
    ):
        print(
            "\nHyperparameter search: "
            f"{model_name}"
        )

        model_rows = []

        for config_index, estimator in enumerate(
            estimators,
            start=1,
        ):
            print(
                f"  Config {config_index}/"
                f"{len(estimators)}"
            )

            fold_metrics = []

            for fold in fold_data:
                fold_number = int(
                    fold[
                        "fold"
                    ]
                )

                checkpoint_path = (
                    SEARCH_CHECKPOINT_DIR
                    / model_name
                    / f"config_{config_index}"
                    / f"fold_{fold_number}.json"
                )

                checkpoint = safe_load_json(
                    checkpoint_path
                )

                checkpoint_valid = (
                    isinstance(
                        checkpoint,
                        dict,
                    )
                    and checkpoint.get(
                        "model"
                    ) == model_name
                    and int(
                        checkpoint.get(
                            "config_id",
                            -1,
                        )
                    ) == config_index
                    and int(
                        checkpoint.get(
                            "fold",
                            -1,
                        )
                    ) == fold_number
                    and isinstance(
                        checkpoint.get(
                            "metrics"
                        ),
                        dict,
                    )
                )

                if checkpoint_valid:
                    metrics = checkpoint[
                        "metrics"
                    ]

                    print(
                        f"    Fold {fold_number}: "
                        "checkpoint loaded"
                    )
                else:
                    print(
                        f"    Fold {fold_number}: training"
                    )

                    model = make_pipeline(
                        estimator=clone(
                            estimator
                        ),
                        numeric_columns=(
                            numeric_columns
                        ),
                    )

                    model.fit(
                        fold[
                            "X_train"
                        ],
                        fold[
                            "y_train"
                        ],
                    )

                    metrics, _, _ = (
                        evaluate_model(
                            model,
                            fold[
                                "X_val"
                            ],
                            fold[
                                "y_val"
                            ],
                        )
                    )

                    atomic_write_json(
                        checkpoint_path,
                        {
                            "model": (
                                model_name
                            ),
                            "config_id": (
                                config_index
                            ),
                            "fold": (
                                fold_number
                            ),
                            "metrics": (
                                metrics
                            ),
                        },
                    )

                fold_metrics.append(
                    metrics
                )

                fold_result_rows.append(
                    {
                        "model": model_name,
                        "config_id": (
                            config_index
                        ),
                        "fold": fold_number,
                        "accuracy": metrics[
                            "accuracy"
                        ],
                        "precision": metrics[
                            "precision"
                        ],
                        "recall": metrics[
                            "recall"
                        ],
                        "f1": metrics[
                            "f1"
                        ],
                        "roc_auc": metrics[
                            "roc_auc"
                        ],
                        "pr_auc": metrics[
                            "pr_auc"
                        ],
                        "balanced_accuracy": (
                            metrics[
                                "balanced_accuracy"
                            ]
                        ),
                        "specificity": (
                            metrics[
                                "specificity"
                            ]
                        ),
                    }
                )

            summary = (
                summarize_cv_metrics(
                    fold_metrics
                )
            )

            params = (
                json_safe_params(
                    estimator
                )
            )

            row = {
                "model": model_name,
                "config_id": (
                    config_index
                ),
                "parameters": (
                    json.dumps(
                        params,
                        sort_keys=True,
                    )
                ),
                **summary,
            }

            results.append(
                row
            )
            model_rows.append(
                row
            )

        best_row = max(
            model_rows,
            key=lambda row: (
                row[
                    "cv_roc_auc_mean"
                ],
                row[
                    "cv_f1_mean"
                ],
            ),
        )

        best_config_id = int(
            best_row[
                "config_id"
            ]
        )

        best_configs[
            model_name
        ] = {
            "config_id": (
                best_config_id
            ),
            "estimator": clone(
                estimators[
                    best_config_id - 1
                ]
            ),
            "parameters": (
                best_row[
                    "parameters"
                ]
            ),
            "cv_metrics": {
                key: value
                for key, value
                in best_row.items()
                if key.startswith(
                    "cv_"
                )
            },
        }

    return (
        results,
        best_configs,
        fold_result_rows,
    )


def plot_single_roc(
    y_true: pd.Series,
    y_proba: np.ndarray,
    model_name: str,
    output_path: Path,
) -> None:
    fpr, tpr, _ = roc_curve(
        y_true,
        y_proba,
    )

    auc = roc_auc_score(
        y_true,
        y_proba,
    )

    plt.figure(
        figsize=(7, 6)
    )

    plt.plot(
        fpr,
        tpr,
        label=(
            f"ROC-AUC={auc:.3f}"
        ),
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random classifier",
    )

    plt.title(
        f"{model_name} - "
        "Final Test ROC Curve"
    )
    plt.xlabel(
        "False Positive Rate"
    )
    plt.ylabel(
        "True Positive Rate"
    )
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.legend()
    plt.grid(
        alpha=0.25
    )
    plt.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()


def plot_single_pr(
    y_true: pd.Series,
    y_proba: np.ndarray,
    model_name: str,
    output_path: Path,
) -> None:
    precision, recall, _ = (
        precision_recall_curve(
            y_true,
            y_proba,
        )
    )

    score = (
        average_precision_score(
            y_true,
            y_proba,
        )
    )

    baseline = float(
        y_true.mean()
    )

    plt.figure(
        figsize=(7, 6)
    )

    plt.plot(
        recall,
        precision,
        label=(
            f"PR-AUC={score:.3f}"
        ),
    )

    plt.axhline(
        baseline,
        linestyle="--",
        label=(
            "Positive-rate baseline="
            f"{baseline:.3f}"
        ),
    )

    plt.title(
        f"{model_name} - "
        "Final Test Precision-Recall Curve"
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.legend()
    plt.grid(
        alpha=0.25
    )
    plt.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()


def train_models(
    feature_path: Path,
    model_output: Path,
    metrics_output: Path,
    top_k: int = DEFAULT_TOP_K,
    cv_folds: int = DEFAULT_CV_FOLDS,
) -> tuple[dict, dict]:
    df = pd.read_csv(
        feature_path
    )

    if len(df) < 100:
        raise ValueError(
            "Use at least 100 apps for a meaningful demo. "
            "A few thousand is strongly recommended."
        )

    if cv_folds < 2:
        raise ValueError(
            "cv_folds must be at least 2."
        )

    embedding_columns = sorted(
        column
        for column
        in df.columns
        if column.startswith(
            "emb_"
        )
    )

    if not embedding_columns:
        raise ValueError(
            "No embedding columns found."
        )

    numeric_columns = (
        BASE_NUMERIC_COLUMNS
        + embedding_columns
    )

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

    missing = (
        needed
        - set(
            df.columns
        )
    )

    if missing:
        raise ValueError(
            "Missing model columns: "
            f"{sorted(missing)}"
        )

    model_configs = (
        get_model_configs()
    )

    training_state = (
        build_training_state(
            feature_path=(
                feature_path
            ),
            df=df,
            embedding_columns=(
                embedding_columns
            ),
            top_k=top_k,
            cv_folds=cv_folds,
            model_configs=(
                model_configs
            ),
        )
    )

    ensure_training_state(
        training_state
    )

    y = (
        df["success"]
        .astype(int)
        .to_numpy()
    )

    (
        development_idx,
        comparison_idx,
        final_test_idx,
    ) = load_or_create_split_indices(
        y
    )

    development_df = (
        df.iloc[
            development_idx
        ]
        .reset_index(
            drop=True
        )
    )

    comparison_df = (
        df.iloc[
            comparison_idx
        ]
        .reset_index(
            drop=True
        )
    )

    final_test_df = (
        df.iloc[
            final_test_idx
        ]
        .reset_index(
            drop=True
        )
    )

    model_columns = (
        numeric_columns
        + CATEGORICAL_COLUMNS
    )

    print(
        "\nPreparing leakage-safe "
        "cross-validation folds..."
    )

    fold_data = (
        prepare_cv_folds(
            development_df=(
                development_df
            ),
            embedding_columns=(
                embedding_columns
            ),
            model_columns=(
                model_columns
            ),
            top_k=top_k,
            cv_folds=cv_folds,
        )
    )

    (
        hyperparameter_results,
        best_configs,
        cv_fold_results,
    ) = run_hyperparameter_search(
        model_configs=(
            model_configs
        ),
        fold_data=fold_data,
        numeric_columns=(
            numeric_columns
        ),
    )

    print(
        "\nPreparing development and "
        "comparison features..."
    )

    dev_comparison_cache_path = (
        FEATURE_CACHE_DIR
        / "development_comparison.joblib"
    )

    cached_dev_comparison = (
        load_cached_feature_set(
            dev_comparison_cache_path,
            {
                "development_features",
                "comparison_features",
            },
        )
    )

    if cached_dev_comparison is not None:
        print(
            "  Loaded development/comparison features "
            "from checkpoint."
        )

        development_features = (
            cached_dev_comparison[
                "development_features"
            ]
        )

        comparison_features = (
            cached_dev_comparison[
                "comparison_features"
            ]
        )
    else:
        development_features = (
            build_leakage_safe_features(
                query_df=development_df,
                reference_df=development_df,
                embedding_columns=(
                    embedding_columns
                ),
                top_k=top_k,
                exclude_self=True,
            )
        )

        comparison_features = (
            build_leakage_safe_features(
                query_df=comparison_df,
                reference_df=development_df,
                embedding_columns=(
                    embedding_columns
                ),
                top_k=top_k,
                exclude_self=False,
            )
        )

        atomic_joblib_dump(
            {
                "development_features": (
                    development_features
                ),
                "comparison_features": (
                    comparison_features
                ),
            },
            dev_comparison_cache_path,
        )

    X_development = (
        development_features[
            model_columns
        ]
    )

    y_development = (
        development_features[
            "success"
        ]
        .astype(int)
    )

    X_comparison = (
        comparison_features[
            model_columns
        ]
    )

    y_comparison = (
        comparison_features[
            "success"
        ]
        .astype(int)
    )

    model_results = {}
    curve_data = {}
    prediction_rows = []
    feature_importance_rows = []

    ensure_evaluation_dirs(
        evaluation_dir=(
            EVALUATION_DIR
        ),
        confusion_dir=(
            CONFUSION_MATRIX_DIR
        ),
        feature_importance_dir=(
            FEATURE_IMPORTANCE_DIR
        ),
    )

    confusion_paths = {}
    feature_importance_paths = {}
    feature_importance_methods = {}

    print(
        "\nEvaluating the best configuration "
        "of each model on the comparison set..."
    )

    COMPARISON_CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    app_ids = (
        comparison_df[
            "appId"
        ]
        .astype(str)
        .to_numpy()
    )

    for model_name, config in (
        best_configs.items()
    ):
        checkpoint_path = (
            COMPARISON_CHECKPOINT_DIR
            / f"{model_name}.joblib"
        )

        checkpoint = safe_joblib_load(
            checkpoint_path
        )

        checkpoint_valid = (
            isinstance(
                checkpoint,
                dict,
            )
            and checkpoint.get(
                "model"
            ) == model_name
            and int(
                checkpoint.get(
                    "config_id",
                    -1,
                )
            ) == int(
                config[
                    "config_id"
                ]
            )
            and checkpoint.get(
                "parameters"
            ) == config[
                "parameters"
            ]
            and isinstance(
                checkpoint.get(
                    "metrics"
                ),
                dict,
            )
        )

        if checkpoint_valid:
            print(
                f"  {model_name}: "
                "comparison checkpoint loaded"
            )

            metrics = checkpoint[
                "metrics"
            ]

            pred = np.asarray(
                checkpoint[
                    "pred"
                ],
                dtype=int,
            )

            proba = np.asarray(
                checkpoint[
                    "proba"
                ],
                dtype=float,
            )

            importance_df = pd.DataFrame(
                checkpoint[
                    "importance_records"
                ]
            )

            method = checkpoint[
                "importance_method"
            ]
        else:
            print(
                f"  Evaluating "
                f"{model_name}..."
            )

            model = make_pipeline(
                estimator=clone(
                    config[
                        "estimator"
                    ]
                ),
                numeric_columns=(
                    numeric_columns
                ),
            )

            model.fit(
                X_development,
                y_development,
            )

            (
                metrics,
                pred,
                proba,
            ) = evaluate_model(
                model,
                X_comparison,
                y_comparison,
            )

            importance_df, method = (
                feature_importance_frame(
                    model=model,
                    X_eval=(
                        X_comparison
                    ),
                    y_eval=(
                        y_comparison
                    ),
                    top_n=25,
                    random_state=(
                        RANDOM_STATE
                    ),
                )
            )

            atomic_joblib_dump(
                {
                    "model": (
                        model_name
                    ),
                    "config_id": (
                        config[
                            "config_id"
                        ]
                    ),
                    "parameters": (
                        config[
                            "parameters"
                        ]
                    ),
                    "metrics": (
                        metrics
                    ),
                    "pred": np.asarray(
                        pred,
                        dtype=int,
                    ),
                    "proba": np.asarray(
                        proba,
                        dtype=float,
                    ),
                    "importance_records": (
                        importance_df
                        .to_dict(
                            orient="records"
                        )
                    ),
                    "importance_method": (
                        method
                    ),
                },
                checkpoint_path,
            )

        model_results[
            model_name
        ] = {
            "config_id": (
                config[
                    "config_id"
                ]
            ),
            "parameters": (
                config[
                    "parameters"
                ]
            ),
            "cv_metrics": (
                config[
                    "cv_metrics"
                ]
            ),
            "metrics": (
                metrics
            ),
        }

        curve_data[
            model_name
        ] = {
            "y_true": (
                y_comparison
                .to_numpy()
            ),
            "y_proba": (
                proba
            ),
        }

        for index in range(
            len(
                y_comparison
            )
        ):
            prediction_rows.append(
                {
                    "model": (
                        model_name
                    ),
                    "config_id": (
                        config[
                            "config_id"
                        ]
                    ),
                    "appId": (
                        app_ids[
                            index
                        ]
                    ),
                    "y_true": int(
                        y_comparison.iloc[
                            index
                        ]
                    ),
                    "y_pred": int(
                        pred[
                            index
                        ]
                    ),
                    "probability": float(
                        proba[
                            index
                        ]
                    ),
                }
            )

        confusion_path = (
            CONFUSION_MATRIX_DIR
            / f"{model_name}.png"
        )

        plot_confusion_matrix(
            y_true=(
                y_comparison
            ),
            y_pred=pred,
            model_name=(
                model_name
                .replace(
                    "_",
                    " ",
                )
                .title()
            ),
            output_path=(
                confusion_path
            ),
        )

        confusion_paths[
            model_name
        ] = (
            confusion_path
        )

        for _, row in (
            importance_df
            .iterrows()
        ):
            feature_importance_rows.append(
                {
                    "model": (
                        model_name
                    ),
                    "config_id": (
                        config[
                            "config_id"
                        ]
                    ),
                    "feature": (
                        row[
                            "feature"
                        ]
                    ),
                    "importance": float(
                        row[
                            "importance"
                        ]
                    ),
                    "absolute_importance": float(
                        row[
                            "absolute_importance"
                        ]
                    ),
                    "method": (
                        method
                    ),
                }
            )

        importance_path = (
            FEATURE_IMPORTANCE_DIR
            / f"{model_name}.png"
        )

        plot_feature_importance(
            importance_df=(
                importance_df
            ),
            method=method,
            model_name=(
                model_name
                .replace(
                    "_",
                    " ",
                )
                .title()
            ),
            output_path=(
                importance_path
            ),
        )

        feature_importance_paths[
            model_name
        ] = (
            importance_path
        )

        feature_importance_methods[
            model_name
        ] = (
            method
        )

    best_name = max(
        model_results,
        key=lambda name: (
            model_results[
                name
            ][
                "metrics"
            ][
                "roc_auc"
            ],
            model_results[
                name
            ][
                "metrics"
            ][
                "f1"
            ],
        ),
    )

    selected_config = (
        best_configs[
            best_name
        ]
    )

    print(
        "\nSelected model after comparison: "
        f"{best_name}"
    )

    (
        hyperparameter_df,
        comparison_metrics_df,
    ) = save_results_csv(
        hyperparameter_results=(
            hyperparameter_results
        ),
        model_results=(
            model_results
        ),
        cv_fold_results=(
            cv_fold_results
        ),
        prediction_rows=(
            prediction_rows
        ),
        feature_importance_rows=(
            feature_importance_rows
        ),
        hyperparameter_output=(
            HYPERPARAMETER_RESULTS_CSV
        ),
        comparison_output=(
            MODEL_COMPARISON_CSV
        ),
        cv_output=(
            CV_FOLD_RESULTS_CSV
        ),
        predictions_output=(
            MODEL_PREDICTIONS_CSV
        ),
        feature_importance_output=(
            FEATURE_IMPORTANCE_CSV
        ),
    )

    plot_model_comparison(
        comparison_df=(
            comparison_metrics_df
        ),
        output_path=(
            MODEL_COMPARISON_PNG
        ),
    )

    plot_roc_curves(
        curve_data=(
            curve_data
        ),
        output_path=(
            ROC_CURVES_PNG
        ),
    )

    plot_precision_recall_curves(
        curve_data=(
            curve_data
        ),
        output_path=(
            PR_CURVES_PNG
        ),
    )

    print(
        "\nEvaluating selected model on "
        "the untouched final test set..."
    )

    selection_pool_df = (
        pd.concat(
            [
                development_df,
                comparison_df,
            ],
            axis=0,
            ignore_index=True,
        )
    )

    selection_cache_path = (
        FEATURE_CACHE_DIR
        / "selection_and_final_test.joblib"
    )

    cached_selection = (
        load_cached_feature_set(
            selection_cache_path,
            {
                "selection_pool_features",
                "final_test_features",
            },
        )
    )

    if cached_selection is not None:
        print(
            "  Loaded selection/final-test features "
            "from checkpoint."
        )

        selection_pool_features = (
            cached_selection[
                "selection_pool_features"
            ]
        )

        final_test_features = (
            cached_selection[
                "final_test_features"
            ]
        )
    else:
        selection_pool_features = (
            build_leakage_safe_features(
                query_df=(
                    selection_pool_df
                ),
                reference_df=(
                    selection_pool_df
                ),
                embedding_columns=(
                    embedding_columns
                ),
                top_k=top_k,
                exclude_self=True,
            )
        )

        final_test_features = (
            build_leakage_safe_features(
                query_df=(
                    final_test_df
                ),
                reference_df=(
                    selection_pool_df
                ),
                embedding_columns=(
                    embedding_columns
                ),
                top_k=top_k,
                exclude_self=False,
            )
        )

        atomic_joblib_dump(
            {
                "selection_pool_features": (
                    selection_pool_features
                ),
                "final_test_features": (
                    final_test_features
                ),
            },
            selection_cache_path,
        )

    X_selection = (
        selection_pool_features[
            model_columns
        ]
    )

    y_selection = (
        selection_pool_features[
            "success"
        ]
        .astype(int)
    )

    X_final_test = (
        final_test_features[
            model_columns
        ]
    )

    y_final_test = (
        final_test_features[
            "success"
        ]
        .astype(int)
    )

    final_checkpoint = safe_joblib_load(
        FINAL_TEST_CHECKPOINT
    )

    final_checkpoint_valid = (
        isinstance(
            final_checkpoint,
            dict,
        )
        and final_checkpoint.get(
            "model"
        ) == best_name
        and int(
            final_checkpoint.get(
                "config_id",
                -1,
            )
        ) == int(
            selected_config[
                "config_id"
            ]
        )
        and final_checkpoint.get(
            "parameters"
        ) == selected_config[
            "parameters"
        ]
    )

    if final_checkpoint_valid:
        print(
            "  Final-test checkpoint loaded."
        )

        final_test_metrics = (
            final_checkpoint[
                "metrics"
            ]
        )

        final_pred = np.asarray(
            final_checkpoint[
                "pred"
            ],
            dtype=int,
        )

        final_proba = np.asarray(
            final_checkpoint[
                "proba"
            ],
            dtype=float,
        )

        final_importance_df = (
            pd.DataFrame(
                final_checkpoint[
                    "importance_records"
                ]
            )
        )

        final_importance_method = (
            final_checkpoint[
                "importance_method"
            ]
        )
    else:
        final_eval_model = (
            make_pipeline(
                estimator=clone(
                    selected_config[
                        "estimator"
                    ]
                ),
                numeric_columns=(
                    numeric_columns
                ),
            )
        )

        final_eval_model.fit(
            X_selection,
            y_selection,
        )

        (
            final_test_metrics,
            final_pred,
            final_proba,
        ) = evaluate_model(
            final_eval_model,
            X_final_test,
            y_final_test,
        )

        (
            final_importance_df,
            final_importance_method,
        ) = feature_importance_frame(
            model=(
                final_eval_model
            ),
            X_eval=(
                X_final_test
            ),
            y_eval=(
                y_final_test
            ),
            top_n=25,
            random_state=(
                RANDOM_STATE
            ),
        )

        atomic_joblib_dump(
            {
                "model": (
                    best_name
                ),
                "config_id": (
                    selected_config[
                        "config_id"
                    ]
                ),
                "parameters": (
                    selected_config[
                        "parameters"
                    ]
                ),
                "metrics": (
                    final_test_metrics
                ),
                "pred": np.asarray(
                    final_pred,
                    dtype=int,
                ),
                "proba": np.asarray(
                    final_proba,
                    dtype=float,
                ),
                "importance_records": (
                    final_importance_df
                    .to_dict(
                        orient="records"
                    )
                ),
                "importance_method": (
                    final_importance_method
                ),
            },
            FINAL_TEST_CHECKPOINT,
        )

    final_confusion_path = (
        EVALUATION_DIR
        / "final_selected_confusion_matrix.png"
    )

    final_roc_path = (
        EVALUATION_DIR
        / "final_selected_roc_curve.png"
    )

    final_pr_path = (
        EVALUATION_DIR
        / "final_selected_precision_recall_curve.png"
    )

    final_feature_importance_path = (
        EVALUATION_DIR
        / "final_selected_feature_importance.png"
    )

    plot_confusion_matrix(
        y_true=(
            y_final_test
        ),
        y_pred=(
            final_pred
        ),
        model_name=(
            best_name
            .replace(
                "_",
                " ",
            )
            .title()
            + " - Final Test"
        ),
        output_path=(
            final_confusion_path
        ),
    )

    plot_single_roc(
        y_true=(
            y_final_test
        ),
        y_proba=(
            final_proba
        ),
        model_name=(
            best_name
            .replace(
                "_",
                " ",
            )
            .title()
        ),
        output_path=(
            final_roc_path
        ),
    )

    plot_single_pr(
        y_true=(
            y_final_test
        ),
        y_proba=(
            final_proba
        ),
        model_name=(
            best_name
            .replace(
                "_",
                " ",
            )
            .title()
        ),
        output_path=(
            final_pr_path
        ),
    )

    plot_feature_importance(
        importance_df=(
            final_importance_df
        ),
        method=(
            final_importance_method
        ),
        model_name=(
            best_name
            .replace(
                "_",
                " ",
            )
            .title()
            + " - Final Test"
        ),
        output_path=(
            final_feature_importance_path
        ),
    )


    dataset_summary = {
        "total_rows": int(
            len(df)
        ),
        "development_rows": int(
            len(
                development_df
            )
        ),
        "comparison_rows": int(
            len(
                comparison_df
            )
        ),
        "final_test_rows": int(
            len(
                final_test_df
            )
        ),
        "positive_rate": float(
            df[
                "success"
            ]
            .astype(int)
            .mean()
        ),
        "cv_folds": int(
            cv_folds
        ),
        "model_count": int(
            len(
                model_configs
            )
        ),
        "configuration_count": int(
            sum(
                len(
                    configs
                )
                for configs
                in model_configs.values()
            )
        ),
        "top_k": int(
            top_k
        ),
    }

    selected_model_summary = {
        "name": (
            best_name
        ),
        "config_id": (
            selected_config[
                "config_id"
            ]
        ),
        "parameters": (
            selected_config[
                "parameters"
            ]
        ),
    }

    generate_html_report(
        output_path=(
            MODEL_EVALUATION_REPORT
        ),
        hyperparameter_df=(
            hyperparameter_df
        ),
        comparison_df=(
            comparison_metrics_df
        ),
        model_results=(
            model_results
        ),
        selected_model=(
            selected_model_summary
        ),
        final_test_metrics=(
            final_test_metrics
        ),
        dataset_summary=(
            dataset_summary
        ),
        model_comparison_png=(
            MODEL_COMPARISON_PNG
        ),
        roc_png=(
            ROC_CURVES_PNG
        ),
        pr_png=(
            PR_CURVES_PNG
        ),
        confusion_paths=(
            confusion_paths
        ),
        feature_importance_paths=(
            feature_importance_paths
        ),
        feature_importance_methods=(
            feature_importance_methods
        ),
        final_confusion_path=(
            final_confusion_path
        ),
        final_roc_path=(
            final_roc_path
        ),
        final_pr_path=(
            final_pr_path
        ),
        final_feature_importance_path=(
            final_feature_importance_path
        ),
        final_feature_importance_method=(
            final_importance_method
        ),
    )

    all_features_cache_path = (
        FEATURE_CACHE_DIR
        / "all_features.joblib"
    )

    cached_all_features = (
        safe_joblib_load(
            all_features_cache_path
        )
    )

    if isinstance(
        cached_all_features,
        pd.DataFrame,
    ):
        print(
            "  Loaded full training features "
            "from checkpoint."
        )

        all_features = (
            cached_all_features
        )
    else:
        all_features = (
            build_leakage_safe_features(
                query_df=df,
                reference_df=df,
                embedding_columns=(
                    embedding_columns
                ),
                top_k=top_k,
                exclude_self=True,
            )
        )

        atomic_joblib_dump(
            all_features,
            all_features_cache_path,
        )

    X_full = (
        all_features[
            model_columns
        ]
    )

    y_full = (
        all_features[
            "success"
        ]
        .astype(int)
    )

    metrics = {
        "dataset": (
            dataset_summary
        ),
        "hyperparameter_search": (
            hyperparameter_results
        ),
        "model_comparison": (
            model_results
        ),
        "selected_model": {
            **selected_model_summary,
            "selection_metric": (
                "comparison_roc_auc_then_f1"
            ),
            "final_test_metrics": (
                final_test_metrics
            ),
            "final_fit_rows": int(
                len(
                    X_full
                )
            ),
            "evaluation_market_reference": (
                "split-specific training pool only"
            ),
        },
    }

    final_model_state = {
        "model": (
            best_name
        ),
        "config_id": int(
            selected_config[
                "config_id"
            ]
        ),
        "parameters": (
            selected_config[
                "parameters"
            ]
        ),
        "feature_sha256": (
            training_state[
                "feature_sha256"
            ]
        ),
        "top_k": int(
            top_k
        ),
        "cv_folds": int(
            cv_folds
        ),
    }

    completed_final_model = (
        safe_load_json(
            FINAL_MODEL_DONE_JSON
        )
    )

    bundle = None

    if (
        completed_final_model
        == final_model_state
        and model_output.exists()
    ):
        bundle = safe_joblib_load(
            model_output
        )

        if bundle is not None:
            print(
                "\nFinal trained model already exists for "
                "this exact run. Loading it from disk."
            )

    if bundle is None:
        print(
            "\nRetraining selected model on "
            "all historical apps..."
        )

        final_model = (
            make_pipeline(
                estimator=clone(
                    selected_config[
                        "estimator"
                    ]
                ),
                numeric_columns=(
                    numeric_columns
                ),
            )
        )

        final_model.fit(
            X_full,
            y_full,
        )

        bundle = {
            "model": (
                final_model
            ),
            "model_name": (
                best_name
            ),
            "model_params": (
                json.loads(
                    selected_config[
                        "parameters"
                    ]
                )
            ),
            "cv_metrics": (
                selected_config[
                    "cv_metrics"
                ]
            ),
            "comparison_metrics": (
                model_results[
                    best_name
                ][
                    "metrics"
                ]
            ),
            "test_metrics": (
                final_test_metrics
            ),
            "numeric_columns": (
                numeric_columns
            ),
            "categorical_columns": (
                CATEGORICAL_COLUMNS
            ),
            "embedding_columns": (
                embedding_columns
            ),
            "embedding_model": (
                EMBEDDING_MODEL
            ),
            "top_k": int(
                top_k
            ),
            "training_signature": (
                final_model_state
            ),
        }

        model_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        atomic_joblib_dump(
            bundle,
            model_output,
        )

        atomic_write_json(
            FINAL_MODEL_DONE_JSON,
            final_model_state,
        )

    atomic_write_json(
        metrics_output,
        metrics,
    )

    print(
        "\nSelected model: "
        f"{best_name}"
    )

    print(
        json.dumps(
            final_test_metrics,
            indent=2,
        )
    )

    print(
        "\nEvaluation report: "
        f"{MODEL_EVALUATION_REPORT}"
    )

    print(
        "\nInteractive dashboard data:"
    )
    print(
        f"  {MODEL_COMPARISON_CSV}"
    )
    print(
        f"  {HYPERPARAMETER_RESULTS_CSV}"
    )
    print(
        f"  {CV_FOLD_RESULTS_CSV}"
    )
    print(
        f"  {MODEL_PREDICTIONS_CSV}"
    )
    print(
        f"  {FEATURE_IMPORTANCE_CSV}"
    )

    return (
        bundle,
        metrics,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--features",
        default=str(
            FEATURES_CSV
        ),
    )

    parser.add_argument(
        "--model-output",
        default=str(
            MODEL_BUNDLE
        ),
    )

    parser.add_argument(
        "--metrics-output",
        default=str(
            METRICS_JSON
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
    )

    parser.add_argument(
        "--cv-folds",
        type=int,
        default=DEFAULT_CV_FOLDS,
    )

    args = parser.parse_args()

    train_models(
        feature_path=Path(
            args.features
        ),
        model_output=Path(
            args.model_output
        ),
        metrics_output=Path(
            args.metrics_output
        ),
        top_k=args.top_k,
        cv_folds=(
            args.cv_folds
        ),
    )


if __name__ == "__main__":
    main()
