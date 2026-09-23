from __future__ import annotations

import html
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


COMPARISON_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "balanced_accuracy",
    "specificity",
]


def calculate_metrics(y_true, y_pred, y_proba) -> dict:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "specificity": specificity,
        "confusion_matrix": matrix.tolist(),
    }


def ensure_evaluation_dirs(evaluation_dir: Path, confusion_dir: Path, feature_importance_dir: Path) -> None:
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    confusion_dir.mkdir(parents=True, exist_ok=True)
    feature_importance_dir.mkdir(parents=True, exist_ok=True)

    for directory in [confusion_dir, feature_importance_dir]:
        for old_file in directory.glob("*.png"):
            old_file.unlink()


def save_results_csv(
    hyperparameter_results: list[dict],
    model_results: dict[str, dict],
    hyperparameter_output: Path,
    comparison_output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    hyperparameter_df = pd.DataFrame(hyperparameter_results)

    comparison_rows = []
    for model_name, result in model_results.items():
        row = {
            "model": model_name,
            "config_id": result["config_id"],
            "parameters": result["parameters"],
        }

        for metric in COMPARISON_METRICS:
            row[metric] = result["metrics"][metric]

        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows)

    hyperparameter_output.parent.mkdir(parents=True, exist_ok=True)
    comparison_output.parent.mkdir(parents=True, exist_ok=True)

    hyperparameter_df.to_csv(hyperparameter_output, index=False)
    comparison_df.to_csv(comparison_output, index=False)

    return hyperparameter_df, comparison_df


def plot_model_comparison(comparison_df: pd.DataFrame, output_path: Path) -> None:
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]

    chart_df = (
        comparison_df[["model", *metrics]]
        .set_index("model")
        .sort_values("roc_auc", ascending=False)
    )

    ax = chart_df.plot(kind="bar", figsize=(15, 8))
    ax.set_title("Comparative Model Performance")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_roc_curves(curve_data: dict[str, dict], output_path: Path) -> None:
    plt.figure(figsize=(10, 8))

    for model_name, data in curve_data.items():
        fpr, tpr, _ = roc_curve(data["y_true"], data["y_proba"])
        auc = roc_auc_score(data["y_true"], data["y_proba"])
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
    plt.title("ROC Curves - Best Configuration of Each Model")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.grid(alpha=0.25)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_precision_recall_curves(curve_data: dict[str, dict], output_path: Path) -> None:
    plt.figure(figsize=(10, 8))

    for model_name, data in curve_data.items():
        precision, recall, _ = precision_recall_curve(data["y_true"], data["y_proba"])
        pr_auc = average_precision_score(data["y_true"], data["y_proba"])
        plt.plot(recall, precision, label=f"{model_name} (AP={pr_auc:.3f})")

    positive_rate = float(np.mean(next(iter(curve_data.values()))["y_true"]))
    plt.axhline(
        positive_rate,
        linestyle="--",
        label=f"Positive-rate baseline ({positive_rate:.3f})",
    )

    plt.title("Precision-Recall Curves - Best Configuration of Each Model")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.grid(alpha=0.25)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, model_name: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=[0, 1],
        display_labels=["Not successful", "Successful"],
        values_format="d",
        ax=ax,
    )

    ax.set_title(f"{model_name} - Confusion Matrix")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _clean_feature_names(feature_names: np.ndarray) -> np.ndarray:
    cleaned = []

    for name in feature_names:
        name = str(name)
        if "__" in name:
            name = name.split("__", 1)[1]
        cleaned.append(name)

    return np.asarray(cleaned, dtype=object)


def _native_importance(model) -> tuple[np.ndarray, np.ndarray] | None:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    feature_names = _clean_feature_names(preprocessor.get_feature_names_out())

    if hasattr(classifier, "feature_importances_"):
        values = np.asarray(classifier.feature_importances_, dtype=float)
        if len(values) == len(feature_names):
            return feature_names, values

    if hasattr(classifier, "coef_"):
        coefficients = np.asarray(classifier.coef_, dtype=float)
        if coefficients.ndim == 1:
            values = np.abs(coefficients)
        else:
            values = np.mean(np.abs(coefficients), axis=0)

        if len(values) == len(feature_names):
            return feature_names, values

    return None


def feature_importance_frame(
    model,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    top_n: int = 25,
    random_state: int = 42,
) -> tuple[pd.DataFrame, str]:
    native = _native_importance(model)

    if native is not None:
        feature_names, values = native
        method = "native importance / coefficient magnitude"
    else:
        permutation = permutation_importance(
            model,
            X_eval,
            y_eval,
            scoring="roc_auc",
            n_repeats=3,
            random_state=random_state,
            n_jobs=-1,
        )
        feature_names = np.asarray(X_eval.columns, dtype=object)
        values = np.asarray(permutation.importances_mean, dtype=float)
        method = "permutation importance"

    importance_df = pd.DataFrame({"feature": feature_names, "importance": values})
    importance_df["absolute_importance"] = importance_df["importance"].abs()

    importance_df = (
        importance_df
        .sort_values("absolute_importance", ascending=False)
        .head(top_n)
        .sort_values("absolute_importance", ascending=True)
        .reset_index(drop=True)
    )

    return importance_df, method


def plot_feature_importance(
    model,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    model_name: str,
    output_path: Path,
    top_n: int = 25,
    random_state: int = 42,
) -> str:
    importance_df, method = feature_importance_frame(
        model=model,
        X_eval=X_eval,
        y_eval=y_eval,
        top_n=top_n,
        random_state=random_state,
    )

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(importance_df["feature"], importance_df["absolute_importance"])
    ax.set_title(f"{model_name} - Top {len(importance_df)} Features")
    ax.set_xlabel(f"Importance ({method})")
    ax.set_ylabel("Feature")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return method


def _display_model_name(name: str) -> str:
    return name.replace("_", " ").title()


def _metric_table_html(metrics: dict) -> str:
    rows = []

    for metric in COMPARISON_METRICS:
        if metric in metrics:
            rows.append(
                "<tr>"
                f"<td>{html.escape(metric.replace('_', ' ').title())}</td>"
                f"<td>{metrics[metric]:.4f}</td>"
                "</tr>"
            )

    return (
        "<table>"
        "<thead><tr><th>Metric</th><th>Value</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>"
    )


def generate_html_report(
    output_path: Path,
    hyperparameter_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    model_results: dict[str, dict],
    selected_model: dict,
    final_test_metrics: dict,
    dataset_summary: dict,
    model_comparison_png: Path,
    roc_png: Path,
    pr_png: Path,
    confusion_paths: dict[str, Path],
    feature_importance_paths: dict[str, Path],
    feature_importance_methods: dict[str, str],
    final_confusion_path: Path,
    final_roc_path: Path,
    final_pr_path: Path,
    final_feature_importance_path: Path,
    final_feature_importance_method: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def rel(path: Path) -> str:
        return path.relative_to(output_path.parent).as_posix()

    comparison_table = comparison_df.copy().sort_values("roc_auc", ascending=False)
    numeric_cols = [column for column in COMPARISON_METRICS if column in comparison_table.columns]
    comparison_table[numeric_cols] = comparison_table[numeric_cols].round(4)

    hyper_table = hyperparameter_df.copy().sort_values(
        ["cv_roc_auc_mean", "cv_f1_mean"],
        ascending=False,
    )
    metric_columns = [column for column in hyper_table.columns if column.startswith("cv_")]
    hyper_table[metric_columns] = hyper_table[metric_columns].round(4)

    model_sections = []

    for model_name, result in model_results.items():
        display_name = _display_model_name(model_name)
        model_sections.append(
            f"""
            <section class="model-section">
                <h3>{html.escape(display_name)}</h3>
                <p><strong>Selected configuration:</strong> {result["config_id"]}</p>
                <p><strong>Hyperparameters:</strong>
                   <code>{html.escape(result["parameters"])}</code></p>

                {_metric_table_html(result["metrics"])}

                <div class="image-grid">
                    <figure>
                        <img src="{rel(confusion_paths[model_name])}"
                             alt="{html.escape(display_name)} confusion matrix">
                        <figcaption>Confusion matrix</figcaption>
                    </figure>
                    <figure>
                        <img src="{rel(feature_importance_paths[model_name])}"
                             alt="{html.escape(display_name)} feature importance">
                        <figcaption>
                            Feature importance: {html.escape(feature_importance_methods[model_name])}
                        </figcaption>
                    </figure>
                </div>
            </section>
            """
        )

    selected_display = _display_model_name(selected_model["name"])

    css = """
    body { font-family: Arial, sans-serif; margin: 0; background: #f5f6f8; color: #20242a; line-height: 1.5; }
    .container { max-width: 1280px; margin: 0 auto; padding: 32px; }
    h1, h2, h3 { color: #18202a; }
    section { background: #ffffff; margin-bottom: 24px; padding: 24px; border-radius: 12px; box-shadow: 0 1px 5px rgba(0,0,0,0.08); }
    .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; }
    .summary-card { background: #f7f8fa; border: 1px solid #e5e7eb; padding: 16px; border-radius: 10px; }
    .summary-card strong { display: block; margin-bottom: 6px; }
    table { width: 100%; border-collapse: collapse; margin-top: 16px; font-size: 14px; }
    th, td { border: 1px solid #d9dde3; padding: 8px 10px; text-align: left; vertical-align: top; }
    th { background: #eef1f5; }
    .table-wrap { overflow-x: auto; }
    img { max-width: 100%; height: auto; border: 1px solid #e1e4e8; border-radius: 8px; background: #ffffff; }
    .image-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 18px; margin-top: 18px; }
    figure { margin: 0; }
    figcaption { margin-top: 8px; color: #5a6470; font-size: 13px; }
    code { white-space: normal; word-break: break-word; }
    .winner { border: 2px solid #4b5563; }
    """

    document = f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Model Evaluation Report</title>
    <style>{css}</style>
</head>
<body>
<div class="container">
    <h1>Play Store Startup Success - Model Evaluation Report</h1>

    <section>
        <h2>1. Training Summary</h2>
        <div class="summary-grid">
            <div class="summary-card"><strong>Total rows</strong>{dataset_summary["total_rows"]}</div>
            <div class="summary-card"><strong>Development rows</strong>{dataset_summary["development_rows"]}</div>
            <div class="summary-card"><strong>Comparison rows</strong>{dataset_summary["comparison_rows"]}</div>
            <div class="summary-card"><strong>Final test rows</strong>{dataset_summary["final_test_rows"]}</div>
            <div class="summary-card"><strong>Positive rate</strong>{dataset_summary["positive_rate"]:.4f}</div>
            <div class="summary-card"><strong>CV folds</strong>{dataset_summary["cv_folds"]}</div>
            <div class="summary-card"><strong>Models</strong>{dataset_summary["model_count"]}</div>
            <div class="summary-card"><strong>Total configurations</strong>{dataset_summary["configuration_count"]}</div>
        </div>
        <p>
            Hyperparameter selection is performed inside the development set using stratified cross-validation.
            The best configuration of each algorithm is evaluated on the comparison set. The overall winner is
            selected from those comparison results. Only that winner is evaluated on the final untouched test set.
        </p>
    </section>

    <section>
        <h2>2. Comparative Performance</h2>
        <div class="table-wrap">{comparison_table.to_html(index=False, escape=True)}</div>
        <figure>
            <img src="{rel(model_comparison_png)}" alt="Comparative model performance">
            <figcaption>Accuracy, precision, recall, F1, ROC-AUC and PR-AUC on the comparison set.</figcaption>
        </figure>
    </section>

    <section>
        <h2>3. ROC Analysis</h2>
        <figure><img src="{rel(roc_png)}" alt="ROC curves"></figure>
    </section>

    <section>
        <h2>4. Precision-Recall Analysis</h2>
        <figure><img src="{rel(pr_png)}" alt="Precision recall curves"></figure>
    </section>

    <section>
        <h2>5. Hyperparameter Search - All 30 Experiments</h2>
        <div class="table-wrap">{hyper_table.to_html(index=False, escape=True)}</div>
    </section>

    <section>
        <h2>6. Per-Model Diagnostics</h2>
        {''.join(model_sections)}
    </section>

    <section class="winner">
        <h2>7. Final Selected Model</h2>
        <p><strong>Model:</strong> {html.escape(selected_display)}</p>
        <p><strong>Configuration:</strong> {selected_model["config_id"]}</p>
        <p><strong>Hyperparameters:</strong>
           <code>{html.escape(selected_model["parameters"])}</code></p>

        <h3>Final Untouched Test Metrics</h3>
        {_metric_table_html(final_test_metrics)}

        <div class="image-grid">
            <figure>
                <img src="{rel(final_confusion_path)}" alt="Final model confusion matrix">
                <figcaption>Final-test confusion matrix</figcaption>
            </figure>
            <figure>
                <img src="{rel(final_roc_path)}" alt="Final model ROC curve">
                <figcaption>Final-test ROC curve</figcaption>
            </figure>
            <figure>
                <img src="{rel(final_pr_path)}" alt="Final model precision recall curve">
                <figcaption>Final-test precision-recall curve</figcaption>
            </figure>
            <figure>
                <img src="{rel(final_feature_importance_path)}" alt="Final model feature importance">
                <figcaption>Final feature importance: {html.escape(final_feature_importance_method)}</figcaption>
            </figure>
        </div>
    </section>
</div>
</body>
</html>
"""

    output_path.write_text(document, encoding="utf-8")
