from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
ARTIFACT_DIR = ROOT / "artifacts"
EVALUATION_DIR = ARTIFACT_DIR / "evaluation"

TRAINING_CHECKPOINT_DIR = ARTIFACT_DIR / "training_checkpoints"
CV_CACHE_DIR = TRAINING_CHECKPOINT_DIR / "cv_folds"
FEATURE_CACHE_DIR = TRAINING_CHECKPOINT_DIR / "feature_sets"
SEARCH_CHECKPOINT_DIR = TRAINING_CHECKPOINT_DIR / "hyperparameter_search"
COMPARISON_CHECKPOINT_DIR = TRAINING_CHECKPOINT_DIR / "model_comparison"
TRAINING_STATE_JSON = TRAINING_CHECKPOINT_DIR / "training_state.json"
SPLIT_INDICES_NPZ = TRAINING_CHECKPOINT_DIR / "split_indices.npz"
FINAL_TEST_CHECKPOINT = TRAINING_CHECKPOINT_DIR / "final_selected_test.joblib"
FINAL_MODEL_DONE_JSON = TRAINING_CHECKPOINT_DIR / "final_model_done.json"

RAW_APPS_CSV = RAW_DIR / "playstore_apps.csv"
PREPARED_CSV = PROCESSED_DIR / "apps_prepared.csv"
EMBEDDINGS_NPY = PROCESSED_DIR / "description_embeddings.npy"
FEATURES_CSV = PROCESSED_DIR / "model_features.csv"

MODEL_BUNDLE = ARTIFACT_DIR / "model_bundle.joblib"
METRICS_JSON = ARTIFACT_DIR / "model_metrics.json"
NEIGHBORS_CSV = ARTIFACT_DIR / "last_prediction_neighbors.csv"

MODEL_COMPARISON_CSV = EVALUATION_DIR / "model_comparison.csv"
HYPERPARAMETER_RESULTS_CSV = EVALUATION_DIR / "hyperparameter_results.csv"
CV_FOLD_RESULTS_CSV = EVALUATION_DIR / "cv_fold_results.csv"
MODEL_PREDICTIONS_CSV = EVALUATION_DIR / "model_predictions.csv"
FEATURE_IMPORTANCE_CSV = EVALUATION_DIR / "feature_importance.csv"

MODEL_COMPARISON_PNG = EVALUATION_DIR / "model_comparison.png"
ROC_CURVES_PNG = EVALUATION_DIR / "roc_all_models.png"
PR_CURVES_PNG = EVALUATION_DIR / "precision_recall_all_models.png"

CONFUSION_MATRIX_DIR = EVALUATION_DIR / "confusion_matrices"
FEATURE_IMPORTANCE_DIR = EVALUATION_DIR / "feature_importance"

MODEL_EVALUATION_REPORT = EVALUATION_DIR / "model_evaluation_report.html"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_TOP_K = 30
DEFAULT_CV_FOLDS = 5

SUCCESS_INSTALL_WEIGHT = 0.70
SUCCESS_REVIEW_WEIGHT = 0.30
SUCCESS_THRESHOLD = 0.75

RANDOM_STATE = 42
