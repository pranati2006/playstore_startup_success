from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
ARTIFACT_DIR = ROOT / "artifacts"

RAW_APPS_CSV = RAW_DIR / "playstore_apps.csv"
PREPARED_CSV = PROCESSED_DIR / "apps_prepared.csv"
EMBEDDINGS_NPY = PROCESSED_DIR / "description_embeddings.npy"
FEATURES_CSV = PROCESSED_DIR / "model_features.csv"

MODEL_BUNDLE = ARTIFACT_DIR / "model_bundle.joblib"
METRICS_JSON = ARTIFACT_DIR / "model_metrics.json"
NEIGHBORS_CSV = ARTIFACT_DIR / "last_prediction_neighbors.csv"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_K = 30

SUCCESS_INSTALL_WEIGHT = 0.70
SUCCESS_REVIEW_WEIGHT = 0.30
SUCCESS_THRESHOLD = 0.75

RANDOM_STATE = 42
