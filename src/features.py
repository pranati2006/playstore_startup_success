from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .config import (
    DEFAULT_TOP_K,
    EMBEDDINGS_NPY,
    FEATURES_CSV,
    PREPARED_CSV,
)


MARKET_COLUMNS = [
    "market_mean_similarity",
    "market_median_installs",
    "market_median_reviews",
    "market_mean_rating",
    "market_free_ratio",
    "market_iap_ratio",
    "market_ads_ratio",
    "market_success_ratio",
    "market_close_competitors",
]


def _safe_float_array(
    df: pd.DataFrame,
    column: str,
) -> np.ndarray:
    return (
        pd.to_numeric(df[column], errors="coerce")
        .fillna(0)
        .to_numpy(float)
    )


def _safe_bool_array(
    df: pd.DataFrame,
    column: str,
) -> np.ndarray:
    series = df[column]
    if series.dtype == bool:
        return series.astype(float).to_numpy()

    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y", "t"])
        .astype(float)
        .to_numpy()
    )


def market_features_against_reference(
    query_embeddings: np.ndarray,
    reference_df: pd.DataFrame,
    reference_embeddings: np.ndarray,
    top_k: int = DEFAULT_TOP_K,
    query_app_ids: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, list[np.ndarray], list[np.ndarray]]:
    """
    Build semantic-market features for query apps using only a reference pool.

    If query_app_ids is provided, a reference row with the same appId is
    removed. This is used when training rows query the same training pool.
    """
    if len(reference_df) != len(reference_embeddings):
        raise ValueError(
            "Reference dataframe and embeddings must have the same row count."
        )
    if len(reference_df) == 0:
        raise ValueError("Reference pool is empty.")

    extra = 1 if query_app_ids is not None else 0
    request_k = min(top_k + extra, len(reference_df))

    nn = NearestNeighbors(
        n_neighbors=request_k,
        metric="cosine",
        algorithm="brute",
    )
    nn.fit(reference_embeddings)
    distances, indices = nn.kneighbors(query_embeddings)

    ref_app_ids = reference_df["appId"].astype(str).to_numpy()

    real_installs = _safe_float_array(reference_df, "realInstalls")
    reviews = _safe_float_array(reference_df, "reviews")
    scores = _safe_float_array(reference_df, "score")
    free = _safe_bool_array(reference_df, "free")
    iap = _safe_bool_array(reference_df, "offersIAP")
    ads = _safe_bool_array(reference_df, "containsAds")
    success = _safe_float_array(reference_df, "success")

    rows: list[dict] = []
    kept_indices: list[np.ndarray] = []
    kept_similarities: list[np.ndarray] = []

    for row_number, (row_distances, row_indices) in enumerate(
        zip(distances, indices)
    ):
        similarities = 1.0 - row_distances

        if query_app_ids is not None:
            query_id = str(query_app_ids[row_number])
            keep_mask = ref_app_ids[row_indices] != query_id
            row_indices = row_indices[keep_mask]
            similarities = similarities[keep_mask]

        row_indices = row_indices[:top_k]
        similarities = similarities[:top_k]

        if len(row_indices) == 0:
            raise ValueError("No semantic neighbors remain after self-exclusion.")

        rows.append(
            {
                "market_mean_similarity": float(np.mean(similarities)),
                "market_median_installs": float(
                    np.median(real_installs[row_indices])
                ),
                "market_median_reviews": float(
                    np.median(reviews[row_indices])
                ),
                "market_mean_rating": float(
                    np.mean(scores[row_indices])
                ),
                "market_free_ratio": float(np.mean(free[row_indices])),
                "market_iap_ratio": float(np.mean(iap[row_indices])),
                "market_ads_ratio": float(np.mean(ads[row_indices])),
                "market_success_ratio": float(
                    np.mean(success[row_indices])
                ),
                "market_close_competitors": int(
                    np.sum(similarities >= 0.70)
                ),
            }
        )
        kept_indices.append(row_indices)
        kept_similarities.append(similarities)

    return pd.DataFrame(rows), kept_indices, kept_similarities


def build_training_features(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    top_k: int = DEFAULT_TOP_K,
) -> pd.DataFrame:
    """
    Create an inspectable full-data feature table.

    The training script recomputes neighborhood features after the train/test
    split to keep evaluation leakage-safe.
    """
    if len(df) != len(embeddings):
        raise ValueError(
            "Prepared dataframe and embedding matrix must have same row count."
        )

    market, _, _ = market_features_against_reference(
        query_embeddings=embeddings,
        reference_df=df,
        reference_embeddings=embeddings,
        top_k=top_k,
        query_app_ids=df["appId"].astype(str).tolist(),
    )

    result = df.copy().reset_index(drop=True)

    embedding_frame = pd.DataFrame(
        embeddings,
        columns=[
            f"emb_{i:03d}"
            for i in range(embeddings.shape[1])
        ],
    )

    result = pd.concat(
        [result, embedding_frame, market.reset_index(drop=True)],
        axis=1,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(PREPARED_CSV))
    parser.add_argument("--embeddings", default=str(EMBEDDINGS_NPY))
    parser.add_argument("--output", default=str(FEATURES_CSV))
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    embeddings = np.load(args.embeddings)

    features = build_training_features(
        df=df,
        embeddings=embeddings,
        top_k=args.top_k,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output, index=False)
    print(f"Saved feature table {features.shape} to {output}")


if __name__ == "__main__":
    main()
