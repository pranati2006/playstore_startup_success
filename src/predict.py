from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .config import (
    EMBEDDINGS_NPY,
    MODEL_BUNDLE,
    NEIGHBORS_CSV,
    PREPARED_CSV,
)


@dataclass
class StartupInput:
    description: str
    summary: str = ""
    genre: str = "Unknown"
    content_rating: str = "Everyone"
    free: bool = True
    price: float = 0.0
    offers_iap: bool = False
    contains_ads: bool = False


class StartupSuccessPredictor:
    def __init__(
        self,
        model_bundle_path: Path = MODEL_BUNDLE,
        prepared_path: Path = PREPARED_CSV,
        embeddings_path: Path = EMBEDDINGS_NPY,
    ) -> None:
        self.bundle = joblib.load(model_bundle_path)
        self.apps = pd.read_csv(prepared_path)
        self.embeddings = np.load(embeddings_path)

        if len(self.apps) != len(self.embeddings):
            raise ValueError(
                "Prepared app data and embeddings have different row counts."
            )

        self.embedding_model = SentenceTransformer(
            self.bundle["embedding_model"]
        )

    def _embed(self, text: str) -> np.ndarray:
        vector = self.embedding_model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vector[0].astype(np.float32)

    def _neighbors(
        self,
        query_embedding: np.ndarray,
        top_k: int,
    ) -> pd.DataFrame:
        similarities = self.embeddings @ query_embedding
        top_k = min(top_k, len(similarities))
        indices = np.argsort(similarities)[-top_k:][::-1]

        neighbors = self.apps.iloc[indices].copy()
        neighbors["similarity"] = similarities[indices]
        return neighbors

    @staticmethod
    def _bool_mean(series: pd.Series) -> float:
        if series.dtype == bool:
            return float(series.mean())
        values = (
            series.fillna(False)
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["true", "1", "yes", "y", "t"])
        )
        return float(values.mean())

    def _market_features(
        self,
        neighbors: pd.DataFrame,
    ) -> dict:
        installs = pd.to_numeric(
            neighbors["realInstalls"],
            errors="coerce",
        ).fillna(0)
        reviews = pd.to_numeric(
            neighbors["reviews"],
            errors="coerce",
        ).fillna(0)
        ratings = pd.to_numeric(
            neighbors["score"],
            errors="coerce",
        ).fillna(0)
        successes = pd.to_numeric(
            neighbors["success"],
            errors="coerce",
        ).fillna(0)

        return {
            "market_mean_similarity": float(
                neighbors["similarity"].mean()
            ),
            "market_median_installs": float(installs.median()),
            "market_median_reviews": float(reviews.median()),
            "market_mean_rating": float(ratings.mean()),
            "market_free_ratio": self._bool_mean(neighbors["free"]),
            "market_iap_ratio": self._bool_mean(
                neighbors["offersIAP"]
            ),
            "market_ads_ratio": self._bool_mean(
                neighbors["containsAds"]
            ),
            "market_success_ratio": float(successes.mean()),
            "market_close_competitors": int(
                (neighbors["similarity"] >= 0.70).sum()
            ),
        }

    def predict(
        self,
        startup: StartupInput,
        top_k: int | None = None,
    ) -> dict:
        if len(startup.description.strip()) < 40:
            raise ValueError(
                "Use a meaningful description of at least 40 characters."
            )

        top_k = int(top_k or self.bundle.get("top_k", 30))
        embedding = self._embed(startup.description)
        neighbors = self._neighbors(embedding, top_k)
        market = self._market_features(neighbors)

        row: dict = {
            "price": float(startup.price),
            "free": int(startup.free),
            "offersIAP": int(startup.offers_iap),
            "containsAds": int(startup.contains_ads),
            "description_chars": len(startup.description),
            "description_words": len(startup.description.split()),
            "summary_chars": len(startup.summary.strip()),
            "genre": startup.genre,
            "contentRating": startup.content_rating,
            **market,
        }

        for i, value in enumerate(embedding):
            row[f"emb_{i:03d}"] = float(value)

        model_columns = (
            self.bundle["numeric_columns"]
            + self.bundle["categorical_columns"]
        )
        X = pd.DataFrame([row]).reindex(columns=model_columns)

        model = self.bundle["model"]
        probability = float(model.predict_proba(X)[0, 1])
        predicted_class = int(probability >= 0.50)

        export_columns = [
            column
            for column in [
                "title",
                "appId",
                "genre",
                "score",
                "realInstalls",
                "reviews",
                "success",
                "similarity",
                "url",
            ]
            if column in neighbors.columns
        ]

        NEIGHBORS_CSV.parent.mkdir(parents=True, exist_ok=True)
        neighbors[export_columns].to_csv(
            NEIGHBORS_CSV,
            index=False,
        )

        return {
            "probability": probability,
            "predicted_class": predicted_class,
            "model_name": self.bundle["model_name"],
            "market": market,
            "neighbors": neighbors[export_columns].copy(),
        }


def probability_band(probability: float) -> str:
    if probability >= 0.75:
        return "High success-profile match"
    if probability >= 0.50:
        return "Moderate success-profile match"
    if probability >= 0.30:
        return "Limited success-profile match"
    return "Low success-profile match"
