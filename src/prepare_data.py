from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    PREPARED_CSV,
    RAW_APPS_CSV,
    SUCCESS_INSTALL_WEIGHT,
    SUCCESS_REVIEW_WEIGHT,
    SUCCESS_THRESHOLD,
)


TRUE_VALUES = {"true", "1", "yes", "y", "t"}


def to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(TRUE_VALUES)
    )


def make_success_target(
    df: pd.DataFrame,
    min_group_size: int = 20,
) -> pd.DataFrame:
    work = df.copy()

    work["log_installs"] = np.log1p(work["realInstalls"])
    work["log_reviews"] = np.log1p(work["reviews"])

    global_install_pct = work["log_installs"].rank(pct=True)
    global_review_pct = work["log_reviews"].rank(pct=True)

    group_sizes = work.groupby("genre")["appId"].transform("size")

    group_install_pct = work.groupby("genre")["log_installs"].rank(pct=True)
    group_review_pct = work.groupby("genre")["log_reviews"].rank(pct=True)

    work["install_percentile"] = np.where(
        group_sizes >= min_group_size,
        group_install_pct,
        global_install_pct,
    )
    work["review_percentile"] = np.where(
        group_sizes >= min_group_size,
        group_review_pct,
        global_review_pct,
    )

    work["success_score"] = (
        SUCCESS_INSTALL_WEIGHT * work["install_percentile"]
        + SUCCESS_REVIEW_WEIGHT * work["review_percentile"]
    )
    work["success"] = (
        work["success_score"] >= SUCCESS_THRESHOLD
    ).astype(int)

    return work


def prepare(raw_path: Path, output_path: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_path)

    required = {
        "appId",
        "description",
        "genre",
        "realInstalls",
        "minInstalls",
        "reviews",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["description"] = df["description"].fillna("").astype(str).str.strip()
    if "summary" not in df.columns:
        df["summary"] = ""
    df["summary"] = df["summary"].fillna("").astype(str).str.strip()
    df["genre"] = df["genre"].fillna("Unknown").astype(str)
    if "contentRating" not in df.columns:
        df["contentRating"] = "Unknown"
    df["contentRating"] = (
        df["contentRating"].fillna("Unknown").astype(str)
    )

    for column in [
        "realInstalls",
        "minInstalls",
        "reviews",
        "ratings",
        "score",
        "price",
    ]:
        if column not in df.columns:
            df[column] = 0
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["realInstalls"] = df["realInstalls"].fillna(df["minInstalls"])
    df["realInstalls"] = df["realInstalls"].fillna(0).clip(lower=0)
    df["reviews"] = df["reviews"].fillna(0).clip(lower=0)
    df["ratings"] = df["ratings"].fillna(0).clip(lower=0)
    df["score"] = df["score"].fillna(0).clip(lower=0, upper=5)
    df["price"] = df["price"].fillna(0).clip(lower=0)

    for column in ["free", "offersIAP", "adSupported", "containsAds"]:
        if column not in df.columns:
            df[column] = False
        df[column] = to_bool(df[column])

    # Require meaningful descriptions and enough public outcome data.
    df = df[
        (df["description"].str.len() >= 40)
        & (df["realInstalls"] > 0)
    ].copy()

    df["description_chars"] = df["description"].str.len()
    df["description_words"] = df["description"].str.split().str.len()
    df["summary_chars"] = df["summary"].str.len()

    df = df.drop_duplicates("appId", keep="last").reset_index(drop=True)
    df = make_success_target(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(
        f"Prepared {len(df)} apps. "
        f"Success rate: {df['success'].mean():.1%}"
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(RAW_APPS_CSV))
    parser.add_argument("--output", default=str(PREPARED_CSV))
    args = parser.parse_args()

    prepare(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
