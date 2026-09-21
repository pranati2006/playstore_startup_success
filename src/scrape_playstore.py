from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from google_play_scraper import app, search

from .config import RAW_APPS_CSV


KEEP_COLUMNS = [
    "appId",
    "title",
    "summary",
    "description",
    "installs",
    "minInstalls",
    "realInstalls",
    "score",
    "ratings",
    "reviews",
    "price",
    "free",
    "currency",
    "offersIAP",
    "inAppProductPrice",
    "developer",
    "developerId",
    "developerWebsite",
    "genre",
    "genreId",
    "contentRating",
    "adSupported",
    "containsAds",
    "released",
    "updated",
    "version",
    "url",
]


def read_keywords(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def discover_app_ids(
    keywords: list[str],
    lang: str,
    country: str,
    hits_per_keyword: int,
) -> list[str]:
    app_ids: set[str] = set()

    for index, keyword in enumerate(keywords, start=1):
        print(f"[search {index}/{len(keywords)}] {keyword}")
        try:
            results = search(
                keyword,
                lang=lang,
                country=country,
                n_hits=min(hits_per_keyword, 30),
            )
            for row in results:
                app_id = row.get("appId")
                if app_id:
                    app_ids.add(app_id)
        except Exception as exc:
            print(f"  search failed: {exc}")

    return sorted(app_ids)


def normalize_detail(detail: dict) -> dict:
    row = {column: detail.get(column) for column in KEEP_COLUMNS}
    categories = detail.get("categories")
    if categories:
        row["categories"] = "|".join(
            str(item.get("name", "")).strip()
            for item in categories
            if isinstance(item, dict) and item.get("name")
        )
    else:
        row["categories"] = ""

    screenshots = detail.get("screenshots") or []
    row["screenshotCount"] = len(screenshots)
    return row


def collect_details(
    app_ids: list[str],
    lang: str,
    country: str,
    delay: float,
    output_path: Path,
    checkpoint_every: int = 25,
) -> pd.DataFrame:
    rows: list[dict] = []

    existing_ids: set[str] = set()
    if output_path.exists():
        existing = pd.read_csv(output_path)
        if "appId" in existing.columns:
            existing_ids = set(existing["appId"].dropna().astype(str))
            rows = existing.to_dict("records")
            print(f"Resuming with {len(existing_ids)} already collected apps.")

    pending_ids = [app_id for app_id in app_ids if app_id not in existing_ids]

    for index, app_id in enumerate(pending_ids, start=1):
        print(f"[detail {index}/{len(pending_ids)}] {app_id}")
        try:
            detail = app(app_id, lang=lang, country=country)
            rows.append(normalize_detail(detail))
        except Exception as exc:
            print(f"  detail failed: {exc}")

        if index % checkpoint_every == 0:
            pd.DataFrame(rows).drop_duplicates("appId").to_csv(
                output_path, index=False
            )

        if delay > 0:
            time.sleep(delay)

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.drop_duplicates("appId", keep="last")
        frame.to_csv(output_path, index=False)

    return frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Discover and collect live Google Play app metadata."
    )
    parser.add_argument("--keywords", default="keywords.txt")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--country", default="us")
    parser.add_argument("--hits-per-keyword", type=int, default=30)
    parser.add_argument(
        "--delay",
        type=float,
        default=0.35,
        help="Seconds between app-detail requests.",
    )
    parser.add_argument("--output", default=str(RAW_APPS_CSV))
    args = parser.parse_args()

    keyword_path = Path(args.keywords)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    keywords = read_keywords(keyword_path)
    app_ids = discover_app_ids(
        keywords=keywords,
        lang=args.lang,
        country=args.country,
        hits_per_keyword=args.hits_per_keyword,
    )

    print(f"Discovered {len(app_ids)} unique app IDs.")
    frame = collect_details(
        app_ids=app_ids,
        lang=args.lang,
        country=args.country,
        delay=args.delay,
        output_path=output_path,
    )

    print(f"Saved {len(frame)} apps to {output_path}")


if __name__ == "__main__":
    main()
