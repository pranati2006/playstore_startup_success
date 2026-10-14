from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from google_play_scraper import app, search

from .config import RAW_APPS_CSV, ROOT


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
    "categories",
    "screenshotCount",
]

BASE_TERMS = [
    "fitness", "workout", "gym", "running", "cycling", "yoga", "meditation",
    "diet", "calorie", "nutrition", "health", "medical", "doctor", "pharmacy",
    "finance", "budget", "expense", "money", "investment", "stocks", "trading",
    "loan", "banking", "education", "study", "exam", "language", "math",
    "coding", "learning", "kids learning", "travel", "hotel", "flight",
    "itinerary", "navigation", "maps", "taxi", "parking", "food", "recipe",
    "grocery", "restaurant", "delivery", "shopping", "fashion", "beauty",
    "home", "interior", "real estate", "rent", "jobs", "resume", "business",
    "invoice", "accounting", "inventory", "crm", "productivity", "notes",
    "calendar", "task", "habit", "reminder", "scanner", "document", "photo",
    "video", "music", "podcast", "social", "chat", "dating", "community",
    "news", "weather", "sports", "cricket", "football", "gaming", "puzzle",
    "kids games", "parenting", "baby", "pet", "security", "vpn", "password",
    "storage", "file manager", "wallpaper", "launcher", "ai", "chatbot",
    "translator",
]

MODIFIERS = [
    "app", "tracker", "planner", "manager", "assistant", "ai", "smart",
    "online", "offline", "free", "pro", "daily", "personal", "mobile",
    "calculator", "scanner", "editor", "maker", "creator", "organizer",
    "reminder", "monitor", "guide", "coach", "finder", "booking", "learning",
    "training", "community", "tools",
]


def read_keywords(path: Path) -> list[str]:
    if not path.exists():
        return []

    keywords = []
    for line in path.read_text(encoding="utf-8").splitlines():
        keyword = line.strip()
        if keyword and not keyword.startswith("#"):
            keywords.append(keyword)
    return keywords


def unique_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    output = []

    for value in values:
        cleaned = " ".join(value.strip().split())
        if not cleaned:
            continue

        key = cleaned.lower()
        if key in seen:
            continue

        seen.add(key)
        output.append(cleaned)

    return output


def build_search_queries(keyword_file: Path) -> list[str]:
    queries = []
    queries.extend(read_keywords(keyword_file))
    queries.extend(BASE_TERMS)

    for base in BASE_TERMS:
        for modifier in MODIFIERS:
            if base.lower() == modifier.lower():
                continue
            queries.append(f"{base} {modifier}")

    return unique_preserve_order(queries)


def load_string_set(path: Path) -> set[str]:
    if not path.exists():
        return set()

    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def save_string_set(values: set[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sorted(values)), encoding="utf-8")


def search_with_retry(
    query: str,
    lang: str,
    country: str,
    n_hits: int,
    max_retries: int,
) -> list[dict]:
    for attempt in range(1, max_retries + 1):
        try:
            return search(
                query,
                lang=lang,
                country=country,
                n_hits=n_hits,
            )
        except Exception as exc:
            if attempt == max_retries:
                print(f"Search failed for '{query}': {exc}")
                return []

            wait_seconds = min(30, 2 ** attempt)
            print(
                f"Search retry {attempt}/{max_retries} "
                f"for '{query}' after {wait_seconds}s..."
            )
            time.sleep(wait_seconds)

    return []


def discover_app_ids(
    queries: list[str],
    existing_ids: set[str],
    previously_discovered_ids: set[str],
    completed_queries: set[str],
    target_apps: int,
    lang: str,
    country: str,
    hits_per_query: int,
    search_delay: float,
    max_retries: int,
    discovered_ids_path: Path,
    completed_queries_path: Path,
) -> set[str]:
    app_ids = set(existing_ids) | set(previously_discovered_ids)

    discovery_target = max(
        target_apps,
        int(target_apps * 1.10),
    )

    if len(app_ids) >= discovery_target:
        return app_ids

    remaining_queries = [
        query
        for query in queries
        if query not in completed_queries
    ]

    print(f"Search queries available: {len(queries)}")
    print(f"Queries already completed: {len(completed_queries)}")
    print(f"Queries remaining: {len(remaining_queries)}")
    print(f"Existing raw apps: {len(existing_ids)}")
    print(f"Already discovered IDs: {len(previously_discovered_ids)}")
    print(f"Discovery target: {discovery_target}")

    for query_number, query in enumerate(remaining_queries, start=1):
        if len(app_ids) >= discovery_target:
            break

        before = len(app_ids)

        results = search_with_retry(
            query=query,
            lang=lang,
            country=country,
            n_hits=hits_per_query,
            max_retries=max_retries,
        )

        for result in results:
            app_id = result.get("appId")
            if app_id:
                app_ids.add(str(app_id))

        completed_queries.add(query)
        save_string_set(app_ids, discovered_ids_path)

        completed_queries_path.parent.mkdir(parents=True, exist_ok=True)
        completed_queries_path.write_text(
            "\n".join(sorted(completed_queries)),
            encoding="utf-8",
        )

        new_count = len(app_ids) - before

        print(
            f"[{query_number}/{len(remaining_queries)}] "
            f"{query!r}: {len(results)} hits, "
            f"{new_count} new, "
            f"{len(app_ids)}/{discovery_target} unique IDs"
        )

        if search_delay > 0:
            time.sleep(search_delay)

    if len(app_ids) < discovery_target:
        print(
            "\nWARNING: Query bank finished before the "
            f"discovery target was reached. "
            f"Discovered {len(app_ids)} unique IDs."
        )

    return app_ids


def normalize_detail(detail: dict) -> dict:
    row = {
        column: detail.get(column)
        for column in KEEP_COLUMNS
    }

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


def app_detail_with_retry(
    app_id: str,
    lang: str,
    country: str,
    max_retries: int,
) -> dict | None:
    for attempt in range(1, max_retries + 1):
        try:
            return app(
                app_id,
                lang=lang,
                country=country,
            )
        except Exception as exc:
            if attempt == max_retries:
                print(f"Detail failed for {app_id}: {exc}")
                return None

            wait_seconds = min(60, 2 ** attempt)
            print(
                f"Detail retry {attempt}/{max_retries} "
                f"for {app_id} after {wait_seconds}s..."
            )
            time.sleep(wait_seconds)

    return None


def save_checkpoint(
    existing_df: pd.DataFrame,
    new_rows: list[dict],
    output_path: Path,
) -> pd.DataFrame:
    if new_rows:
        new_df = pd.DataFrame(new_rows).reindex(columns=KEEP_COLUMNS)
        combined = pd.concat(
            [existing_df, new_df],
            ignore_index=True,
        )
    else:
        combined = existing_df.copy()

    if "appId" in combined.columns:
        combined = (
            combined
            .drop_duplicates(subset=["appId"], keep="last")
            .reset_index(drop=True)
        )

    combined = combined.reindex(columns=KEEP_COLUMNS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    return combined


def collect_details(
    app_ids: set[str],
    existing_df: pd.DataFrame,
    output_path: Path,
    target_apps: int,
    lang: str,
    country: str,
    delay: float,
    max_retries: int,
    checkpoint_every: int,
) -> pd.DataFrame:
    if "appId" in existing_df.columns:
        existing_ids = set(
            existing_df["appId"]
            .dropna()
            .astype(str)
        )
    else:
        existing_ids = set()

    pending_ids = [
        app_id
        for app_id in sorted(app_ids)
        if app_id not in existing_ids
    ]

    print(f"\nRaw apps already saved: {len(existing_ids)}")
    print(
        "New app IDs available for detail fetch: "
        f"{len(pending_ids)}"
    )
    print(f"Raw target: {target_apps}")

    if len(existing_ids) >= target_apps:
        print(
            "Raw target already reached. "
            "No new detail requests are needed."
        )
        return existing_df

    new_rows = []

    for number, app_id in enumerate(pending_ids, start=1):
        current_total = len(existing_df) + len(new_rows)
        if current_total >= target_apps:
            break

        detail = app_detail_with_retry(
            app_id=app_id,
            lang=lang,
            country=country,
            max_retries=max_retries,
        )

        if detail is not None:
            row = normalize_detail(detail)
            row["appId"] = row.get("appId") or app_id
            new_rows.append(row)

        if len(new_rows) >= checkpoint_every:
            existing_df = save_checkpoint(
                existing_df=existing_df,
                new_rows=new_rows,
                output_path=output_path,
            )
            new_rows = []

            print(
                f"Checkpoint saved: "
                f"{len(existing_df)}/{target_apps} raw apps"
            )

        if delay > 0:
            time.sleep(delay)

    existing_df = save_checkpoint(
        existing_df=existing_df,
        new_rows=new_rows,
        output_path=output_path,
    )

    print(
        f"\nDetail collection complete: "
        f"{len(existing_df)} raw apps saved."
    )

    return existing_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Discover and download a broad Google Play "
            "dataset for startup-success modeling."
        )
    )

    parser.add_argument(
        "--keywords",
        default=str(ROOT / "keywords.txt"),
        help=(
            "Optional seed keyword file. Existing keywords "
            "are used before automatically generated queries."
        ),
    )

    parser.add_argument(
        "--country",
        default="us",
        help=(
            "Google Play country code. Use the same country "
            "as the existing raw dataset when resuming."
        ),
    )

    parser.add_argument(
        "--lang",
        default="en",
        help="Google Play language code.",
    )

    parser.add_argument(
        "--target-apps",
        type=int,
        default=15000,
        help="Desired number of raw unique apps saved to CSV.",
    )

    parser.add_argument(
        "--hits-per-query",
        type=int,
        default=30,
        help=(
            "Search results requested per query. "
            "Values above 30 are clamped to 30."
        ),
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.50,
        help="Delay in seconds between app-detail requests.",
    )

    parser.add_argument(
        "--search-delay",
        type=float,
        default=0.15,
        help="Delay in seconds between search queries.",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries for temporary search/detail failures.",
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Save raw CSV after this many new app details.",
    )

    parser.add_argument(
        "--output",
        default=str(RAW_APPS_CSV),
        help="Raw output CSV path.",
    )

    args = parser.parse_args()

    if args.target_apps <= 0:
        raise ValueError("--target-apps must be greater than 0.")

    if args.max_retries <= 0:
        raise ValueError("--max-retries must be greater than 0.")

    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be greater than 0.")

    output_path = Path(args.output)
    keyword_file = Path(args.keywords)

    hits_per_query = max(
        1,
        min(
            int(args.hits_per_query),
            30,
        ),
    )

    if args.hits_per_query > 30:
        print(
            "hits-per-query was greater than 30; "
            "using 30."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        existing_df = pd.read_csv(output_path)
    else:
        existing_df = pd.DataFrame(columns=KEEP_COLUMNS)

    if not existing_df.empty and "appId" not in existing_df.columns:
        raise ValueError(
            f"{output_path} exists but has no appId column."
        )

    existing_df = existing_df.reindex(columns=KEEP_COLUMNS)

    existing_ids = set(
        existing_df["appId"]
        .dropna()
        .astype(str)
    )

    state_suffix = (
        f"{args.country.lower()}_"
        f"{args.lang.lower()}"
    )

    discovered_ids_path = (
        output_path.parent
        / f"discovered_app_ids_{state_suffix}.txt"
    )

    completed_queries_path = (
        output_path.parent
        / f"completed_search_queries_{state_suffix}.txt"
    )

    previously_discovered_ids = load_string_set(
        discovered_ids_path
    )

    completed_queries = load_string_set(
        completed_queries_path
    )

    queries = build_search_queries(keyword_file)

    print(
        f"Generated {len(queries)} unique search queries."
    )

    discovered_ids = discover_app_ids(
        queries=queries,
        existing_ids=existing_ids,
        previously_discovered_ids=previously_discovered_ids,
        completed_queries=completed_queries,
        target_apps=args.target_apps,
        lang=args.lang,
        country=args.country,
        hits_per_query=hits_per_query,
        search_delay=args.search_delay,
        max_retries=args.max_retries,
        discovered_ids_path=discovered_ids_path,
        completed_queries_path=completed_queries_path,
    )

    final_df = collect_details(
        app_ids=discovered_ids,
        existing_df=existing_df,
        output_path=output_path,
        target_apps=args.target_apps,
        lang=args.lang,
        country=args.country,
        delay=args.delay,
        max_retries=args.max_retries,
        checkpoint_every=args.checkpoint_every,
    )

    print("\n========================================")
    print("SCRAPE SUMMARY")
    print("========================================")
    print(f"Raw rows saved: {len(final_df)}")
    print(
        "Unique appIds: "
        f"{final_df['appId'].nunique()}"
    )
    print(f"Output: {output_path}")
    print("\nNext step:")
    print("python -m src.prepare_data")


if __name__ == "__main__":
    main()
