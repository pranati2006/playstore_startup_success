from __future__ import annotations

import argparse
import subprocess
import sys


def run(command: list[str]) -> None:
    print("\n$", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the complete Play Store startup-success pipeline."
    )
    parser.add_argument(
        "--skip-scrape",
        action="store_true",
        help="Use an existing data/raw/playstore_apps.csv.",
    )
    parser.add_argument(
        "--country",
        default="us",
        help="Google Play country code used while scraping.",
    )
    parser.add_argument(
        "--lang",
        default="en",
        help="Google Play language code used while scraping.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.35,
        help="Delay between app detail requests.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Semantic neighbors used to create market features.",
    )
    args = parser.parse_args()

    python = sys.executable

    if not args.skip_scrape:
        run(
            [
                python,
                "-m",
                "src.scrape_playstore",
                "--country",
                args.country,
                "--lang",
                args.lang,
                "--delay",
                str(args.delay),
            ]
        )

    run([python, "-m", "src.prepare_data"])
    run([python, "-m", "src.embeddings"])
    run(
        [
            python,
            "-m",
            "src.features",
            "--top-k",
            str(args.top_k),
        ]
    )
    run([python, "-m", "src.train", "--top-k", str(args.top_k)])

    print("\nPipeline complete.")
    print("Launch the UI with: streamlit run app.py")


if __name__ == "__main__":
    main()


#python run_pipeline.py --country in --lang en --top-k 30
#for no scrape
#python run_pipeline.py --skip-scrape --top-k 30