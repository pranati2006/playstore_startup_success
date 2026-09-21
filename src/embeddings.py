from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL, EMBEDDINGS_NPY, PREPARED_CSV


def encode_descriptions(
    descriptions: list[str],
    model_name: str = EMBEDDING_MODEL,
    batch_size: int = 64,
) -> np.ndarray:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        descriptions,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(PREPARED_CSV))
    parser.add_argument("--output", default=str(EMBEDDINGS_NPY))
    parser.add_argument("--model", default=EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    descriptions = df["description"].fillna("").astype(str).tolist()

    embeddings = encode_descriptions(
        descriptions,
        model_name=args.model,
        batch_size=args.batch_size,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, embeddings)

    print(f"Saved embeddings {embeddings.shape} to {output}")


if __name__ == "__main__":
    main()
