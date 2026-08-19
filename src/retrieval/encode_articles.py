import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from src.data.load import load_news


def process_news(path: Path) -> pd.DataFrame:
    news = load_news(path)

    news = (
        news[["news_id", "title"]]
        .drop_duplicates(subset="news_id")
        .dropna(subset=["news_id", "title"])
        .reset_index(drop=True)
    )

    return news


def choose_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def encode_articles(
    news: pd.DataFrame,
    model_name: str,
    batch_size: int,
) -> np.ndarray:
    device = choose_device()

    print(f"Loading encoder: {model_name}")
    print(f"Using device: {device}")
    print(f"Encoding {len(news):,} articles")

    model = SentenceTransformer(
        model_name,
        device=device,
    )

    embeddings = model.encode(
        news["title"].tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings.astype(np.float32)


def save_artifacts(
    news: pd.DataFrame,
    embeddings: np.ndarray,
    output_dir: Path,
    model_name: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(
        output_dir / "article_embeddings.npy",
        embeddings,
    )

    with open(
        output_dir / "article_ids.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(news["news_id"].tolist(), f)

    metadata = {
        "model_name": model_name,
        "text_field": "title",
        "num_articles": len(news),
        "embedding_dim": int(embeddings.shape[1]),
        "normalized": True,
    }

    with open(
        output_dir / "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--news",
        type=Path,
        default=Path("data/raw/train/news.tsv"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/embeddings"),
    )

    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    args = parser.parse_args()

    news = process_news(args.news)

    embeddings = encode_articles(
        news=news,
        model_name=args.model,
        batch_size=args.batch_size,
    )

    save_artifacts(
        news=news,
        embeddings=embeddings,
        output_dir=args.output_dir,
        model_name=args.model,
    )

    print()
    print("Article embedding generation complete.")
    print(f"Articles: {embeddings.shape[0]:,}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Shape: {embeddings.shape}")
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()