import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.data.load import load_news


DEFAULT_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

DEFAULT_OUTPUT = Path(
    "artifacts/embeddings_title_abstract"
)


def choose_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def process_news(
    path: Path,
):
    news = load_news(path)

    news = (
        news[
            [
                "news_id",
                "title",
                "abstract",
            ]
        ]
        .drop_duplicates(
            subset="news_id"
        )
        .dropna(
            subset=[
                "news_id",
                "title",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    news["abstract"] = (
        news["abstract"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    news["title"] = (
        news["title"]
        .astype(str)
        .str.strip()
    )

    # Preserve the title even when an article
    # does not have an abstract.
    news["text"] = np.where(
        news["abstract"].ne(""),
        (
            news["title"]
            + ". "
            + news["abstract"]
        ),
        news["title"],
    )

    return news


def encode_articles(
    news,
    model_name: str,
    batch_size: int,
) -> np.ndarray:

    device = choose_device()

    print(
        f"Using device: {device}"
    )

    print(
        f"Loading encoder: {model_name}"
    )

    model = SentenceTransformer(
        model_name,
        device=device,
    )

    embeddings = model.encode(
        news["text"].tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings.astype(
        np.float32
    )


def save_artifacts(
    news,
    embeddings: np.ndarray,
    output_dir: Path,
    model_name: str,
) -> None:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        output_dir
        / "article_embeddings.npy",
        embeddings,
    )

    with (
        output_dir
        / "article_ids.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            news[
                "news_id"
            ].tolist(),
            file,
        )

    metadata = {
        "model_name": model_name,
        "text_field": (
            "title + abstract"
        ),
        "num_articles": len(
            news
        ),
        "embedding_dim": int(
            embeddings.shape[1]
        ),
        "normalized": True,
    }

    with (
        output_dir
        / "metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--news",
        type=Path,
        default=Path(
            "data/raw/train/news.tsv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    args = parser.parse_args()

    news = process_news(
        args.news
    )

    print(
        f"Articles: {len(news):,}"
    )

    print(
        "Representation: "
        "title + abstract"
    )

    embeddings = encode_articles(
        news,
        model_name=args.model,
        batch_size=args.batch_size,
    )

    print(
        f"Embedding shape: "
        f"{embeddings.shape}"
    )

    save_artifacts(
        news,
        embeddings,
        output_dir=args.output_dir,
        model_name=args.model,
    )

    print(
        f"Saved to: "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()