import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.data.load import load_behaviors
from src.data.parse import (
    parse_history,
    parse_impressions,
)
from src.data.split import chronological_split
from src.ranking.baselines import build_popularity_counts
from src.retrieval.faiss_retriever import (
    FaissArticleRetriever,
)
from src.retrieval.user_embeddings import (
    build_article_embedding_map,
    build_user_embedding,
)


BEHAVIORS_PATH = Path(
    "data/raw/train/behaviors.tsv"
)

VALIDATION_DATE = "2019-11-14"


def retrieval_recall(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """
    Fraction of relevant clicked articles retrieved
    within the Top-K results.
    """

    if not relevant_ids:
        return 0.0

    retrieved_at_k = set(
        retrieved_ids[:k]
    )

    hits = len(
        retrieved_at_k.intersection(
            relevant_ids
        )
    )

    return hits / len(relevant_ids)


def build_global_popularity_ranking(
    article_ids: list[str],
    popularity,
) -> list[str]:
    """
    Rank the full article corpus by training-period clicks.

    Article ID provides deterministic ordering for ties.
    """

    return sorted(
        article_ids,
        key=lambda news_id: (
            -popularity[news_id],
            news_id,
        ),
    )


def popularity_retrieve(
    ranked_article_ids: list[str],
    exclude_ids: set[str],
    k: int,
) -> list[str]:
    results = []

    for news_id in ranked_article_ids:

        if news_id in exclude_ids:
            continue

        results.append(news_id)

        if len(results) == k:
            break

    return results


def collect_observed_articles(
    behaviors,
) -> set[str]:
    """
    Collect all articles observable during a behavior period.

    Includes articles appearing in user histories and
    impression candidate sets.
    """

    observed = set()

    for row in behaviors.itertuples(index=False):

        observed.update(
            parse_history(row.history)
        )

        candidates, _ = parse_impressions(
            row.impressions
        )

        observed.update(candidates)

    return observed


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=Path("artifacts/embeddings"),
        help=(
            "Directory containing article_embeddings.npy, "
            "article_ids.json, and metadata.json."
        ),
    )

    parser.add_argument(
        "--index-output",
        type=Path,
        default=Path(
            "artifacts/retrieval/article.index"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/metrics/retrieval.json"
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional number of validation impressions "
            "for a quick experiment."
        ),
    )

    args = parser.parse_args()

    embeddings_path = (
        args.embeddings_dir
        / "article_embeddings.npy"
    )

    article_ids_path = (
        args.embeddings_dir
        / "article_ids.json"
    )

    metadata_path = (
        args.embeddings_dir
        / "metadata.json"
    )

    print("=" * 70)
    print(
        "PERSONALIZED NEWS RANKING ENGINE"
    )
    print("=" * 70)

    # --------------------------------------------------
    # Load article representations
    # --------------------------------------------------

    print("\nLoading article embeddings...")

    article_embeddings = np.load(
        embeddings_path
    )

    with article_ids_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        article_ids = json.load(file)

    embedding_metadata = {}

    if metadata_path.exists():
        with metadata_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            embedding_metadata = json.load(file)

    print(
        f"Articles:            {len(article_ids):,}"
    )
    print(
        f"Embedding dimension: "
        f"{article_embeddings.shape[1]}"
    )

    print(
        f"Embeddings source:     "
        f"{args.embeddings_dir}"
    )

    if embedding_metadata:
        print(
            f"Encoder:               "
            f"{embedding_metadata.get('model_name')}"
        )

    article_embedding_map = (
        build_article_embedding_map(
            article_ids,
            article_embeddings,
        )
    )

    # --------------------------------------------------
    # Load behavior data
    # --------------------------------------------------

    print("\nLoading behaviors...")

    behaviors = load_behaviors(
        BEHAVIORS_PATH
    )

    train, validation = chronological_split(
        behaviors,
        validation_date=VALIDATION_DATE,
    )

    if args.limit is not None:
        validation = validation.head(
            args.limit
        )

    print(
        f"Training impressions:   {len(train):,}"
    )

    print(
        f"Validation impressions: {len(validation):,}"
    )

    # --------------------------------------------------
    # Build training-observed article corpus
    # --------------------------------------------------

    print(
        "\nCollecting articles observed during training..."
    )

    train_article_ids = collect_observed_articles(
        train
    )

    article_id_to_index = {
        news_id: index
        for index, news_id in enumerate(article_ids)
    }

    retrieval_article_ids = [
        news_id
        for news_id in article_ids
        if news_id in train_article_ids
    ]

    retrieval_embeddings = np.asarray(
        [
            article_embeddings[
                article_id_to_index[news_id]
            ]
            for news_id in retrieval_article_ids
        ],
        dtype=np.float32,
    )

    print(
        f"Training-observed articles: "
        f"{len(retrieval_article_ids):,}"
    )

    print(
        f"Excluded unseen articles:   "
        f"{len(article_ids) - len(retrieval_article_ids):,}"
    )

    # --------------------------------------------------
    # FAISS
    # --------------------------------------------------

    print("\nBuilding FAISS index...")

    retriever = FaissArticleRetriever(
        retrieval_article_ids,
        retrieval_embeddings,
    )

    retriever.save_index(
        args.index_output
    )

    print(
        f"FAISS vectors: {retriever.index.ntotal:,}"
    )

    print(
        f"Saved index to: {args.index_output}"
    )

    # --------------------------------------------------
    # Simple retrieval baseline
    # --------------------------------------------------

    print(
        "\nBuilding global popularity retrieval baseline..."
    )

    popularity = build_popularity_counts(
        train
    )

    popularity_ranking = (
        build_global_popularity_ranking(
            retrieval_article_ids,
            popularity,
        )
    )

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    embedding_recall_50 = []
    embedding_recall_100 = []

    popularity_recall_50 = []
    popularity_recall_100 = []

    retrieval_latencies_ms = []

    no_usable_history = 0
    no_positive_articles = 0
    cold_only_impressions = 0

    evaluated = 0

    print("\nEvaluating retrieval...")

    for row in validation.itertuples(
        index=False
    ):
        candidate_ids, labels = (
            parse_impressions(
                row.impressions
            )
        )

        relevant_ids = {
            news_id
            for news_id, label
            in zip(candidate_ids, labels)
            if label == 1
        }

        if not relevant_ids:
            no_positive_articles += 1
            continue

        warm_relevant_ids = {
            news_id
            for news_id in relevant_ids
            if news_id in train_article_ids
        }

        if not warm_relevant_ids:
            cold_only_impressions += 1
            continue

        user_embedding = build_user_embedding(
            row.history,
            article_embedding_map,
        )

        if user_embedding is None:
            no_usable_history += 1
            continue

        history_ids = set(
            parse_history(row.history)
        )

        # ----------------------------------------------
        # Embedding + FAISS retrieval
        # ----------------------------------------------

        start = time.perf_counter()

        retrieved_ids, _ = (
            retriever.retrieve(
                user_embedding,
                k=100,
                exclude_ids=history_ids,
            )
        )

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        retrieval_latencies_ms.append(
            elapsed_ms
        )

        embedding_recall_50.append(
            retrieval_recall(
                retrieved_ids,
                warm_relevant_ids,
                k=50,
            )
        )

        embedding_recall_100.append(
            retrieval_recall(
                retrieved_ids,
                warm_relevant_ids,
                k=100,
            )
        )

        # ----------------------------------------------
        # Global popularity retrieval
        # ----------------------------------------------

        popular_ids = popularity_retrieve(
            popularity_ranking,
            exclude_ids=history_ids,
            k=100,
        )

        popularity_recall_50.append(
            retrieval_recall(
                popular_ids,
                warm_relevant_ids,
                k=50,
            )
        )

        popularity_recall_100.append(
            retrieval_recall(
                popular_ids,
                warm_relevant_ids,
                k=100,
            )
        )

        evaluated += 1

    if evaluated == 0:
        raise RuntimeError(
            "No validation impressions could be evaluated."
        )

    # --------------------------------------------------
    # Aggregate results
    # --------------------------------------------------

    faiss_recall_50 = float(
        np.mean(embedding_recall_50)
    )

    faiss_recall_100 = float(
        np.mean(embedding_recall_100)
    )

    pop_recall_50 = float(
        np.mean(popularity_recall_50)
    )

    pop_recall_100 = float(
        np.mean(popularity_recall_100)
    )

    latencies = np.asarray(
        retrieval_latencies_ms
    )

    valid_for_retrieval = (
        evaluated + no_usable_history
    )

    coverage = (
        evaluated / valid_for_retrieval
        if valid_for_retrieval
        else 0.0
    )

    results = {
        "dataset": "MINDsmall",
        "embeddings": {
            "directory": str(
                args.embeddings_dir
            ),
            "model_name": (
                embedding_metadata.get(
                    "model_name"
                )
            ),
            "embedding_dim": int(
                article_embeddings.shape[1]
            ),
        },
        "split": {
            "strategy": "chronological",
            "validation_date": VALIDATION_DATE,
            "train_impressions": len(train),
            "validation_impressions": len(validation),
        },
        "corpus": {
            "all_articles": len(article_ids),
            "training_observed_articles": len(
                retrieval_article_ids
            ),
        },
        "evaluation": {
            "evaluated_impressions": evaluated,
            "no_usable_history": no_usable_history,
            "no_positive_articles": no_positive_articles,
            "cold_only_impressions": cold_only_impressions,
            "personalization_coverage": coverage,
        },
        "retrieval": {
            "embedding_faiss": {
                "index": "IndexFlatIP",
                "candidate_k": 100,
                "recall@50": faiss_recall_50,
                "recall@100": faiss_recall_100,
                "latency_ms": {
                    "mean": float(
                        np.mean(latencies)
                    ),
                    "p50": float(
                        np.percentile(
                            latencies,
                            50,
                        )
                    ),
                    "p95": float(
                        np.percentile(
                            latencies,
                            95,
                        )
                    ),
                    "p99": float(
                        np.percentile(
                            latencies,
                            99,
                        )
                    ),
                },
            },
            "global_popularity": {
                "candidate_k": 100,
                "recall@50": pop_recall_50,
                "recall@100": pop_recall_100,
            },
        },
    }

    # --------------------------------------------------
    # Display results
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("CANDIDATE RETRIEVAL RESULTS")
    print("=" * 70)

    print(
        f"\nEvaluated impressions: "
        f"{evaluated:,}"
    )

    print(
        f"No usable history:     "
        f"{no_usable_history:,}"
    )

    print(
        f"Cold-only impressions: "
        f"{cold_only_impressions:,}"
    )

    print(
        f"Coverage:              "
        f"{coverage:.2%}"
    )

    print("\n" + "-" * 70)

    print(
        f"{'Retriever':<25}"
        f"{'Recall@50':>15}"
        f"{'Recall@100':>15}"
    )

    print("-" * 55)

    print(
        f"{'Global popularity':<25}"
        f"{pop_recall_50:>15.4f}"
        f"{pop_recall_100:>15.4f}"
    )

    print(
        f"{'Embedding + FAISS':<25}"
        f"{faiss_recall_50:>15.4f}"
        f"{faiss_recall_100:>15.4f}"
    )

    print("\nFAISS retrieval latency")

    print(
        f"Mean: {np.mean(latencies):.3f} ms"
    )
    print(
        f"p50:  "
        f"{np.percentile(latencies, 50):.3f} ms"
    )
    print(
        f"p95:  "
        f"{np.percentile(latencies, 95):.3f} ms"
    )
    print(
        f"p99:  "
        f"{np.percentile(latencies, 99):.3f} ms"
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    print(
        f"\nSaved metrics to: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()