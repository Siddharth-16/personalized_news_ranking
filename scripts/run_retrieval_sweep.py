import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.data.load import load_behaviors
from src.data.parse import parse_history, parse_impressions
from src.data.split import chronological_split
from src.ranking.baselines import build_popularity_counts
from src.retrieval.faiss_retriever import FaissArticleRetriever
from src.retrieval.user_embeddings import (
    build_article_embedding_map,
    build_user_embedding,
)


BEHAVIORS_PATH = Path("data/raw/train/behaviors.tsv")
EMBEDDINGS_DIR = Path("artifacts/embeddings")
OUTPUT_PATH = Path(
    "artifacts/metrics/retrieval_k_sweep.json"
)

VALIDATION_DATE = "2019-11-14"

DEFAULT_K_VALUES = [
    50,
    100,
    200,
    500,
    1000,
]


def collect_observed_articles(
    behaviors,
) -> set[str]:
    observed = set()

    for row in behaviors.itertuples(
        index=False
    ):
        observed.update(
            parse_history(row.history)
        )

        candidates, _ = parse_impressions(
            row.impressions
        )

        observed.update(candidates)

    return observed


def retrieval_recall(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    if not relevant_ids:
        return 0.0

    retrieved_at_k = set(
        retrieved_ids[:k]
    )

    return (
        len(
            retrieved_at_k
            & relevant_ids
        )
        / len(relevant_ids)
    )


def build_popularity_ranking(
    article_ids: list[str],
    popularity,
) -> list[str]:
    return sorted(
        article_ids,
        key=lambda news_id: (
            -popularity[news_id],
            news_id,
        ),
    )


def popularity_retrieve(
    popularity_ranking: list[str],
    exclude_ids: set[str],
    k: int,
) -> list[str]:
    results = []

    for news_id in popularity_ranking:
        if news_id in exclude_ids:
            continue

        results.append(news_id)

        if len(results) == k:
            break

    return results


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--behaviors",
        type=Path,
        default=BEHAVIORS_PATH,
    )

    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=EMBEDDINGS_DIR,
    )

    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=DEFAULT_K_VALUES,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
    )

    args = parser.parse_args()

    k_values = sorted(
        set(args.k_values)
    )

    if any(k <= 0 for k in k_values):
        raise ValueError(
            "All K values must be positive."
        )

    max_k = max(k_values)

    print("=" * 70)
    print(
        "PERSONALIZED NEWS RANKING ENGINE "
        "— RETRIEVAL K SWEEP"
    )
    print("=" * 70)

    # --------------------------------------------------
    # Load chronological split
    # --------------------------------------------------

    behaviors = load_behaviors(
        args.behaviors
    )

    train, validation = (
        chronological_split(
            behaviors,
            VALIDATION_DATE,
        )
    )

    print(
        f"\nTraining impressions:   "
        f"{len(train):,}"
    )

    print(
        f"Validation impressions: "
        f"{len(validation):,}"
    )

    # --------------------------------------------------
    # Load frozen article embeddings
    # --------------------------------------------------

    embeddings = np.load(
        args.embeddings_dir
        / "article_embeddings.npy"
    )

    with (
        args.embeddings_dir
        / "article_ids.json"
    ).open(
        "r",
        encoding="utf-8",
    ) as file:
        article_ids = json.load(file)

    article_embedding_map = (
        build_article_embedding_map(
            article_ids,
            embeddings,
        )
    )

    # --------------------------------------------------
    # Build same warm corpus 
    # --------------------------------------------------

    train_observed = (
        collect_observed_articles(train)
    )

    article_id_to_index = {
        news_id: index
        for index, news_id
        in enumerate(article_ids)
    }

    retrieval_article_ids = [
        news_id
        for news_id in article_ids
        if news_id in train_observed
    ]

    retrieval_embeddings = np.asarray(
        [
            embeddings[
                article_id_to_index[
                    news_id
                ]
            ]
            for news_id
            in retrieval_article_ids
        ],
        dtype=np.float32,
    )

    retriever = FaissArticleRetriever(
        retrieval_article_ids,
        retrieval_embeddings,
    )

    print(
        f"Warm retrieval corpus:  "
        f"{len(retrieval_article_ids):,}"
    )

    # --------------------------------------------------
    # Popularity comparison
    # --------------------------------------------------

    popularity = build_popularity_counts(
        train
    )

    popularity_ranking = (
        build_popularity_ranking(
            retrieval_article_ids,
            popularity,
        )
    )

    faiss_recalls = {
        k: []
        for k in k_values
    }

    popularity_recalls = {
        k: []
        for k in k_values
    }

    retrieval_latencies = []

    evaluated = 0
    no_usable_history = 0
    cold_only = 0

    # --------------------------------------------------
    # Validation evaluation
    # --------------------------------------------------

    for row in validation.itertuples(
        index=False
    ):
        candidates, labels = (
            parse_impressions(
                row.impressions
            )
        )

        relevant_ids = {
            news_id
            for news_id, label
            in zip(candidates, labels)
            if label == 1
        }

        warm_relevant_ids = {
            news_id
            for news_id in relevant_ids
            if news_id in train_observed
        }

        if not warm_relevant_ids:
            cold_only += 1
            continue

        user_embedding = (
            build_user_embedding(
                row.history,
                article_embedding_map,
            )
        )

        if user_embedding is None:
            no_usable_history += 1
            continue

        history_ids = set(
            parse_history(row.history)
        )

        # Retrieve ONCE at the largest K.
        start = time.perf_counter()

        retrieved_ids, _ = (
            retriever.retrieve(
                user_embedding,
                k=max_k,
                exclude_ids=history_ids,
            )
        )

        retrieval_latencies.append(
            (
                time.perf_counter()
                - start
            )
            * 1000
        )

        popularity_ids = (
            popularity_retrieve(
                popularity_ranking,
                exclude_ids=history_ids,
                k=max_k,
            )
        )

        for k in k_values:
            faiss_recalls[k].append(
                retrieval_recall(
                    retrieved_ids,
                    warm_relevant_ids,
                    k,
                )
            )

            popularity_recalls[k].append(
                retrieval_recall(
                    popularity_ids,
                    warm_relevant_ids,
                    k,
                )
            )

        evaluated += 1

    # --------------------------------------------------
    # Aggregate
    # --------------------------------------------------

    faiss_results = {
        f"recall@{k}": float(
            np.mean(
                faiss_recalls[k]
            )
        )
        for k in k_values
    }

    popularity_results = {
        f"recall@{k}": float(
            np.mean(
                popularity_recalls[k]
            )
        )
        for k in k_values
    }

    latency = np.asarray(
        retrieval_latencies,
        dtype=float,
    )

    latency_results = {
        "mean_ms": float(
            np.mean(latency)
        ),
        "p50_ms": float(
            np.percentile(
                latency,
                50,
            )
        ),
        "p95_ms": float(
            np.percentile(
                latency,
                95,
            )
        ),
        "p99_ms": float(
            np.percentile(
                latency,
                99,
            )
        ),
    }

    # --------------------------------------------------
    # Print
    # --------------------------------------------------

    print(
        f"\nEvaluated impressions:  "
        f"{evaluated:,}"
    )

    print(
        f"No usable history:      "
        f"{no_usable_history:,}"
    )

    print(
        f"Cold-only impressions:  "
        f"{cold_only:,}"
    )

    print("\n" + "=" * 70)
    print("RECALL@K SWEEP")
    print("=" * 70)

    print(
        f"\n{'K':>8}"
        f"{'Popularity':>16}"
        f"{'Embedding + FAISS':>20}"
        f"{'vs R@100':>14}"
    )

    print("-" * 58)

    baseline_r100 = (
        faiss_results.get(
            "recall@100"
        )
    )

    for k in k_values:
        faiss_value = (
            faiss_results[
                f"recall@{k}"
            ]
        )

        pop_value = (
            popularity_results[
                f"recall@{k}"
            ]
        )

        if (
            baseline_r100
            and baseline_r100 > 0
        ):
            relative = (
                faiss_value
                / baseline_r100
            )
        else:
            relative = float("nan")

        print(
            f"{k:>8}"
            f"{pop_value:>16.4f}"
            f"{faiss_value:>20.4f}"
            f"{relative:>13.2f}x"
        )

    print("\nMax-K retrieval latency:")

    print(
        f"Mean: "
        f"{latency_results['mean_ms']:.3f} ms"
    )

    print(
        f"p50:  "
        f"{latency_results['p50_ms']:.3f} ms"
    )

    print(
        f"p95:  "
        f"{latency_results['p95_ms']:.3f} ms"
    )

    print(
        f"p99:  "
        f"{latency_results['p99_ms']:.3f} ms"
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    output = {
        "experiment": (
            "frozen_embedding_retrieval_k_sweep"
        ),
        "k_values": k_values,
        "max_k": max_k,
        "warm_corpus_articles": len(
            retrieval_article_ids
        ),
        "evaluated_impressions": (
            evaluated
        ),
        "no_usable_history": (
            no_usable_history
        ),
        "cold_only_impressions": (
            cold_only
        ),
        "popularity": (
            popularity_results
        ),
        "embedding_faiss": (
            faiss_results
        ),
        "max_k_latency": (
            latency_results
        ),
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    print(
        f"\nSaved to: {args.output}"
    )


if __name__ == "__main__":
    main()