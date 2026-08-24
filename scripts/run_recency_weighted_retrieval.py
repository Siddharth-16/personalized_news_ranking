import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.data.load import load_behaviors
from src.data.parse import parse_history, parse_impressions
from src.data.split import chronological_split
from src.retrieval.faiss_retriever import FaissArticleRetriever
from src.retrieval.user_embeddings import (
    build_article_embedding_map,
    build_user_embedding,
)


BEHAVIORS_PATH = Path(
    "data/raw/train/behaviors.tsv"
)

EMBEDDINGS_DIR = Path(
    "artifacts/embeddings"
)

OUTPUT_PATH = Path(
    "artifacts/metrics/"
    "recency_weighted_retrieval.json"
)

VALIDATION_DATE = "2019-11-14"

K_VALUES = [50, 100]


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

        observed.update(
            candidates
        )

    return observed


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:

    if not relevant_ids:
        return 0.0

    return (
        len(
            set(retrieved_ids[:k])
            & relevant_ids
        )
        / len(relevant_ids)
    )


def build_recency_weighted_embedding(
    history,
    article_embedding_map,
    half_life_clicks: float,
) -> np.ndarray | None:
    """
    Build one user vector where recent clicks
    receive more weight.

    Weight:
        0.5 ** (age / half_life_clicks)

    The newest click has age 0 and weight 1.
    A click 10 positions back has weight 0.5
    when half_life_clicks=10.
    """

    history_ids = parse_history(
        history
    )

    if not history_ids:
        return None

    embeddings = []
    weights = []

    history_length = len(
        history_ids
    )

    for position, news_id in enumerate(
        history_ids
    ):
        embedding = (
            article_embedding_map.get(
                news_id
            )
        )

        if embedding is None:
            continue

        age = (
            history_length
            - 1
            - position
        )

        weight = (
            0.5
            ** (
                age
                / half_life_clicks
            )
        )

        embeddings.append(
            embedding
        )

        weights.append(
            weight
        )

    if not embeddings:
        return None

    matrix = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    weights_array = np.asarray(
        weights,
        dtype=np.float32,
    )

    user_embedding = np.average(
        matrix,
        axis=0,
        weights=weights_array,
    )

    norm = np.linalg.norm(
        user_embedding
    )

    if norm == 0:
        return None

    return (
        user_embedding / norm
    ).astype(
        np.float32
    )


def summarize_latency(
    values: list[float],
) -> dict[str, float]:

    values = np.asarray(
        values,
        dtype=float,
    )

    return {
        "mean_ms": float(
            np.mean(values)
        ),
        "p50_ms": float(
            np.percentile(
                values,
                50,
            )
        ),
        "p95_ms": float(
            np.percentile(
                values,
                95,
            )
        ),
        "p99_ms": float(
            np.percentile(
                values,
                99,
            )
        ),
    }


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
        "--half-life-clicks",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
    )

    args = parser.parse_args()

    if args.half_life_clicks <= 0:
        raise ValueError(
            "--half-life-clicks must be positive."
        )

    print("=" * 70)

    print(
        "PERSONALIZED NEWS RANKING ENGINE "
        "— RECENCY-WEIGHTED RETRIEVAL"
    )

    print("=" * 70)

    # --------------------------------------------------
    # Load data
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
    # Frozen embeddings
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

        article_ids = json.load(
            file
        )

    article_embedding_map = (
        build_article_embedding_map(
            article_ids,
            embeddings,
        )
    )

    # --------------------------------------------------
    # Same warm retrieval corpus
    # --------------------------------------------------

    train_observed = (
        collect_observed_articles(
            train
        )
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

    print(
        f"Recency half-life:       "
        f"{args.half_life_clicks:g} clicks"
    )

    # --------------------------------------------------
    # Evaluation containers
    # --------------------------------------------------

    baseline_recalls = {
        k: []
        for k in K_VALUES
    }

    weighted_recalls = {
        k: []
        for k in K_VALUES
    }

    baseline_latencies = []
    weighted_latencies = []

    evaluated = 0
    no_usable_history = 0
    cold_only = 0

    # --------------------------------------------------
    # Validation loop
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
            in zip(
                candidates,
                labels,
            )
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

        baseline_embedding = (
            build_user_embedding(
                row.history,
                article_embedding_map,
            )
        )

        weighted_embedding = (
            build_recency_weighted_embedding(
                row.history,
                article_embedding_map,
                half_life_clicks=(
                    args.half_life_clicks
                ),
            )
        )

        if (
            baseline_embedding is None
            or weighted_embedding is None
        ):
            no_usable_history += 1
            continue

        history_ids = set(
            parse_history(
                row.history
            )
        )

        # ----------------------------------------------
        # Baseline mean pooling
        # ----------------------------------------------

        start = time.perf_counter()

        baseline_ids, _ = (
            retriever.retrieve(
                baseline_embedding,
                k=100,
                exclude_ids=history_ids,
            )
        )

        baseline_latencies.append(
            (
                time.perf_counter()
                - start
            )
            * 1000
        )

        # ----------------------------------------------
        # Recency-weighted pooling
        # ----------------------------------------------

        start = time.perf_counter()

        weighted_ids, _ = (
            retriever.retrieve(
                weighted_embedding,
                k=100,
                exclude_ids=history_ids,
            )
        )

        weighted_latencies.append(
            (
                time.perf_counter()
                - start
            )
            * 1000
        )

        for k in K_VALUES:

            baseline_recalls[k].append(
                recall_at_k(
                    baseline_ids,
                    warm_relevant_ids,
                    k,
                )
            )

            weighted_recalls[k].append(
                recall_at_k(
                    weighted_ids,
                    warm_relevant_ids,
                    k,
                )
            )

        evaluated += 1

    # --------------------------------------------------
    # Aggregate
    # --------------------------------------------------

    baseline = {
        f"recall@{k}": float(
            np.mean(
                baseline_recalls[k]
            )
        )
        for k in K_VALUES
    }

    weighted = {
        f"recall@{k}": float(
            np.mean(
                weighted_recalls[k]
            )
        )
        for k in K_VALUES
    }

    baseline_latency = (
        summarize_latency(
            baseline_latencies
        )
    )

    weighted_latency = (
        summarize_latency(
            weighted_latencies
        )
    )

    # --------------------------------------------------
    # Print
    # --------------------------------------------------

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
        f"{cold_only:,}"
    )

    print("\n" + "=" * 70)
    print("RETRIEVAL RESULTS")
    print("=" * 70)

    print(
        f"\n{'Metric':<14}"
        f"{'Mean Pool':>14}"
        f"{'Recency':>14}"
        f"{'Change':>14}"
    )

    print("-" * 56)

    for k in K_VALUES:

        metric = f"recall@{k}"

        old = baseline[
            metric
        ]

        new = weighted[
            metric
        ]

        change = (
            (new / old - 1.0) * 100
            if old > 0
            else 0.0
        )

        print(
            f"{metric:<14}"
            f"{old:>14.4f}"
            f"{new:>14.4f}"
            f"{change:>+13.1f}%"
        )

    print("\nRetrieval latency:")

    print(
        f"Mean-pool p95: "
        f"{baseline_latency['p95_ms']:.3f} ms"
    )

    print(
        f"Recency p95:   "
        f"{weighted_latency['p95_ms']:.3f} ms"
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    output = {
        "experiment": (
            "recency_weighted_user_pooling"
        ),
        "half_life_clicks": (
            args.half_life_clicks
        ),
        "warm_corpus_articles": (
            len(retrieval_article_ids)
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
        "mean_pooling": baseline,
        "recency_weighted": weighted,
        "mean_pooling_latency": (
            baseline_latency
        ),
        "recency_weighted_latency": (
            weighted_latency
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