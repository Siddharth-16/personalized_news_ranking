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

EMBEDDINGS_DIR = Path(
    "artifacts/embeddings"
)

OUTPUT_PATH = Path(
    "artifacts/metrics/"
    "multi_interest_retrieval.json"
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
            parse_history(
                row.history
            )
        )

        candidates, _ = (
            parse_impressions(
                row.impressions
            )
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

    retrieved = set(
        retrieved_ids[:k]
    )

    return (
        len(
            retrieved
            & relevant_ids
        )
        / len(relevant_ids)
    )


def build_multi_interest_queries(
    history,
    article_embedding_map,
    recent_clicks: int,
) -> list[np.ndarray]:
    """
    Query 1:
        mean-pooled full user history.

    Remaining queries:
        embeddings of the most recent clicked
        articles available in the history.
    """

    mean_embedding = (
        build_user_embedding(
            history,
            article_embedding_map,
        )
    )

    if mean_embedding is None:
        return []

    queries = [
        mean_embedding
    ]

    history_ids = parse_history(
        history
    )

    # History is chronological, so traverse
    # backwards to find recent valid articles.
    recent_embeddings = []

    for news_id in reversed(
        history_ids
    ):
        embedding = (
            article_embedding_map.get(
                news_id
            )
        )

        if embedding is None:
            continue

        recent_embeddings.append(
            embedding
        )

        if (
            len(recent_embeddings)
            == recent_clicks
        ):
            break

    queries.extend(
        recent_embeddings
    )

    return queries


def multi_interest_retrieve(
    retriever: FaissArticleRetriever,
    queries: list[np.ndarray],
    exclude_ids: set[str],
    final_k: int,
    per_query_k: int,
) -> list[str]:
    """
    Retrieve separately for each user-interest query.

    An article's fused score is the maximum cosine
    similarity it achieved under any query.
    """

    fused_scores = {}

    for query in queries:
        ids, scores = retriever.retrieve(
            query,
            k=per_query_k,
            exclude_ids=exclude_ids,
        )

        for news_id, score in zip(
            ids,
            scores,
        ):
            previous = fused_scores.get(
                news_id
            )

            if (
                previous is None
                or score > previous
            ):
                fused_scores[
                    news_id
                ] = score

    ranked = sorted(
        fused_scores.items(),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    )

    return [
        news_id
        for news_id, _
        in ranked[:final_k]
    ]


def summarize_latency(
    values: list[float],
) -> dict[str, float]:
    array = np.asarray(
        values,
        dtype=float,
    )

    return {
        "mean_ms": float(
            np.mean(array)
        ),
        "p50_ms": float(
            np.percentile(
                array,
                50,
            )
        ),
        "p95_ms": float(
            np.percentile(
                array,
                95,
            )
        ),
        "p99_ms": float(
            np.percentile(
                array,
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
        "--recent-clicks",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--per-query-k",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
    )

    args = parser.parse_args()

    if args.recent_clicks <= 0:
        raise ValueError(
            "--recent-clicks must be positive."
        )

    if args.per_query_k < max(
        K_VALUES
    ):
        raise ValueError(
            "--per-query-k must be at least 100."
        )

    print("=" * 70)

    print(
        "PERSONALIZED NEWS RANKING ENGINE "
        "— MULTI-INTEREST RETRIEVAL"
    )

    print("=" * 70)

    # --------------------------------------------------
    # Data
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
    # Same warm corpus 
    # --------------------------------------------------

    train_observed = (
        collect_observed_articles(
            train
        )
    )

    article_index = {
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
                article_index[
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
        f"Recent interest queries:"
        f" {args.recent_clicks}"
    )

    print(
        "Queries per user:       "
        f"up to {args.recent_clicks + 1}"
    )

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    baseline_recalls = {
        k: []
        for k in K_VALUES
    }

    multi_recalls = {
        k: []
        for k in K_VALUES
    }

    baseline_latencies = []
    multi_latencies = []

    query_counts = []

    evaluated = 0
    no_usable_history = 0
    cold_only = 0

    # --------------------------------------------------
    # Evaluation
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
            for news_id
            in relevant_ids
            if news_id in train_observed
        }

        if not warm_relevant_ids:
            cold_only += 1
            continue

        mean_embedding = (
            build_user_embedding(
                row.history,
                article_embedding_map,
            )
        )

        if mean_embedding is None:
            no_usable_history += 1
            continue

        history_ids = set(
            parse_history(
                row.history
            )
        )

        # ----------------------------------------------
        # Baseline single-vector retrieval
        # ----------------------------------------------

        start = time.perf_counter()

        baseline_ids, _ = (
            retriever.retrieve(
                mean_embedding,
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
        # Multi-interest retrieval
        # ----------------------------------------------

        queries = (
            build_multi_interest_queries(
                row.history,
                article_embedding_map,
                recent_clicks=(
                    args.recent_clicks
                ),
            )
        )

        query_counts.append(
            len(queries)
        )

        start = time.perf_counter()

        multi_ids = (
            multi_interest_retrieve(
                retriever=retriever,
                queries=queries,
                exclude_ids=history_ids,
                final_k=100,
                per_query_k=(
                    args.per_query_k
                ),
            )
        )

        multi_latencies.append(
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

            multi_recalls[k].append(
                recall_at_k(
                    multi_ids,
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

    multi = {
        f"recall@{k}": float(
            np.mean(
                multi_recalls[k]
            )
        )
        for k in K_VALUES
    }

    baseline_latency = (
        summarize_latency(
            baseline_latencies
        )
    )

    multi_latency = (
        summarize_latency(
            multi_latencies
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

    print(
        f"Mean queries/user:     "
        f"{np.mean(query_counts):.2f}"
    )

    print("\n" + "=" * 70)
    print("RETRIEVAL RESULTS")
    print("=" * 70)

    print(
        f"\n{'Metric':<14}"
        f"{'Single Vector':>16}"
        f"{'Multi-Interest':>18}"
        f"{'Change':>14}"
    )

    print("-" * 62)

    for k in K_VALUES:
        metric = f"recall@{k}"

        old = baseline[
            metric
        ]

        new = multi[
            metric
        ]

        relative_change = (
            (new / old - 1.0) * 100
            if old > 0
            else 0.0
        )

        print(
            f"{metric:<14}"
            f"{old:>16.4f}"
            f"{new:>18.4f}"
            f"{relative_change:>+13.1f}%"
        )

    print("\nRetrieval latency:")

    print(
        f"Single-vector p95: "
        f"{baseline_latency['p95_ms']:.3f} ms"
    )

    print(
        f"Multi-interest p95: "
        f"{multi_latency['p95_ms']:.3f} ms"
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    output = {
        "experiment": (
            "multi_interest_retrieval"
        ),
        "recent_click_queries": (
            args.recent_clicks
        ),
        "includes_mean_query": True,
        "fusion": (
            "maximum_cosine_similarity"
        ),
        "per_query_k": (
            args.per_query_k
        ),
        "final_k": 100,
        "warm_corpus_articles": (
            len(
                retrieval_article_ids
            )
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
        "mean_queries_per_user": float(
            np.mean(
                query_counts
            )
        ),
        "single_vector": (
            baseline
        ),
        "multi_interest": (
            multi
        ),
        "single_vector_latency": (
            baseline_latency
        ),
        "multi_interest_latency": (
            multi_latency
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