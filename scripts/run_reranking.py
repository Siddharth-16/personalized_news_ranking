import json
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

from src.data.load import load_behaviors, load_news
from src.data.parse import (
    parse_history,
    parse_impressions,
)
from src.data.split import chronological_split
from src.evaluation.metrics import (
    mrr,
    ndcg_at_k,
)
from src.ranking.baselines import (
    build_article_category_map,
    build_popularity_counts,
)
from src.ranking.features import (
    build_article_subcategory_map,
    build_candidate_features,
    build_user_ranking_context,
)
from src.retrieval.faiss_retriever import (
    FaissArticleRetriever,
)
from src.retrieval.user_embeddings import (
    build_article_embedding_map,
)


BEHAVIORS_PATH = Path(
    "data/raw/train/behaviors.tsv"
)

NEWS_PATH = Path(
    "data/raw/train/news.tsv"
)

EMBEDDINGS_PATH = Path(
    "artifacts/embeddings/article_embeddings.npy"
)

ARTICLE_IDS_PATH = Path(
    "artifacts/embeddings/article_ids.json"
)

MODEL_PATH = Path(
    "artifacts/models/ranker.joblib"
)

OUTPUT_PATH = Path(
    "artifacts/metrics/reranking.json"
)

VALIDATION_DATE = "2019-11-14"


def collect_observed_articles(
    behaviors,
) -> set[str]:
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


def retrieval_recall(
    retrieved_ids: list[str],
    relevant_ids: set[str],
) -> float:

    if not relevant_ids:
        return 0.0

    hits = len(
        set(retrieved_ids).intersection(
            relevant_ids
        )
    )

    return hits / len(relevant_ids)


def percentile_summary(
    values: list[float],
) -> dict[str, float]:

    values_array = np.asarray(
        values,
        dtype=float,
    )

    return {
        "mean": float(
            np.mean(values_array)
        ),
        "p50": float(
            np.percentile(
                values_array,
                50,
            )
        ),
        "p95": float(
            np.percentile(
                values_array,
                95,
            )
        ),
        "p99": float(
            np.percentile(
                values_array,
                99,
            )
        ),
    }


def main() -> None:

    print("=" * 70)
    print(
        "PERSONALIZED NEWS RANKING ENGINE"
    )
    print("=" * 70)

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------

    print("\nLoading behaviors and news...")

    behaviors = load_behaviors(
        BEHAVIORS_PATH
    )

    news = load_news(
        NEWS_PATH
    )

    train, validation = chronological_split(
        behaviors,
        validation_date=VALIDATION_DATE,
    )

    print(
        f"Training impressions:   "
        f"{len(train):,}"
    )

    print(
        f"Validation impressions: "
        f"{len(validation):,}"
    )

    # --------------------------------------------------
    # Load article embeddings
    # --------------------------------------------------

    print("\nLoading article embeddings...")

    article_embeddings = np.load(
        EMBEDDINGS_PATH
    )

    with ARTICLE_IDS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        article_ids = json.load(file)

    article_embedding_map = (
        build_article_embedding_map(
            article_ids,
            article_embeddings,
        )
    )

    # --------------------------------------------------
    # Build warm article corpus
    # --------------------------------------------------

    print(
        "\nBuilding training-observed "
        "retrieval corpus..."
    )

    train_article_ids = (
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
        if news_id in train_article_ids
    ]

    retrieval_embeddings = np.asarray(
        [
            article_embeddings[
                article_id_to_index[news_id]
            ]
            for news_id
            in retrieval_article_ids
        ],
        dtype=np.float32,
    )

    print(
        f"Retrieval corpus: "
        f"{len(retrieval_article_ids):,}"
    )

    retriever = FaissArticleRetriever(
        retrieval_article_ids,
        retrieval_embeddings,
    )

    # --------------------------------------------------
    # Ranking resources
    # --------------------------------------------------

    print("\nLoading reranker...")

    model = joblib.load(
        MODEL_PATH
    )

    article_categories = (
        build_article_category_map(
            news
        )
    )

    article_subcategories = (
        build_article_subcategory_map(
            news
        )
    )

    # Validation happens entirely after training,
    # so training-period clicks are safe historical
    # popularity features.
    popularity = build_popularity_counts(
        train
    )

    # --------------------------------------------------
    # Metric accumulators
    # --------------------------------------------------

    faiss_mrr = []
    faiss_ndcg_5 = []
    faiss_ndcg_10 = []

    reranker_mrr = []
    reranker_ndcg_5 = []
    reranker_ndcg_10 = []

    # Conditional metrics isolate ranking quality when
    # retrieval successfully finds a relevant article.
    conditional_faiss_mrr = []
    conditional_faiss_ndcg_5 = []
    conditional_faiss_ndcg_10 = []

    conditional_reranker_mrr = []
    conditional_reranker_ndcg_5 = []
    conditional_reranker_ndcg_10 = []

    retrieval_recall_100 = []

    retrieval_latencies_ms = []
    ranking_latencies_ms = []

    no_usable_history = 0
    cold_only_impressions = 0
    retrieval_hit_impressions = 0
    evaluated = 0

    print("\nEvaluating two-stage ranking...")

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    for row in validation.itertuples(
        index=False
    ):

        candidate_ids, candidate_labels = (
            parse_impressions(
                row.impressions
            )
        )

        relevant_ids = {
            news_id
            for news_id, label
            in zip(
                candidate_ids,
                candidate_labels,
            )
            if label == 1
        }

        if not relevant_ids:
            continue

        warm_relevant_ids = {
            news_id
            for news_id in relevant_ids
            if news_id in train_article_ids
        }

        if not warm_relevant_ids:
            cold_only_impressions += 1
            continue

        context = build_user_ranking_context(
            row.history,
            article_embedding_map,
            article_categories,
            article_subcategories,
        )

        if context is None:
            no_usable_history += 1
            continue

        history_ids = set(
            parse_history(row.history)
        )

        # ----------------------------------------------
        # Stage 1: FAISS retrieval
        # ----------------------------------------------

        retrieval_start = (
            time.perf_counter()
        )

        retrieved_ids, faiss_scores = (
            retriever.retrieve(
                context.user_embedding,
                k=100,
                exclude_ids=history_ids,
            )
        )

        retrieval_elapsed = (
            time.perf_counter()
            - retrieval_start
        ) * 1000

        retrieval_latencies_ms.append(
            retrieval_elapsed
        )

        retrieved_labels = np.asarray(
            [
                int(
                    news_id
                    in warm_relevant_ids
                )
                for news_id
                in retrieved_ids
            ],
            dtype=np.int8,
        )

        faiss_scores_array = np.asarray(
            faiss_scores,
            dtype=float,
        )

        recall_100 = retrieval_recall(
            retrieved_ids,
            warm_relevant_ids,
        )

        retrieval_recall_100.append(
            recall_100
        )

        # ----------------------------------------------
        # Raw FAISS ranking metrics
        # ----------------------------------------------

        faiss_impression_mrr = mrr(
            retrieved_labels,
            faiss_scores_array,
        )

        faiss_impression_ndcg_5 = (
            ndcg_at_k(
                retrieved_labels,
                faiss_scores_array,
                k=5,
            )
        )

        faiss_impression_ndcg_10 = (
            ndcg_at_k(
                retrieved_labels,
                faiss_scores_array,
                k=10,
            )
        )

        faiss_mrr.append(
            faiss_impression_mrr
        )

        faiss_ndcg_5.append(
            faiss_impression_ndcg_5
        )

        faiss_ndcg_10.append(
            faiss_impression_ndcg_10
        )

        # ----------------------------------------------
        # Stage 2: learned reranking
        # ----------------------------------------------

        ranking_start = time.perf_counter()

        feature_rows = []

        for news_id in retrieved_ids:

            features = build_candidate_features(
                news_id=news_id,
                context=context,
                article_embedding_map=(
                    article_embedding_map
                ),
                article_categories=(
                    article_categories
                ),
                article_subcategories=(
                    article_subcategories
                ),
                popularity=popularity,
            )

            if features is None:
                raise RuntimeError(
                    "Missing features for retrieved "
                    f"article {news_id}."
                )

            feature_rows.append(
                features
            )

        X_candidates = np.vstack(
            feature_rows
        ).astype(
            np.float32,
            copy=False,
        )

        reranker_scores = (
            model.predict_proba(
                X_candidates
            )[:, 1]
        )

        ranking_elapsed = (
            time.perf_counter()
            - ranking_start
        ) * 1000

        ranking_latencies_ms.append(
            ranking_elapsed
        )

        reranker_impression_mrr = mrr(
            retrieved_labels,
            reranker_scores,
        )

        reranker_impression_ndcg_5 = (
            ndcg_at_k(
                retrieved_labels,
                reranker_scores,
                k=5,
            )
        )

        reranker_impression_ndcg_10 = (
            ndcg_at_k(
                retrieved_labels,
                reranker_scores,
                k=10,
            )
        )

        reranker_mrr.append(
            reranker_impression_mrr
        )

        reranker_ndcg_5.append(
            reranker_impression_ndcg_5
        )

        reranker_ndcg_10.append(
            reranker_impression_ndcg_10
        )

        # ----------------------------------------------
        # Conditional ranking metrics
        # ----------------------------------------------

        if np.any(
            retrieved_labels == 1
        ):
            retrieval_hit_impressions += 1

            conditional_faiss_mrr.append(
                faiss_impression_mrr
            )

            conditional_faiss_ndcg_5.append(
                faiss_impression_ndcg_5
            )

            conditional_faiss_ndcg_10.append(
                faiss_impression_ndcg_10
            )

            conditional_reranker_mrr.append(
                reranker_impression_mrr
            )

            conditional_reranker_ndcg_5.append(
                reranker_impression_ndcg_5
            )

            conditional_reranker_ndcg_10.append(
                reranker_impression_ndcg_10
            )

        evaluated += 1

    if evaluated == 0:
        raise RuntimeError(
            "No validation impressions evaluated."
        )

    # --------------------------------------------------
    # Aggregate
    # --------------------------------------------------

    faiss_metrics = {
        "mrr": float(
            np.mean(faiss_mrr)
        ),
        "ndcg@5": float(
            np.mean(faiss_ndcg_5)
        ),
        "ndcg@10": float(
            np.mean(faiss_ndcg_10)
        ),
    }

    reranker_metrics = {
        "mrr": float(
            np.mean(reranker_mrr)
        ),
        "ndcg@5": float(
            np.mean(reranker_ndcg_5)
        ),
        "ndcg@10": float(
            np.mean(reranker_ndcg_10)
        ),
    }

    conditional_faiss_metrics = {
        "mrr": float(
            np.mean(
                conditional_faiss_mrr
            )
        ),
        "ndcg@5": float(
            np.mean(
                conditional_faiss_ndcg_5
            )
        ),
        "ndcg@10": float(
            np.mean(
                conditional_faiss_ndcg_10
            )
        ),
    }

    conditional_reranker_metrics = {
        "mrr": float(
            np.mean(
                conditional_reranker_mrr
            )
        ),
        "ndcg@5": float(
            np.mean(
                conditional_reranker_ndcg_5
            )
        ),
        "ndcg@10": float(
            np.mean(
                conditional_reranker_ndcg_10
            )
        ),
    }

    results = {
        "dataset": "MINDsmall",
        "validation_date": VALIDATION_DATE,
        "retrieval_corpus_size": len(
            retrieval_article_ids
        ),
        "evaluated_impressions": evaluated,
        "no_usable_history": (
            no_usable_history
        ),
        "cold_only_impressions": (
            cold_only_impressions
        ),
        "retrieval_hit_impressions": (
            retrieval_hit_impressions
        ),
        "recall@100": float(
            np.mean(
                retrieval_recall_100
            )
        ),
        "end_to_end": {
            "faiss_similarity": (
                faiss_metrics
            ),
            "retrieval_plus_reranker": (
                reranker_metrics
            ),
        },
        "conditional_on_retrieval_hit": {
            "faiss_similarity": (
                conditional_faiss_metrics
            ),
            "retrieval_plus_reranker": (
                conditional_reranker_metrics
            ),
        },
        "latency_ms": {
            "retrieval": (
                percentile_summary(
                    retrieval_latencies_ms
                )
            ),
            "reranking": (
                percentile_summary(
                    ranking_latencies_ms
                )
            ),
        },
    }

    # --------------------------------------------------
    # Print
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("RANKING RESULTS")
    print("=" * 70)

    print(
        f"\nEvaluated impressions:     "
        f"{evaluated:,}"
    )

    print(
        f"Retrieval-hit impressions: "
        f"{retrieval_hit_impressions:,}"
    )

    print(
        f"Cold-only impressions:     "
        f"{cold_only_impressions:,}"
    )

    print(
        f"No usable history:          "
        f"{no_usable_history:,}"
    )

    print(
        f"Recall@100:                 "
        f"{np.mean(retrieval_recall_100):.4f}"
    )

    print("\nEND-TO-END METRICS")

    print(
        f"\n{'Approach':<28}"
        f"{'MRR':>10}"
        f"{'nDCG@5':>12}"
        f"{'nDCG@10':>12}"
    )

    print("-" * 62)

    print(
        f"{'FAISS similarity':<28}"
        f"{faiss_metrics['mrr']:>10.4f}"
        f"{faiss_metrics['ndcg@5']:>12.4f}"
        f"{faiss_metrics['ndcg@10']:>12.4f}"
    )

    print(
        f"{'FAISS + reranker':<28}"
        f"{reranker_metrics['mrr']:>10.4f}"
        f"{reranker_metrics['ndcg@5']:>12.4f}"
        f"{reranker_metrics['ndcg@10']:>12.4f}"
    )

    print(
        "\nCONDITIONAL ON RETRIEVAL HIT"
    )

    print(
        f"\n{'Approach':<28}"
        f"{'MRR':>10}"
        f"{'nDCG@5':>12}"
        f"{'nDCG@10':>12}"
    )

    print("-" * 62)

    print(
        f"{'FAISS similarity':<28}"
        f"{conditional_faiss_metrics['mrr']:>10.4f}"
        f"{conditional_faiss_metrics['ndcg@5']:>12.4f}"
        f"{conditional_faiss_metrics['ndcg@10']:>12.4f}"
    )

    print(
        f"{'FAISS + reranker':<28}"
        f"{conditional_reranker_metrics['mrr']:>10.4f}"
        f"{conditional_reranker_metrics['ndcg@5']:>12.4f}"
        f"{conditional_reranker_metrics['ndcg@10']:>12.4f}"
    )

    retrieval_latency = percentile_summary(
        retrieval_latencies_ms
    )

    reranking_latency = percentile_summary(
        ranking_latencies_ms
    )

    print("\nLATENCY")

    print(
        f"Retrieval p95: "
        f"{retrieval_latency['p95']:.3f} ms"
    )

    print(
        f"Reranking p95: "
        f"{reranking_latency['p95']:.3f} ms"
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    print(
        f"\nSaved results to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()