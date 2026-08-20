import json
from pathlib import Path

import numpy as np

from src.data.load import load_behaviors
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
    build_popularity_counts,
    popularity_scores,
)
from src.retrieval.user_embeddings import (
    build_article_embedding_map,
    build_user_embedding,
)


BEHAVIORS_PATH = Path(
    "data/raw/train/behaviors.tsv"
)

EMBEDDINGS_PATH = Path(
    "artifacts/embeddings/article_embeddings.npy"
)

ARTICLE_IDS_PATH = Path(
    "artifacts/embeddings/article_ids.json"
)

OUTPUT_PATH = Path(
    "artifacts/metrics/cold_start.json"
)

VALIDATION_DATE = "2019-11-14"


def collect_observed_articles(
    behaviors,
) -> set[str]:
    """
    Collect articles observable during the training period.

    An article is considered training-observed if it appears
    in either a user history or an impression candidate set.
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


def content_similarity_scores(
    candidate_ids: list[str],
    user_embedding: np.ndarray,
    article_embedding_map: dict[
        str,
        np.ndarray,
    ],
) -> np.ndarray:
    """
    Score candidate articles by cosine similarity
    to the user's content representation.

    Article and user embeddings are already normalized,
    so their inner product equals cosine similarity.
    """

    scores = []

    for news_id in candidate_ids:

        article_embedding = (
            article_embedding_map.get(
                news_id
            )
        )

        if article_embedding is None:
            raise RuntimeError(
                "Missing embedding for article "
                f"{news_id}."
            )

        scores.append(
            float(
                np.dot(
                    user_embedding,
                    article_embedding,
                )
            )
        )

    return np.asarray(
        scores,
        dtype=float,
    )


def main() -> None:

    print("=" * 70)
    print(
        "PERSONALIZED NEWS RANKING ENGINE"
    )
    print("=" * 70)

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

    print(
        f"Training impressions:   "
        f"{len(train):,}"
    )

    print(
        f"Validation impressions: "
        f"{len(validation):,}"
    )

    # --------------------------------------------------
    # Determine warm vs cold articles
    # --------------------------------------------------

    print(
        "\nCollecting training-observed articles..."
    )

    train_article_ids = (
        collect_observed_articles(
            train
        )
    )

    print(
        f"Training-observed articles: "
        f"{len(train_article_ids):,}"
    )

    # --------------------------------------------------
    # Load frozen article embeddings
    # --------------------------------------------------

    print(
        "\nLoading frozen article embeddings..."
    )

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

    print(
        f"Article embeddings: "
        f"{len(article_embedding_map):,}"
    )

    # --------------------------------------------------
    # Behavioral baseline
    # --------------------------------------------------

    print(
        "\nBuilding training-period popularity..."
    )

    popularity = build_popularity_counts(
        train
    )

    # --------------------------------------------------
    # Metric accumulators
    # --------------------------------------------------

    popularity_mrr_values = []
    popularity_ndcg_5_values = []
    popularity_ndcg_10_values = []

    content_mrr_values = []
    content_ndcg_5_values = []
    content_ndcg_10_values = []

    cold_candidate_counts = []

    impressions_with_cold_click = 0
    evaluated_impressions = 0

    no_usable_history = 0
    insufficient_cold_candidates = 0
    missing_embeddings = 0

    print(
        "\nEvaluating cold-start ranking..."
    )

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    for row in validation.itertuples(
        index=False
    ):

        candidate_ids, labels = (
            parse_impressions(
                row.impressions
            )
        )

        # ----------------------------------------------
        # Keep only articles that had never appeared
        # during the training period.
        # ----------------------------------------------

        cold_candidates = []
        cold_labels = []

        for news_id, label in zip(
            candidate_ids,
            labels,
        ):
            if news_id not in train_article_ids:

                cold_candidates.append(
                    news_id
                )

                cold_labels.append(
                    label
                )

        # This impression is not useful for a cold-start
        # experiment unless a cold article was clicked.
        if not any(cold_labels):
            continue

        impressions_with_cold_click += 1

        # With only one cold candidate, every scoring
        # method would rank it first, so the impression
        # provides no meaningful ranking comparison.
        if len(cold_candidates) < 2:
            insufficient_cold_candidates += 1
            continue

        if any(
            news_id
            not in article_embedding_map
            for news_id in cold_candidates
        ):
            missing_embeddings += 1
            continue

        user_embedding = build_user_embedding(
            row.history,
            article_embedding_map,
        )

        if user_embedding is None:
            no_usable_history += 1
            continue

        cold_candidate_counts.append(
            len(cold_candidates)
        )

        # ----------------------------------------------
        # Baseline 1:
        # training-click popularity
        #
        # Since these articles are cold, their primary
        # popularity should be zero. Existing deterministic
        # tie-breaking prevents logged candidate order from
        # deciding the result.
        # ----------------------------------------------

        pop_scores = popularity_scores(
            cold_candidates,
            popularity,
        )

        popularity_mrr_values.append(
            mrr(
                cold_labels,
                pop_scores,
            )
        )

        popularity_ndcg_5_values.append(
            ndcg_at_k(
                cold_labels,
                pop_scores,
                k=5,
            )
        )

        popularity_ndcg_10_values.append(
            ndcg_at_k(
                cold_labels,
                pop_scores,
                k=10,
            )
        )

        # ----------------------------------------------
        # Baseline 2:
        # content-aware semantic similarity
        # ----------------------------------------------

        content_scores = (
            content_similarity_scores(
                cold_candidates,
                user_embedding,
                article_embedding_map,
            )
        )

        content_mrr_values.append(
            mrr(
                cold_labels,
                content_scores,
            )
        )

        content_ndcg_5_values.append(
            ndcg_at_k(
                cold_labels,
                content_scores,
                k=5,
            )
        )

        content_ndcg_10_values.append(
            ndcg_at_k(
                cold_labels,
                content_scores,
                k=10,
            )
        )

        evaluated_impressions += 1

    if evaluated_impressions == 0:
        raise RuntimeError(
            "No cold-start impressions "
            "could be evaluated."
        )

    # --------------------------------------------------
    # Aggregate metrics
    # --------------------------------------------------

    popularity_metrics = {
        "mrr": float(
            np.mean(
                popularity_mrr_values
            )
        ),
        "ndcg@5": float(
            np.mean(
                popularity_ndcg_5_values
            )
        ),
        "ndcg@10": float(
            np.mean(
                popularity_ndcg_10_values
            )
        ),
    }

    content_metrics = {
        "mrr": float(
            np.mean(
                content_mrr_values
            )
        ),
        "ndcg@5": float(
            np.mean(
                content_ndcg_5_values
            )
        ),
        "ndcg@10": float(
            np.mean(
                content_ndcg_10_values
            )
        ),
    }

    mean_cold_candidates = float(
        np.mean(
            cold_candidate_counts
        )
    )

    median_cold_candidates = float(
        np.median(
            cold_candidate_counts
        )
    )

    # --------------------------------------------------
    # Print
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("COLD-START RESULTS")
    print("=" * 70)

    print(
        f"\nImpressions with cold click: "
        f"{impressions_with_cold_click:,}"
    )

    print(
        f"Evaluated impressions:       "
        f"{evaluated_impressions:,}"
    )

    print(
        f"No usable history:           "
        f"{no_usable_history:,}"
    )

    print(
        f"Single cold candidate:       "
        f"{insufficient_cold_candidates:,}"
    )

    print(
        f"Missing embeddings:          "
        f"{missing_embeddings:,}"
    )

    print(
        f"Mean cold candidates:        "
        f"{mean_cold_candidates:.2f}"
    )

    print(
        f"Median cold candidates:      "
        f"{median_cold_candidates:.0f}"
    )

    print(
        f"\n{'Approach':<28}"
        f"{'MRR':>10}"
        f"{'nDCG@5':>12}"
        f"{'nDCG@10':>12}"
    )

    print("-" * 62)

    print(
        f"{'Popularity':<28}"
        f"{popularity_metrics['mrr']:>10.4f}"
        f"{popularity_metrics['ndcg@5']:>12.4f}"
        f"{popularity_metrics['ndcg@10']:>12.4f}"
    )

    print(
        f"{'Content similarity':<28}"
        f"{content_metrics['mrr']:>10.4f}"
        f"{content_metrics['ndcg@5']:>12.4f}"
        f"{content_metrics['ndcg@10']:>12.4f}"
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    results = {
        "dataset": "MINDsmall",
        "validation_date": VALIDATION_DATE,

        "cold_start_definition": (
            "candidate article not observed in any "
            "training-period history or impression"
        ),

        "evaluation": {
            "impressions_with_cold_click": (
                impressions_with_cold_click
            ),
            "evaluated_impressions": (
                evaluated_impressions
            ),
            "no_usable_history": (
                no_usable_history
            ),
            "insufficient_cold_candidates": (
                insufficient_cold_candidates
            ),
            "missing_embeddings": (
                missing_embeddings
            ),
            "mean_cold_candidates": (
                mean_cold_candidates
            ),
            "median_cold_candidates": (
                median_cold_candidates
            ),
        },

        "models": {
            "training_click_popularity": (
                popularity_metrics
            ),
            "frozen_content_similarity": (
                content_metrics
            ),
        },
    }

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