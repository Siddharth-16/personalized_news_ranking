import argparse
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

from src.data.load import load_behaviors, load_news
from src.data.parse import parse_impressions
from src.data.split import chronological_split
from src.ranking.baselines import build_article_category_map
from src.ranking.features import (
    FEATURE_NAMES,
    build_article_subcategory_map,
    build_candidate_features,
    build_user_ranking_context,
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

METADATA_PATH = Path(
    "artifacts/models/ranker_metadata.json"
)

VALIDATION_DATE = "2019-11-14"
RANDOM_SEED = 42


def sample_training_indices(
    labels: list[int],
    negative_ratio: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Keep every positive example and sample at most
    negative_ratio negatives per positive.
    """

    labels_array = np.asarray(
        labels,
        dtype=np.int8,
    )

    positive_indices = np.flatnonzero(
        labels_array == 1
    )

    negative_indices = np.flatnonzero(
        labels_array == 0
    )

    if len(positive_indices) == 0:
        return np.array([], dtype=int)

    max_negatives = (
        negative_ratio * len(positive_indices)
    )

    number_of_negatives = min(
        max_negatives,
        len(negative_indices),
    )

    if number_of_negatives > 0:
        sampled_negatives = rng.choice(
            negative_indices,
            size=number_of_negatives,
            replace=False,
        )

        selected = np.concatenate(
            [
                positive_indices,
                sampled_negatives,
            ]
        )
    else:
        selected = positive_indices.copy()

    rng.shuffle(selected)

    return selected


def update_popularity(
    popularity: Counter,
    candidates: list[str],
    labels: list[int],
) -> None:
    """
    Update historical popularity after an impression
    has occurred.
    """

    for news_id, label in zip(
        candidates,
        labels,
    ):
        if label == 1:
            popularity[news_id] += 1


def build_training_dataset(
    train_behaviors,
    article_embedding_map,
    article_categories,
    article_subcategories,
    negative_ratio: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Build leakage-safe reranker training examples.

    Popularity features at time t contain clicks strictly
    from timestamps earlier than t.
    """

    rng = np.random.default_rng(
        random_seed
    )

    popularity = Counter()

    feature_rows = []
    target_values = []

    impressions_used = 0
    no_usable_history = 0
    no_positive_candidates = 0
    missing_candidate_features = 0

    # Grouping by timestamp prevents impressions occurring
    # at the same timestamp from leaking clicks into each
    # other's popularity feature.
    grouped = train_behaviors.groupby(
        "time",
        sort=True,
    )

    for _, timestamp_group in grouped:

        popularity_updates = []

        for row in timestamp_group.itertuples(
            index=False
        ):
            candidates, labels = (
                parse_impressions(
                    row.impressions
                )
            )

            # These clicks become visible only after every
            # impression at this timestamp is processed.
            popularity_updates.append(
                (candidates, labels)
            )

            selected_indices = (
                sample_training_indices(
                    labels,
                    negative_ratio=negative_ratio,
                    rng=rng,
                )
            )

            if len(selected_indices) == 0:
                no_positive_candidates += 1
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

            added_for_impression = 0

            for index in selected_indices:

                news_id = candidates[index]

                features = (
                    build_candidate_features(
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
                )

                if features is None:
                    missing_candidate_features += 1
                    continue

                feature_rows.append(
                    features
                )

                target_values.append(
                    labels[index]
                )

                added_for_impression += 1

            if added_for_impression > 0:
                impressions_used += 1

        # Only now do clicks at this timestamp become
        # available to later timestamps.
        for candidates, labels in popularity_updates:

            update_popularity(
                popularity,
                candidates,
                labels,
            )

    if not feature_rows:
        raise RuntimeError(
            "No ranker training examples were generated."
        )

    X = np.vstack(
        feature_rows
    ).astype(
        np.float32,
        copy=False,
    )

    y = np.asarray(
        target_values,
        dtype=np.int8,
    )

    if len(np.unique(y)) < 2:
        raise RuntimeError(
            "Training data must contain both positive "
            "and negative examples."
        )

    stats = {
        "examples": int(len(y)),
        "positive_examples": int(
            np.sum(y == 1)
        ),
        "negative_examples": int(
            np.sum(y == 0)
        ),
        "impressions_used": int(
            impressions_used
        ),
        "no_usable_history": int(
            no_usable_history
        ),
        "no_positive_candidates": int(
            no_positive_candidates
        ),
        "missing_candidate_features": int(
            missing_candidate_features
        ),
    }

    return X, y, stats


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional number of training impressions "
            "for smoke testing."
        ),
    )

    parser.add_argument(
        "--negative-ratio",
        type=int,
        default=4,
        help=(
            "Maximum sampled negatives per positive "
            "training example."
        ),
    )

    args = parser.parse_args()

    if args.negative_ratio <= 0:
        raise ValueError(
            "--negative-ratio must be greater than 0."
        )

    print("=" * 70)
    print(
        "PERSONALIZED NEWS RANKING ENGINE"
    )
    print("=" * 70)

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------

    print("\nLoading behaviors and news...")

    behaviors = load_behaviors(
        BEHAVIORS_PATH
    )

    news = load_news(
        NEWS_PATH
    )

    train, _ = chronological_split(
        behaviors,
        validation_date=VALIDATION_DATE,
    )

    if args.limit is not None:
        train = train.head(
            args.limit
        ).copy()

    print(
        f"Training impressions: "
        f"{len(train):,}"
    )

    # --------------------------------------------------
    # Load article representations
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

    print(
        f"Article embeddings: "
        f"{len(article_embedding_map):,}"
    )

    # --------------------------------------------------
    # Article metadata
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Generate training examples
    # --------------------------------------------------

    print(
        "\nBuilding leakage-safe training dataset..."
    )

    X, y, dataset_stats = (
        build_training_dataset(
            train_behaviors=train,
            article_embedding_map=(
                article_embedding_map
            ),
            article_categories=(
                article_categories
            ),
            article_subcategories=(
                article_subcategories
            ),
            negative_ratio=(
                args.negative_ratio
            ),
            random_seed=RANDOM_SEED,
        )
    )

    print(
        f"\nTraining examples:  "
        f"{len(y):,}"
    )

    print(
        f"Positive examples:  "
        f"{dataset_stats['positive_examples']:,}"
    )

    print(
        f"Negative examples:  "
        f"{dataset_stats['negative_examples']:,}"
    )

    print(
        f"Impressions used:   "
        f"{dataset_stats['impressions_used']:,}"
    )

    print(
        f"No usable history:  "
        f"{dataset_stats['no_usable_history']:,}"
    )

    print(
        f"Feature matrix:      "
        f"{X.shape}"
    )

    # --------------------------------------------------
    # Train reranker
    # --------------------------------------------------

    print("\nTraining reranker...")

    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.08,
        max_iter=120,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=RANDOM_SEED,
    )

    model.fit(
        X,
        y,
    )

    print("Ranker training complete.")

    # --------------------------------------------------
    # Save model + metadata
    # --------------------------------------------------

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    metadata = {
        "model": (
            "HistGradientBoostingClassifier"
        ),
        "features": FEATURE_NAMES,
        "feature_count": len(
            FEATURE_NAMES
        ),
        "negative_ratio": (
            args.negative_ratio
        ),
        "random_seed": RANDOM_SEED,
        "validation_cutoff": (
            VALIDATION_DATE
        ),
        "training_impressions": (
            len(train)
        ),
        "dataset_stats": dataset_stats,
        "scikit_learn_version": (
            sklearn.__version__
        ),
    }

    with METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print(
        f"\nSaved model to: "
        f"{MODEL_PATH}"
    )

    print(
        f"Saved metadata to: "
        f"{METADATA_PATH}"
    )


if __name__ == "__main__":
    main()