from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from src.data.parse import parse_history
from src.ranking.baselines import build_category_profile
from src.retrieval.user_embeddings import build_user_embedding


FEATURE_NAMES = [
    "semantic_similarity",
    "category_affinity",
    "subcategory_affinity",
    "log_popularity",
    "log_history_length",
]


@dataclass(frozen=True)
class UserRankingContext:
    """
    User-level information that is constant across all
    candidate articles within one impression.
    """

    user_embedding: np.ndarray
    category_profile: dict[str, float]
    subcategory_profile: dict[str, float]
    history_length: int


def build_article_subcategory_map(
    news,
) -> dict[str, str]:
    """
    Map article IDs to subcategories.
    """

    return dict(
        zip(
            news["news_id"],
            news["subcategory"],
        )
    )


def build_subcategory_profile(
    history: list[str],
    article_subcategories: dict[str, str],
) -> dict[str, float]:
    """
    Build normalized subcategory preferences from
    a user's historical clicks.
    """

    counts: dict[str, int] = {}

    for news_id in history:
        subcategory = article_subcategories.get(
            news_id
        )

        if subcategory is None:
            continue

        counts[subcategory] = (
            counts.get(subcategory, 0) + 1
        )

    total = sum(counts.values())

    if total == 0:
        return {}

    return {
        subcategory: count / total
        for subcategory, count in counts.items()
    }


def build_user_ranking_context(
    history,
    article_embedding_map: dict[str, np.ndarray],
    article_categories: dict[str, str],
    article_subcategories: dict[str, str],
) -> UserRankingContext | None:
    """
    Build all user-level features once per impression.

    Returns None when no usable historical articles exist.
    """

    history_ids = parse_history(history)

    user_embedding = build_user_embedding(
        history,
        article_embedding_map,
    )

    if user_embedding is None:
        return None

    category_profile = build_category_profile(
        history_ids,
        article_categories,
    )

    subcategory_profile = (
        build_subcategory_profile(
            history_ids,
            article_subcategories,
        )
    )

    return UserRankingContext(
        user_embedding=user_embedding,
        category_profile=category_profile,
        subcategory_profile=subcategory_profile,
        history_length=len(history_ids),
    )


def build_candidate_features(
    news_id: str,
    context: UserRankingContext,
    article_embedding_map: dict[str, np.ndarray],
    article_categories: dict[str, str],
    article_subcategories: dict[str, str],
    popularity: Mapping[str, int],
) -> np.ndarray | None:
    """
    Build the feature vector used by the reranker.
    """

    article_embedding = (
        article_embedding_map.get(news_id)
    )

    if article_embedding is None:
        return None

    # Both user and article representations are normalized,
    # so the inner product equals cosine similarity.
    semantic_similarity = float(
        np.dot(
            context.user_embedding,
            article_embedding,
        )
    )

    category = article_categories.get(
        news_id
    )

    category_affinity = float(
        context.category_profile.get(
            category,
            0.0,
        )
    )

    subcategory = article_subcategories.get(
        news_id
    )

    subcategory_affinity = float(
        context.subcategory_profile.get(
            subcategory,
            0.0,
        )
    )

    log_popularity = float(
        np.log1p(
            popularity.get(news_id, 0)
        )
    )

    log_history_length = float(
        np.log1p(
            context.history_length
        )
    )

    return np.asarray(
        [
            semantic_similarity,
            category_affinity,
            subcategory_affinity,
            log_popularity,
            log_history_length,
        ],
        dtype=np.float32,
    )