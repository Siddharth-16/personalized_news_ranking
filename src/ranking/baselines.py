from collections import Counter

import numpy as np
import pandas as pd

from src.data.parse import parse_impressions, parse_history


def build_popularity_counts(
    train_behaviors: pd.DataFrame,
) -> Counter:
    """
    Count article clicks using training impressions only.
    """

    popularity = Counter()

    for impressions in train_behaviors["impressions"]:
        candidates, labels = parse_impressions(impressions)

        for news_id, label in zip(candidates, labels):
            if label == 1:
                popularity[news_id] += 1

    return popularity


def popularity_score(
    news_id: str,
    popularity: Counter,
) -> int:
    """Return training-period click count for an article."""

    return popularity[news_id]


def _add_deterministic_tiebreak(
    candidates: list[str],
    primary_scores: list[float],
) -> np.ndarray:
    """
    Add a tiny deterministic article-ID-based value so equal
    primary scores do not inherit the logged candidate order.

    The adjustment is deliberately tiny and cannot change the
    ordering of distinct integer popularity counts.
    """

    if len(candidates) != len(primary_scores):
        raise ValueError(
            "candidates and primary_scores must have the same length."
        )

    if not candidates:
        return np.array([], dtype=float)

    sorted_ids = sorted(set(candidates))

    tie_priority = {
        news_id: len(sorted_ids) - index
        for index, news_id in enumerate(sorted_ids)
    }

    denominator = len(sorted_ids) + 1

    adjusted_scores = []

    for news_id, primary_score in zip(
        candidates,
        primary_scores,
    ):
        tie_break = (
            tie_priority[news_id] / denominator
        ) * 1e-6

        adjusted_scores.append(
            float(primary_score) + tie_break
        )

    return np.asarray(adjusted_scores)


def popularity_scores(
    candidates: list[str],
    popularity: Counter,
) -> np.ndarray:
    """
    Score candidate articles using training-period popularity.
    """

    raw_scores = [
        popularity_score(news_id, popularity)
        for news_id in candidates
    ]

    return _add_deterministic_tiebreak(
        candidates,
        raw_scores,
    )


def build_article_category_map(
    news: pd.DataFrame,
) -> dict[str, str]:
    """
    Map each news article ID to its category.
    """

    return dict(
        zip(
            news["news_id"],
            news["category"],
        )
    )


def build_category_profile(
    history: list[str],
    article_categories: dict[str, str],
) -> dict[str, float]:
    """
    Build a normalized category-affinity profile from
    the user's click history.
    """

    category_counts = Counter()

    for news_id in history:
        category = article_categories.get(news_id)

        if category is not None:
            category_counts[category] += 1

    total = sum(category_counts.values())

    if total == 0:
        return {}

    return {
        category: count / total
        for category, count in category_counts.items()
    }


def category_affinity_scores(
    history: list[str],
    candidates: list[str],
    article_categories: dict[str, str],
) -> np.ndarray:
    """
    Score candidates according to the user's historical
    category preferences.
    """

    profile = build_category_profile(
        history,
        article_categories,
    )

    raw_scores = []

    for news_id in candidates:
        category = article_categories.get(news_id)

        score = profile.get(category, 0.0)

        raw_scores.append(score)

    return _add_deterministic_tiebreak(
        candidates,
        raw_scores,
    )