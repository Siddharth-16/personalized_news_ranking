from collections import Counter

import numpy as np
import pytest

from src.ranking.features import (
    FEATURE_NAMES,
    build_candidate_features,
    build_user_ranking_context,
)


def test_user_ranking_context_contains_expected_profiles():
    article_embedding_map = {
        "N1": np.asarray(
            [1.0, 0.0],
            dtype=np.float32,
        ),
        "N2": np.asarray(
            [0.0, 1.0],
            dtype=np.float32,
        ),
    }

    article_categories = {
        "N1": "sports",
        "N2": "sports",
    }

    article_subcategories = {
        "N1": "football",
        "N2": "basketball",
    }

    context = build_user_ranking_context(
        "N1 N2",
        article_embedding_map,
        article_categories,
        article_subcategories,
    )

    assert context is not None

    assert (
        context.history_length
        == 2
    )

    assert (
        context.category_profile[
            "sports"
        ]
        == pytest.approx(1.0)
    )

    assert (
        context.subcategory_profile[
            "football"
        ]
        == pytest.approx(0.5)
    )


def test_user_ranking_context_returns_none_for_empty_history():
    context = build_user_ranking_context(
        None,
        {},
        {},
        {},
    )

    assert context is None


def test_candidate_feature_vector_has_correct_shape():
    article_embedding_map = {
        "N1": np.asarray(
            [1.0, 0.0],
            dtype=np.float32,
        ),
        "N2": np.asarray(
            [0.0, 1.0],
            dtype=np.float32,
        ),
        "N3": np.asarray(
            [1.0, 0.0],
            dtype=np.float32,
        ),
    }

    article_categories = {
        "N1": "sports",
        "N2": "finance",
        "N3": "sports",
    }

    article_subcategories = {
        "N1": "football",
        "N2": "markets",
        "N3": "football",
    }

    context = build_user_ranking_context(
        "N1 N2",
        article_embedding_map,
        article_categories,
        article_subcategories,
    )

    popularity = Counter(
        {
            "N3": 3,
        }
    )

    features = build_candidate_features(
        news_id="N3",
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

    assert features is not None

    assert features.shape == (
        len(FEATURE_NAMES),
    )

    assert features.dtype == np.float32

    assert np.all(
        np.isfinite(features)
    )


def test_candidate_features_capture_category_affinity():
    article_embedding_map = {
        "N1": np.asarray(
            [1.0, 0.0],
            dtype=np.float32,
        ),
        "N2": np.asarray(
            [1.0, 0.0],
            dtype=np.float32,
        ),
    }

    article_categories = {
        "N1": "sports",
        "N2": "sports",
    }

    article_subcategories = {
        "N1": "football",
        "N2": "football",
    }

    context = build_user_ranking_context(
        "N1",
        article_embedding_map,
        article_categories,
        article_subcategories,
    )

    features = build_candidate_features(
        news_id="N2",
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
        popularity=Counter(),
    )

    category_index = FEATURE_NAMES.index(
        "category_affinity"
    )

    subcategory_index = FEATURE_NAMES.index(
        "subcategory_affinity"
    )

    assert features[
        category_index
    ] == pytest.approx(1.0)

    assert features[
        subcategory_index
    ] == pytest.approx(1.0)