import pandas as pd
import pytest

from src.ranking.baselines import (
    build_popularity_counts,
    popularity_score,
    popularity_scores,
    build_article_category_map,
    build_category_profile,
    category_affinity_scores,
)

def test_build_popularity_counts_only_counts_positive_clicks():
    behaviors = pd.DataFrame(
        {
            "impressions": [
                "N1-1 N2-0 N3-1",
                "N1-1 N2-1 N3-0",
            ]
        }
    )

    popularity = build_popularity_counts(behaviors)

    assert popularity["N1"] == 2
    assert popularity["N2"] == 1
    assert popularity["N3"] == 1


def test_unseen_article_has_zero_popularity():
    behaviors = pd.DataFrame(
        {
            "impressions": [
                "N1-1 N2-0",
            ]
        }
    )

    popularity = build_popularity_counts(behaviors)

    assert popularity_score(
        "N999",
        popularity,
    ) == 0


def test_higher_popularity_always_has_higher_score():
    behaviors = pd.DataFrame(
        {
            "impressions": [
                "N1-1 N2-0",
                "N1-1 N2-1",
            ]
        }
    )

    popularity = build_popularity_counts(behaviors)

    candidates = ["N2", "N1"]

    scores = popularity_scores(
        candidates,
        popularity,
    )

    assert scores[1] > scores[0]


def test_popularity_ties_do_not_depend_on_candidate_order():
    behaviors = pd.DataFrame(
        {
            "impressions": [
                "N1-1",
            ]
        }
    )

    popularity = build_popularity_counts(behaviors)

    candidates_a = ["N20", "N10"]
    candidates_b = ["N10", "N20"]

    scores_a = popularity_scores(
        candidates_a,
        popularity,
    )

    scores_b = popularity_scores(
        candidates_b,
        popularity,
    )

    scores_by_id_a = dict(
        zip(candidates_a, scores_a)
    )

    scores_by_id_b = dict(
        zip(candidates_b, scores_b)
    )

    assert scores_by_id_a == scores_by_id_b


def test_build_category_profile():
    history = ["N1", "N2", "N3", "N4"]

    article_categories = {
        "N1": "sports",
        "N2": "sports",
        "N3": "finance",
        "N4": "sports",
    }

    profile = build_category_profile(
        history,
        article_categories,
    )

    assert profile["sports"] == pytest.approx(0.75)
    assert profile["finance"] == pytest.approx(0.25)


def test_category_affinity_prefers_user_category():
    history = ["N1", "N2", "N3"]

    article_categories = {
        "N1": "sports",
        "N2": "sports",
        "N3": "finance",
        "N4": "sports",
        "N5": "travel",
    }

    candidates = ["N5", "N4"]

    scores = category_affinity_scores(
        history,
        candidates,
        article_categories,
    )

    assert scores[1] > scores[0]


def test_category_affinity_handles_empty_history():
    candidates = ["N1", "N2"]

    article_categories = {
        "N1": "sports",
        "N2": "finance",
    }

    scores = category_affinity_scores(
        [],
        candidates,
        article_categories,
    )

    assert len(scores) == 2