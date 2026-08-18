import pytest
import numpy as np

from src.evaluation.metrics import mrr, ndcg_at_k, recall_at_k


def test_mrr_single_positive_at_rank_one():
    labels = [1, 0, 0]
    scores = [0.9, 0.5, 0.1]

    assert mrr(labels, scores) == pytest.approx(1.0)


def test_mrr_single_positive_at_rank_two():
    labels = [0, 1, 0]
    scores = [0.9, 0.8, 0.1]

    assert mrr(labels, scores) == pytest.approx(0.5)


def test_mrr_multiple_positives():
    labels = [0, 1, 0, 1]
    scores = [0.9, 0.8, 0.7, 0.6]

    expected = (1 / 2 + 1 / 4) / 2

    assert mrr(labels, scores) == pytest.approx(expected)


def test_mrr_multiple_positives_at_ranks_two_and_three():
    labels = [1, 0, 1]
    scores = [0.2, 0.9, 0.8]

    expected = (1 / 2 + 1 / 3) / 2

    assert mrr(labels, scores) == pytest.approx(expected)


def test_mrr_zero_positives():
    labels = [0, 0, 0]
    scores = [0.9, 0.5, 0.1]

    assert mrr(labels, scores) == pytest.approx(0.0)


def test_mrr_rejects_mismatched_lengths():
    labels = [1, 0, 1]
    scores = [0.9, 0.3]

    with pytest.raises(ValueError):
        mrr(labels, scores)


def test_ndcg_perfect_ranking():
    labels = [1, 1, 0, 0]
    scores = [0.9, 0.8, 0.2, 0.1]

    assert ndcg_at_k(
        labels,
        scores,
        k=4,
    ) == pytest.approx(1.0)


def test_ndcg_imperfect_ranking():
    labels = [0, 1, 0, 1]
    scores = [0.9, 0.8, 0.7, 0.6]

    result = ndcg_at_k(
        labels,
        scores,
        k=4,
    )

    assert 0.0 < result < 1.0


def test_ndcg_exact_value():
    labels = [0, 1, 0]
    scores = [0.9, 0.8, 0.1]

    actual_dcg = 1 / np.log2(3)

    ideal_dcg = 1 / np.log2(2)

    expected = actual_dcg / ideal_dcg

    assert ndcg_at_k(
        labels,
        scores,
        k=3,
    ) == pytest.approx(expected)


def test_ndcg_respects_k():
    labels = [0, 0, 1]
    scores = [0.9, 0.8, 0.7]

    assert ndcg_at_k(
        labels,
        scores,
        k=2,
    ) == pytest.approx(0.0)

    assert ndcg_at_k(
        labels,
        scores,
        k=3,
    ) > 0.0


def test_ndcg_k_larger_than_candidate_count():
    labels = [1, 0]
    scores = [0.9, 0.1]

    assert ndcg_at_k(
        labels,
        scores,
        k=10,
    ) == pytest.approx(1.0)


def test_ndcg_zero_positives():
    labels = [0, 0, 0]
    scores = [0.9, 0.5, 0.1]

    assert ndcg_at_k(
        labels,
        scores,
        k=3,
    ) == pytest.approx(0.0)


def test_ndcg_rejects_invalid_k():
    labels = [1, 0]
    scores = [0.9, 0.1]

    with pytest.raises(ValueError):
        ndcg_at_k(
            labels,
            scores,
            k=0,
        )


def test_recall_at_k_all_relevant_found():
    labels = [1, 1, 0, 0]
    scores = [0.9, 0.8, 0.2, 0.1]

    assert recall_at_k(
        labels,
        scores,
        k=2,
    ) == pytest.approx(1.0)


def test_recall_at_k_partial():
    labels = [0, 1, 1, 0]
    scores = [0.9, 0.8, 0.3, 0.7]

    assert recall_at_k(
        labels,
        scores,
        k=2,
    ) == pytest.approx(0.5)


def test_recall_at_k_none_found():
    labels = [0, 0, 1]
    scores = [0.9, 0.8, 0.7]

    assert recall_at_k(
        labels,
        scores,
        k=2,
    ) == pytest.approx(0.0)


def test_recall_at_k_increases_when_more_relevant_items_are_included():
    labels = [0, 1, 0, 1]
    scores = [0.9, 0.8, 0.7, 0.6]

    recall_at_2 = recall_at_k(
        labels,
        scores,
        k=2,
    )

    recall_at_4 = recall_at_k(
        labels,
        scores,
        k=4,
    )

    assert recall_at_2 == pytest.approx(0.5)
    assert recall_at_4 == pytest.approx(1.0)


def test_recall_at_k_larger_than_candidate_count():
    labels = [1, 0, 1]
    scores = [0.9, 0.5, 0.4]

    assert recall_at_k(
        labels,
        scores,
        k=100,
    ) == pytest.approx(1.0)


def test_recall_at_k_zero_positives():
    labels = [0, 0, 0]
    scores = [0.9, 0.5, 0.1]

    assert recall_at_k(
        labels,
        scores,
        k=2,
    ) == pytest.approx(0.0)


def test_recall_at_k_rejects_invalid_k():
    labels = [1, 0]
    scores = [0.9, 0.1]

    with pytest.raises(ValueError):
        recall_at_k(
            labels,
            scores,
            k=0,
        )


def test_recall_is_non_decreasing_with_k():
    labels = [0, 1, 0, 1, 1]
    scores = [0.9, 0.8, 0.7, 0.6, 0.5]

    recalls = [
        recall_at_k(labels, scores, k)
        for k in range(1, 6)
    ]

    assert recalls == sorted(recalls)