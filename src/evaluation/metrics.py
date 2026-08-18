import numpy as np


def _validate_ranking_inputs(
    labels: list[int] | np.ndarray,
    scores: list[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)

    if labels.ndim != 1 or scores.ndim != 1:
        raise ValueError("labels and scores must be one-dimensional.")

    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length.")

    if len(labels) == 0:
        raise ValueError("labels and scores cannot be empty.")

    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("labels must contain only 0 and 1.")

    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must contain only finite values.")

    return labels.astype(int), scores


def mrr(
    labels: list[int] | np.ndarray,
    scores: list[float] | np.ndarray,
) -> float:
    """
    Compute MIND-style Mean Reciprocal Rank for one impression.

    For impressions containing multiple positive items, reciprocal
    ranks of all positive items are averaged.
    """

    labels, scores = _validate_ranking_inputs(labels, scores)

    positive_count = labels.sum()

    if positive_count == 0:
        return 0.0

    order = np.argsort(-scores, kind="stable")
    ranked_labels = labels[order]

    ranks = np.arange(1, len(ranked_labels) + 1)

    reciprocal_rank_contributions = ranked_labels / ranks

    return float(
        reciprocal_rank_contributions.sum() / positive_count
    )


def _dcg_at_k(
    ranked_labels: np.ndarray,
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be greater than 0.")

    k = min(k, len(ranked_labels))

    relevance = ranked_labels[:k]

    ranks = np.arange(1, k + 1)

    gains = np.power(2.0, relevance) - 1.0
    discounts = np.log2(ranks + 1)

    return float(np.sum(gains / discounts))

def ndcg_at_k(
    labels: list[int] | np.ndarray,
    scores: list[float] | np.ndarray,
    k: int,
) -> float:
    """
    Compute normalized discounted cumulative gain at K
    for one impression.
    """

    labels, scores = _validate_ranking_inputs(labels, scores)

    if k <= 0:
        raise ValueError("k must be greater than 0.")

    order = np.argsort(-scores, kind="stable")
    ranked_labels = labels[order]

    dcg = _dcg_at_k(ranked_labels, k)

    ideal_labels = np.sort(labels)[::-1]
    idcg = _dcg_at_k(ideal_labels, k)

    if idcg == 0:
        return 0.0

    return float(dcg / idcg)


def recall_at_k(
    labels: list[int] | np.ndarray,
    scores: list[float] | np.ndarray,
    k: int,
) -> float:
    """
    Compute Recall@K for one impression.

    Recall@K is the fraction of all relevant items that appear
    within the top-K items ranked by score.
    """

    labels, scores = _validate_ranking_inputs(labels, scores)

    if k <= 0:
        raise ValueError("k must be greater than 0.")

    positive_count = labels.sum()

    if positive_count == 0:
        return 0.0

    order = np.argsort(-scores, kind="stable")

    top_k_indices = order[:k]

    relevant_in_top_k = labels[top_k_indices].sum()

    return float(relevant_in_top_k / positive_count)