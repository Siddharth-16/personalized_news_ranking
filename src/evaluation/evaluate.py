from collections.abc import Callable

import numpy as np
import pandas as pd

from src.data.parse import parse_impressions
from src.evaluation.metrics import (
    mrr,
    ndcg_at_k,
    recall_at_k,
)


def evaluate_ranking(
    behaviors: pd.DataFrame,
    score_fn: Callable,
) -> dict[str, float]:
    """
    Evaluate a scoring function across behavior impressions.

    score_fn receives:
        row
        candidates

    and must return one score per candidate.
    """

    mrr_values = []
    ndcg_5_values = []
    ndcg_10_values = []
    recall_5_values = []
    recall_10_values = []

    for row in behaviors.itertuples(index=False):

        candidates, labels = parse_impressions(
            row.impressions
        )

        scores = score_fn(
            row,
            candidates,
        )

        if len(scores) != len(labels):
            raise ValueError(
                "Scoring function must return one score "
                "per candidate."
            )

        mrr_values.append(
            mrr(labels, scores)
        )

        ndcg_5_values.append(
            ndcg_at_k(labels, scores, k=5)
        )

        ndcg_10_values.append(
            ndcg_at_k(labels, scores, k=10)
        )

        recall_5_values.append(
            recall_at_k(labels, scores, k=5)
        )

        recall_10_values.append(
            recall_at_k(labels, scores, k=10)
        )

    return {
        "mrr": float(np.mean(mrr_values)),
        "ndcg@5": float(np.mean(ndcg_5_values)),
        "ndcg@10": float(np.mean(ndcg_10_values)),
        "recall@5": float(np.mean(recall_5_values)),
        "recall@10": float(np.mean(recall_10_values)),
    }