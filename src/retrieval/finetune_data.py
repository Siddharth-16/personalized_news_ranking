import numpy as np
import pandas as pd

from src.data.parse import (
    parse_history,
    parse_impressions,
)


def build_title_map(
    news: pd.DataFrame,
) -> dict[str, str]:
    """
    Map article IDs to non-empty titles.
    """

    valid_news = news.dropna(
        subset=["news_id", "title"]
    )

    return dict(
        zip(
            valid_news["news_id"],
            valid_news["title"],
        )
    )


def build_click_triplets(
    behaviors: pd.DataFrame,
    title_map: dict[str, str],
    max_examples: int | None = None,
    random_seed: int = 42,
) -> tuple[
    list[str],
    list[str],
    list[str],
    dict[str, int],
]:
    """
    Build (anchor, positive, negative) title triplets
    from training impressions.

    anchor:
        a previously clicked article

    positive:
        an article clicked in the current impression

    negative:
        a displayed but unclicked article from the
        current impression
    """

    rng = np.random.default_rng(
        random_seed
    )

    anchors = []
    positives = []
    negatives = []

    skipped_no_history = 0
    skipped_no_positive = 0
    skipped_no_negative = 0
    skipped_missing_title = 0

    for row in behaviors.itertuples(
        index=False
    ):
        history_ids = [
            news_id
            for news_id in parse_history(
                row.history
            )
            if news_id in title_map
        ]

        if not history_ids:
            skipped_no_history += 1
            continue

        candidate_ids, labels = (
            parse_impressions(
                row.impressions
            )
        )

        positive_ids = [
            news_id
            for news_id, label
            in zip(candidate_ids, labels)
            if label == 1
            and news_id in title_map
        ]

        negative_ids = [
            news_id
            for news_id, label
            in zip(candidate_ids, labels)
            if label == 0
            and news_id in title_map
        ]

        if not positive_ids:
            skipped_no_positive += 1
            continue

        if not negative_ids:
            skipped_no_negative += 1
            continue

        for positive_id in positive_ids:

            usable_history = [
                news_id
                for news_id in history_ids
                if news_id != positive_id
            ]

            if not usable_history:
                skipped_missing_title += 1
                continue

            anchor_id = rng.choice(
                usable_history
            )

            negative_id = rng.choice(
                negative_ids
            )

            anchors.append(
                title_map[anchor_id]
            )

            positives.append(
                title_map[positive_id]
            )

            negatives.append(
                title_map[negative_id]
            )

            if (
                max_examples is not None
                and len(anchors) >= max_examples
            ):
                stats = {
                    "triplets": len(anchors),
                    "skipped_no_history": (
                        skipped_no_history
                    ),
                    "skipped_no_positive": (
                        skipped_no_positive
                    ),
                    "skipped_no_negative": (
                        skipped_no_negative
                    ),
                    "skipped_missing_title": (
                        skipped_missing_title
                    ),
                }

                return (
                    anchors,
                    positives,
                    negatives,
                    stats,
                )

    stats = {
        "triplets": len(anchors),
        "skipped_no_history": (
            skipped_no_history
        ),
        "skipped_no_positive": (
            skipped_no_positive
        ),
        "skipped_no_negative": (
            skipped_no_negative
        ),
        "skipped_missing_title": (
            skipped_missing_title
        ),
    }

    return (
        anchors,
        positives,
        negatives,
        stats,
    )