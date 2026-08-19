import json
from pathlib import Path

from src.data.load import load_behaviors, load_news
from src.data.parse import parse_history
from src.data.split import chronological_split
from src.evaluation.evaluate import evaluate_ranking
from src.ranking.baselines import (
    build_popularity_counts,
    popularity_scores,
    build_article_category_map,
    category_affinity_scores,
)


TRAIN_BEHAVIORS_PATH = (
    "data/raw/train/behaviors.tsv"
)

TRAIN_NEWS_PATH = "data/raw/train/news.tsv"

VALIDATION_DATE = "2019-11-14"

OUTPUT_PATH = Path(
    "artifacts/metrics/baselines.json"
)


def main():

    print("=" * 70)
    print("PERSONALIZED NEWS RANKING ENGINE")
    print("=" * 70)

    behaviors = load_behaviors(
        TRAIN_BEHAVIORS_PATH
    )

    news = load_news(TRAIN_NEWS_PATH)

    article_categories = build_article_category_map(
        news
    )

    train, validation = chronological_split(
        behaviors,
        validation_date=VALIDATION_DATE,
    )

    print(f"\nTrain impressions:      {len(train):,}")
    print(f"Validation impressions: {len(validation):,}")

    # --------------------------------------------------
    # Popularity baseline
    # --------------------------------------------------

    print("\nBuilding popularity counts...")

    popularity = build_popularity_counts(
        train
    )

    print(
        f"Articles with training clicks: "
        f"{len(popularity):,}"
    )

    print("\nEvaluating popularity baseline...")

    def popularity_score_fn(row, candidates):
        return popularity_scores(
            candidates,
            popularity,
        )

    popularity_metrics = evaluate_ranking(
        validation,
        popularity_score_fn,
    )

    print("\n" + "-" * 70)
    print("POPULARITY BASELINE")
    print("-" * 70)

    for metric, value in popularity_metrics.items():
        print(
            f"{metric:<12} {value:.6f}"
        )

    print("\nEvaluating category-affinity baseline...")


    def category_score_fn(row, candidates):

        history = parse_history(
            row.history
        )

        return category_affinity_scores(
            history,
            candidates,
            article_categories,
        )


    category_metrics = evaluate_ranking(
        validation,
        category_score_fn,
    )

    print("\n" + "-" * 70)
    print("CATEGORY-AFFINITY BASELINE")
    print("-" * 70)

    for metric, value in category_metrics.items():
        print(
            f"{metric:<12} {value:.6f}"
        )

    results = {
        "dataset": "MINDsmall",
        "split": {
            "strategy": "chronological",
            "training_period": (
                "2019-11-09 through 2019-11-13"
            ),
            "validation_period": "2019-11-14",
            "train_impressions": len(train),
            "validation_impressions": len(validation),
        },
        "models": {
            "popularity": popularity_metrics,
            "category_affinity": category_metrics,
        },
    }

    print("\n" + "=" * 70)
    print("BASELINE COMPARISON")
    print("=" * 70)

    print(
        f"{'Model':<22}"
        f"{'MRR':>10}"
        f"{'nDCG@5':>12}"
        f"{'nDCG@10':>12}"
    )

    print("-" * 56)

    print(
        f"{'Popularity':<22}"
        f"{popularity_metrics['mrr']:>10.4f}"
        f"{popularity_metrics['ndcg@5']:>12.4f}"
        f"{popularity_metrics['ndcg@10']:>12.4f}"
    )

    print(
        f"{'Category Affinity':<22}"
        f"{category_metrics['mrr']:>10.4f}"
        f"{category_metrics['ndcg@5']:>12.4f}"
        f"{category_metrics['ndcg@10']:>12.4f}"
    )

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