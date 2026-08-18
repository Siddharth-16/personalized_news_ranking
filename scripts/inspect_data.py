import numpy as np

from src.data.load import load_behaviors, load_news
from src.data.parse import parse_history, parse_impressions
from src.data.split import chronological_split


TRAIN_BEHAVIORS = "data/raw/train/behaviors.tsv"
TRAIN_NEWS = "data/raw/train/news.tsv"


def main():
    behaviors = load_behaviors(TRAIN_BEHAVIORS)
    news = load_news(TRAIN_NEWS)

    print("=" * 60)
    print("DATASET SHAPES")
    print("=" * 60)

    print(f"Behavior rows: {len(behaviors):,}")
    print(f"News rows:     {len(news):,}")

    print("\nBehavior columns:")
    print(behaviors.columns.tolist())

    print("\nNews columns:")
    print(news.columns.tolist())

    print("\n" + "=" * 60)
    print("DATE RANGE")
    print("=" * 60)

    print(f"Earliest: {behaviors['time'].min()}")
    print(f"Latest:   {behaviors['time'].max()}")

    print("\n" + "=" * 60)
    print("FIRST IMPRESSION")
    print("=" * 60)

    row = behaviors.iloc[0]

    history = parse_history(row["history"])
    candidates, labels = parse_impressions(row["impressions"])

    print(f"Impression ID: {row['impression_id']}")
    print(f"User:          {row['user_id']}")
    print(f"Time:          {row['time']}")

    print(f"\nHistory ({len(history)} articles):")
    print(history)

    print(f"\nCandidates ({len(candidates)} articles):")
    print(candidates)

    print("\nLabels:")
    print(labels)

    print(f"\nPositive clicks: {sum(labels)}")
    print(f"Negative items:  {len(labels) - sum(labels)}")

    print("\n" + "=" * 60)
    print("USER / ARTICLE COUNTS")
    print("=" * 60)

    print(f"Unique users:    {behaviors['user_id'].nunique():,}")
    print(f"Unique articles: {news['news_id'].nunique():,}")


    print("\n" + "=" * 60)
    print("IMPRESSIONS BY DATE")
    print("=" * 60)

    impressions_by_date = (
        behaviors["time"]
        .dt.date
        .value_counts()
        .sort_index()
    )

    print(impressions_by_date)


    print("\n" + "=" * 60)
    print("HISTORY STATISTICS")
    print("=" * 60)

    history_lengths = []

    for history in behaviors["history"]:
        parsed_history = parse_history(history)
        history_lengths.append(len(parsed_history))

    history_lengths = np.array(history_lengths)

    print(f"Empty histories: {(history_lengths == 0).sum():,}")
    print(
        f"Empty history %: "
        f"{(history_lengths == 0).mean() * 100:.2f}%"
    )

    print(f"Mean length:   {history_lengths.mean():.2f}")
    print(f"Median length: {np.median(history_lengths):.2f}")
    print(f"P95 length:    {np.percentile(history_lengths, 95):.2f}")
    print(f"Max length:    {history_lengths.max():,}")


    print("\n" + "=" * 60)
    print("CANDIDATE / CLICK STATISTICS")
    print("=" * 60)

    candidate_counts = []
    positive_counts = []

    for impressions in behaviors["impressions"]:
        candidates, labels = parse_impressions(impressions)

        candidate_counts.append(len(candidates))
        positive_counts.append(sum(labels))

    candidate_counts = np.array(candidate_counts)
    positive_counts = np.array(positive_counts)

    print("Candidate count per impression:")
    print(f"Mean:   {candidate_counts.mean():.2f}")
    print(f"Median: {np.median(candidate_counts):.2f}")
    print(f"P95:    {np.percentile(candidate_counts, 95):.2f}")
    print(f"Max:    {candidate_counts.max():,}")

    print("\nPositive clicks per impression:")
    print(f"Mean:   {positive_counts.mean():.2f}")
    print(f"Median: {np.median(positive_counts):.2f}")
    print(f"P95:    {np.percentile(positive_counts, 95):.2f}")
    print(f"Max:    {positive_counts.max():,}")

    print(
        f"\nImpressions with zero positives: "
        f"{(positive_counts == 0).sum():,}"
    )


    print("\n" + "=" * 60)
    print("NEWS CATEGORIES")
    print("=" * 60)

    print(news["category"].value_counts())


    print("\n" + "=" * 60)
    print("DATA INTEGRITY")
    print("=" * 60)

    news_ids = set(news["news_id"])

    history_ids = set()
    candidate_ids = set()

    invalid_labels = 0

    for _, row in behaviors.iterrows():

        history = parse_history(row["history"])
        candidates, labels = parse_impressions(row["impressions"])

        history_ids.update(history)
        candidate_ids.update(candidates)

        invalid_labels += sum(
            label not in {0, 1}
            for label in labels
        )

    missing_history_ids = history_ids - news_ids
    missing_candidate_ids = candidate_ids - news_ids

    print(
        f"History article IDs missing from news.tsv: "
        f"{len(missing_history_ids):,}"
    )

    print(
        f"Candidate article IDs missing from news.tsv: "
        f"{len(missing_candidate_ids):,}"
    )

    print(f"Invalid click labels: {invalid_labels:,}")

    print("\n" + "=" * 60)
    print("CHRONOLOGICAL SPLIT")
    print("=" * 60)

    train, validation = chronological_split(
        behaviors,
        validation_date="2019-11-14",
    )

    print(f"Train impressions:      {len(train):,}")
    print(f"Validation impressions: {len(validation):,}")

    print(
        f"Train percentage: "
        f"{len(train) / len(behaviors) * 100:.2f}%"
    )

    print(
        f"Validation percentage: "
        f"{len(validation) / len(behaviors) * 100:.2f}%"
    )

    print(f"\nTrain start: {train['time'].min()}")
    print(f"Train end:   {train['time'].max()}")

    print(
        f"Validation start: "
        f"{validation['time'].min()}"
    )

    print(
        f"Validation end:   "
        f"{validation['time'].max()}"
    )

    assert train["time"].max() < validation["time"].min()


if __name__ == "__main__":
    main()