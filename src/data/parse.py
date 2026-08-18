import pandas as pd


def parse_history(history) -> list[str]:
    """Parse a user's previously clicked news IDs."""

    if pd.isna(history):
        return []

    return history.split()


def parse_impressions(impressions: str) -> tuple[list[str], list[int]]:
    """Parse candidate news IDs and click labels."""

    candidates = []
    labels = []

    for token in impressions.split():
        news_id, label = token.rsplit("-", 1)

        candidates.append(news_id)
        labels.append(int(label))

    return candidates, labels