from pathlib import Path

import pandas as pd


BEHAVIOR_COLUMNS = [
    "impression_id",
    "user_id",
    "time",
    "history",
    "impressions",
]


NEWS_COLUMNS = [
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]


def load_behaviors(path: str | Path) -> pd.DataFrame:
    """Load a MIND behaviors.tsv file."""

    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=BEHAVIOR_COLUMNS,
    )

    df["time"] = pd.to_datetime(
        df["time"],
        format="%m/%d/%Y %I:%M:%S %p",
    )

    return df


def load_news(path: str | Path) -> pd.DataFrame:
    """Load a MIND news.tsv file."""

    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=NEWS_COLUMNS,
    )