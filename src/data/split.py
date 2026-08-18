from __future__ import annotations

import pandas as pd


def chronological_split(
    behaviors: pd.DataFrame,
    validation_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split behavior impressions chronologically.

    Rows before validation_date are used for training.
    Rows on or after validation_date are used for validation.
    """

    cutoff = pd.Timestamp(validation_date)

    train = behaviors[
        behaviors["time"] < cutoff
    ].copy()

    validation = behaviors[
        behaviors["time"] >= cutoff
    ].copy()

    if train.empty:
        raise ValueError("Training split is empty.")

    if validation.empty:
        raise ValueError("Validation split is empty.")

    train = train.sort_values("time").reset_index(drop=True)
    validation = validation.sort_values("time").reset_index(drop=True)

    return train, validation