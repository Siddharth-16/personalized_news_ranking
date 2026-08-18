import pandas as pd

from src.data.split import chronological_split


def test_chronological_split():
    df = pd.DataFrame(
        {
            "time": pd.to_datetime(
                [
                    "2019-11-12 12:00:00",
                    "2019-11-14 08:00:00",
                    "2019-11-13 14:00:00",
                    "2019-11-14 18:00:00",
                ]
            ),
            "value": [1, 2, 3, 4],
        }
    )

    train, validation = chronological_split(
        df,
        validation_date="2019-11-14",
    )

    assert len(train) == 2
    assert len(validation) == 2

    assert train["time"].max() < validation["time"].min()

    assert all(
        train["time"] < pd.Timestamp("2019-11-14")
    )

    assert all(
        validation["time"] >= pd.Timestamp("2019-11-14")
    )