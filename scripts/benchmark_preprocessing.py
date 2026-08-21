import argparse
import json
import resource
import sys
import time
from pathlib import Path

import pandas as pd

from src.data.load import BEHAVIOR_COLUMNS


DEFAULT_OUTPUT = Path(
    "artifacts/metrics/preprocessing_benchmark.json"
)


def peak_rss_mb() -> float:
    """
    Return peak process resident memory in MB.

    macOS reports ru_maxrss in bytes.
    Linux reports it in KiB.
    """

    rss = resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss

    if sys.platform == "darwin":
        return rss / (1024 ** 2)

    return rss / 1024


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--behaviors",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--label",
        required=True,
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=25_000,
    )

    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help=(
            "Optional chunk limit for smoke testing."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    args = parser.parse_args()

    print("=" * 70)
    print(
        "PERSONALIZED NEWS RANKING ENGINE "
        "— PREPROCESSING SCALE BENCHMARK"
    )
    print("=" * 70)

    print(
        f"\nDataset:    {args.label}"
    )

    print(
        f"Input:      {args.behaviors}"
    )

    print(
        f"Chunksize:  {args.chunksize:,}"
    )

    input_size_mb = (
        args.behaviors.stat().st_size
        / (1024 ** 2)
    )

    print(
        f"File size:  {input_size_mb:.2f} MB"
    )

    total_start = time.perf_counter()

    read_seconds = 0.0
    transform_seconds = 0.0

    impression_count = 0
    interaction_count = 0
    click_count = 0
    history_click_count = 0
    empty_history_count = 0

    unique_articles: set[str] = set()

    chunk_number = 0

    reader = pd.read_csv(
        args.behaviors,
        sep="\t",
        header=None,
        names=BEHAVIOR_COLUMNS,
        chunksize=args.chunksize,
    )

    while True:

        if (
            args.max_chunks is not None
            and chunk_number
            >= args.max_chunks
        ):
            break

        read_start = time.perf_counter()

        try:
            chunk = next(reader)
        except StopIteration:
            break

        read_seconds += (
            time.perf_counter()
            - read_start
        )

        transform_start = (
            time.perf_counter()
        )

        # ----------------------------------------------
        # Parse timestamp
        # ----------------------------------------------

        chunk["time"] = pd.to_datetime(
            chunk["time"],
            format="%m/%d/%Y %I:%M:%S %p",
        )

        # ----------------------------------------------
        # History preprocessing
        # ----------------------------------------------

        empty_histories = (
            chunk["history"].isna()
        )

        empty_history_count += int(
            empty_histories.sum()
        )

        history_lengths = (
            chunk["history"]
            .fillna("")
            .str.split()
            .str.len()
        )

        history_click_count += int(
            history_lengths.sum()
        )

        # ----------------------------------------------
        # Candidate-level interaction preprocessing
        #
        # One behavior impression becomes many
        # (user, article, label) interactions.
        # ----------------------------------------------

        impression_tokens = (
            chunk["impressions"]
            .str.split()
            .explode()
        )

        interaction_count += len(
            impression_tokens
        )

        click_count += int(
            impression_tokens
            .str.endswith("-1")
            .sum()
        )

        candidate_ids = (
            impression_tokens
            .str.rsplit(
                "-",
                n=1,
            )
            .str[0]
        )

        unique_articles.update(
            candidate_ids.unique()
        )

        impression_count += len(
            chunk
        )

        transform_seconds += (
            time.perf_counter()
            - transform_start
        )

        chunk_number += 1

        print(
            f"\rProcessed "
            f"{impression_count:,} impressions "
            f"| {interaction_count:,} interactions "
            f"| peak RSS {peak_rss_mb():.0f} MB",
            end="",
            flush=True,
        )

    total_seconds = (
        time.perf_counter()
        - total_start
    )

    print("\n")

    interactions_per_second = (
        interaction_count / total_seconds
        if total_seconds > 0
        else 0.0
    )

    results = {
        "dataset": args.label,
        "input": str(
            args.behaviors
        ),
        "input_size_mb": (
            input_size_mb
        ),
        "chunksize": (
            args.chunksize
        ),
        "chunks_processed": (
            chunk_number
        ),
        "impressions": (
            impression_count
        ),
        "candidate_interactions": (
            interaction_count
        ),
        "clicks": (
            click_count
        ),
        "history_clicks": (
            history_click_count
        ),
        "empty_histories": (
            empty_history_count
        ),
        "unique_candidate_articles": (
            len(unique_articles)
        ),
        "timing_seconds": {
            "read": read_seconds,
            "transform": (
                transform_seconds
            ),
            "total": total_seconds,
        },
        "throughput": {
            "interactions_per_second": (
                interactions_per_second
            ),
        },
        "peak_rss_mb": (
            peak_rss_mb()
        ),
    }

    print(
        f"Impressions:            "
        f"{impression_count:,}"
    )

    print(
        f"Candidate interactions: "
        f"{interaction_count:,}"
    )

    print(
        f"Positive clicks:         "
        f"{click_count:,}"
    )

    print(
        f"Unique articles:         "
        f"{len(unique_articles):,}"
    )

    print(
        f"Read time:               "
        f"{read_seconds:.2f} s"
    )

    print(
        f"Transform time:          "
        f"{transform_seconds:.2f} s"
    )

    print(
        f"Total time:              "
        f"{total_seconds:.2f} s"
    )

    print(
        f"Throughput:              "
        f"{interactions_per_second:,.0f} "
        f"interactions/s"
    )

    print(
        f"Peak RSS:                "
        f"{peak_rss_mb():.0f} MB"
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    print(
        f"\nSaved benchmark to: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()