import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import httpx
import numpy as np

from src.data.load import load_behaviors
from src.data.split import chronological_split


BEHAVIORS_PATH = Path(
    "data/raw/train/behaviors.tsv"
)

OUTPUT_PATH = Path(
    "artifacts/metrics/api_baseline.json"
)

VALIDATION_DATE = "2019-11-14"
RANDOM_SEED = 42


def percentile_summary(
    latencies_ms: list[float],
) -> dict[str, float]:
    values = np.asarray(
        latencies_ms,
        dtype=float,
    )

    return {
        "mean": float(
            np.mean(values)
        ),
        "p50": float(
            np.percentile(values, 50)
        ),
        "p95": float(
            np.percentile(values, 95)
        ),
        "p99": float(
            np.percentile(values, 99)
        ),
        "min": float(
            np.min(values)
        ),
        "max": float(
            np.max(values)
        ),
    }


def load_known_users() -> list[str]:
    """
    Load users known to the serving system.

    We benchmark personalized requests separately from
    the much cheaper popularity fallback path.
    """

    behaviors = load_behaviors(
        BEHAVIORS_PATH
    )

    train, _ = chronological_split(
        behaviors,
        validation_date=VALIDATION_DATE,
    )

    users = (
        train["user_id"]
        .drop_duplicates()
        .tolist()
    )

    return users


async def warm_up(
    client: httpx.AsyncClient,
    base_url: str,
    user_ids: list[str],
    requests: int,
) -> None:

    print(
        f"Warming up with {requests} requests..."
    )

    for index in range(requests):

        user_id = user_ids[
            index % len(user_ids)
        ]

        response = await client.get(
            f"{base_url}/recommend/{user_id}"
        )

        response.raise_for_status()


async def run_single_request(
    client: httpx.AsyncClient,
    url: str,
    semaphore: asyncio.Semaphore,
) -> tuple[float, bool]:

    async with semaphore:

        start = time.perf_counter()

        try:
            response = await client.get(
                url
            )

            success = (
                response.status_code == 200
            )

        except httpx.HTTPError:
            success = False

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        return elapsed_ms, success


async def benchmark_concurrency(
    client: httpx.AsyncClient,
    base_url: str,
    user_ids: list[str],
    concurrency: int,
    total_requests: int,
) -> dict:

    semaphore = asyncio.Semaphore(
        concurrency
    )

    urls = [
        (
            f"{base_url}/recommend/"
            f"{user_ids[index % len(user_ids)]}"
        )
        for index in range(total_requests)
    ]

    benchmark_start = time.perf_counter()

    results = await asyncio.gather(
        *[
            run_single_request(
                client,
                url,
                semaphore,
            )
            for url in urls
        ]
    )

    total_elapsed = (
        time.perf_counter()
        - benchmark_start
    )

    latencies_ms = [
        latency
        for latency, success in results
        if success
    ]

    successes = sum(
        success
        for _, success in results
    )

    failures = (
        total_requests - successes
    )

    requests_per_second = (
        successes / total_elapsed
        if total_elapsed > 0
        else 0.0
    )

    latency = (
        percentile_summary(
            latencies_ms
        )
        if latencies_ms
        else {}
    )

    return {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "successful_requests": successes,
        "failed_requests": failures,
        "failure_rate": (
            failures / total_requests
        ),
        "wall_time_seconds": (
            total_elapsed
        ),
        "requests_per_second": (
            requests_per_second
        ),
        "latency_ms": latency,
    }


async def main_async(
    args,
) -> None:

    print("=" * 70)
    print(
        "PERSONALIZED NEWS RANKING ENGINE"
    )
    print("=" * 70)

    users = load_known_users()

    rng = random.Random(
        RANDOM_SEED
    )

    rng.shuffle(
        users
    )

    print(
        f"\nKnown users available: "
        f"{len(users):,}"
    )

    print(
        f"Requests per level:    "
        f"{args.requests}"
    )

    print(
        "Concurrency levels:    "
        + ", ".join(
            str(value)
            for value
            in args.concurrency
        )
    )

    max_concurrency = max(
        args.concurrency
    )

    # Keep the client connection pool larger than the
    # maximum benchmark concurrency.
    limits = httpx.Limits(
        max_connections=(
            max_concurrency + 10
        ),
        max_keepalive_connections=(
            max_concurrency + 10
        ),
    )

    timeout = httpx.Timeout(
        30.0
    )

    async with httpx.AsyncClient(
        limits=limits,
        timeout=timeout,
    ) as client:

        await warm_up(
            client=client,
            base_url=args.base_url,
            user_ids=users,
            requests=args.warmup,
        )

        all_results = []

        for concurrency in args.concurrency:

            print(
                "\n"
                + "-" * 70
            )

            print(
                f"Concurrency: {concurrency}"
            )

            result = (
                await benchmark_concurrency(
                    client=client,
                    base_url=args.base_url,
                    user_ids=users,
                    concurrency=concurrency,
                    total_requests=(
                        args.requests
                    ),
                )
            )

            all_results.append(
                result
            )

            latency = result[
                "latency_ms"
            ]

            print(
                f"Successful: "
                f"{result['successful_requests']:,}"
            )

            print(
                f"Failed:     "
                f"{result['failed_requests']:,}"
            )

            print(
                f"RPS:        "
                f"{result['requests_per_second']:.2f}"
            )

            if latency:

                print(
                    f"Mean:       "
                    f"{latency['mean']:.2f} ms"
                )

                print(
                    f"p50:        "
                    f"{latency['p50']:.2f} ms"
                )

                print(
                    f"p95:        "
                    f"{latency['p95']:.2f} ms"
                )

                print(
                    f"p99:        "
                    f"{latency['p99']:.2f} ms"
                )

    output = {
        "benchmark": (
            "unoptimized_api"
        ),
        "endpoint": (
            "/recommend/{user_id}"
        ),
        "request_type": (
            "personalized_known_users"
        ),
        "requests_per_concurrency": (
            args.requests
        ),
        "warmup_requests": (
            args.warmup
        ),
        "results": (
            all_results
        ),
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
        )

    print(
        f"\nSaved benchmark to: "
        f"{args.output}"
    )


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
    )

    parser.add_argument(
        "--requests",
        type=int,
        default=500,
        help=(
            "Total requests at each "
            "concurrency level."
        ),
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=[1, 5, 10, 25],
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
    )

    args = parser.parse_args()

    if args.requests <= 0:
        raise ValueError(
            "--requests must be greater than 0."
        )

    if args.warmup < 0:
        raise ValueError(
            "--warmup cannot be negative."
        )

    if any(
        value <= 0
        for value in args.concurrency
    ):
        raise ValueError(
            "Concurrency levels must be "
            "greater than 0."
        )

    asyncio.run(
        main_async(args)
    )


if __name__ == "__main__":
    main()