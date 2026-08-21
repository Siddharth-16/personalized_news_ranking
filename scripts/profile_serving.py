import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.data.parse import parse_history
from src.ranking.features import (
    build_candidate_features,
    build_user_ranking_context,
)
from src.serving.engine import RecommendationEngine


OUTPUT_PATH = Path(
    "artifacts/metrics/serving_profile.json"
)


def summarize(
    values: list[float],
) -> dict[str, float]:

    array = np.asarray(
        values,
        dtype=float,
    )

    return {
        "mean": float(
            np.mean(array)
        ),
        "p50": float(
            np.percentile(array, 50)
        ),
        "p95": float(
            np.percentile(array, 95)
        ),
        "p99": float(
            np.percentile(array, 99)
        ),
    }


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--requests",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
    )

    args = parser.parse_args()

    print("=" * 70)
    print(
        "PERSONALIZED NEWS RANKING ENGINE "
        "— SERVING PROFILE"
    )
    print("=" * 70)

    engine = RecommendationEngine()

    user_ids = list(
        engine.user_histories.keys()
    )

    stage_times = {
        "user_context": [],
        "retrieval": [],
        "feature_building": [],
        "matrix_creation": [],
        "ranker_prediction": [],
        "sorting": [],
        "total": [],
    }

    profiled = 0
    skipped = 0

    total_needed = (
        args.warmup + args.requests
    )

    print(
        f"\nProfiling {args.requests:,} "
        "personalized recommendations..."
    )

    for user_id in user_ids:

        if profiled >= total_needed:
            break

        history = engine.user_histories.get(
            user_id
        )

        total_start = time.perf_counter()

        # ----------------------------------------------
        # User representation/context
        # ----------------------------------------------

        start = time.perf_counter()

        context = build_user_ranking_context(
            history,
            engine.article_embedding_map,
            engine.article_categories,
            engine.article_subcategories,
        )

        user_context_ms = (
            time.perf_counter() - start
        ) * 1000

        if context is None:
            skipped += 1
            continue

        history_ids = set(
            parse_history(history)
        )

        # ----------------------------------------------
        # FAISS retrieval
        # ----------------------------------------------

        start = time.perf_counter()

        candidate_ids = (
            engine._retrieve_candidates(
                context.user_embedding,
                exclude_ids=history_ids,
                candidate_k=100,
            )
        )

        retrieval_ms = (
            time.perf_counter() - start
        ) * 1000

        # ----------------------------------------------
        # Candidate feature generation
        # ----------------------------------------------

        start = time.perf_counter()

        feature_rows = []
        valid_candidate_ids = []

        for news_id in candidate_ids:

            features = build_candidate_features(
                news_id=news_id,
                context=context,
                article_embedding_map=(
                    engine.article_embedding_map
                ),
                article_categories=(
                    engine.article_categories
                ),
                article_subcategories=(
                    engine.article_subcategories
                ),
                popularity=engine.popularity,
            )

            if features is None:
                continue

            feature_rows.append(
                features
            )

            valid_candidate_ids.append(
                news_id
            )

        feature_building_ms = (
            time.perf_counter() - start
        ) * 1000

        # ----------------------------------------------
        # Feature matrix creation
        # ----------------------------------------------

        start = time.perf_counter()

        X_candidates = np.vstack(
            feature_rows
        ).astype(
            np.float32,
            copy=False,
        )

        matrix_creation_ms = (
            time.perf_counter() - start
        ) * 1000

        # ----------------------------------------------
        # Ranker inference
        # ----------------------------------------------

        start = time.perf_counter()

        scores = engine.ranker.predict_proba(
            X_candidates
        )[:, 1]

        ranker_prediction_ms = (
            time.perf_counter() - start
        ) * 1000

        # ----------------------------------------------
        # Top-K sorting
        # ----------------------------------------------

        start = time.perf_counter()

        order = np.argsort(
            -scores,
            kind="stable",
        )

        _ = [
            (
                valid_candidate_ids[index],
                scores[index],
            )
            for index in order[:10]
        ]

        sorting_ms = (
            time.perf_counter() - start
        ) * 1000

        total_ms = (
            time.perf_counter()
            - total_start
        ) * 1000

        # Warm-up requests are executed but not recorded.
        if profiled >= args.warmup:

            stage_times[
                "user_context"
            ].append(
                user_context_ms
            )

            stage_times[
                "retrieval"
            ].append(
                retrieval_ms
            )

            stage_times[
                "feature_building"
            ].append(
                feature_building_ms
            )

            stage_times[
                "matrix_creation"
            ].append(
                matrix_creation_ms
            )

            stage_times[
                "ranker_prediction"
            ].append(
                ranker_prediction_ms
            )

            stage_times[
                "sorting"
            ].append(
                sorting_ms
            )

            stage_times[
                "total"
            ].append(
                total_ms
            )

        profiled += 1

    results = {
        stage: summarize(values)
        for stage, values
        in stage_times.items()
    }

    print("\n" + "=" * 70)
    print("PROFILE RESULTS")
    print("=" * 70)

    print(
        f"\n{'Stage':<24}"
        f"{'Mean':>12}"
        f"{'p50':>12}"
        f"{'p95':>12}"
    )

    print("-" * 60)

    for stage, metrics in results.items():

        print(
            f"{stage:<24}"
            f"{metrics['mean']:>10.3f}ms"
            f"{metrics['p50']:>10.3f}ms"
            f"{metrics['p95']:>10.3f}ms"
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "profiled_requests": (
            len(stage_times["total"])
        ),
        "warmup_requests": (
            args.warmup
        ),
        "skipped_users": skipped,
        "candidate_k": 100,
        "top_k": 10,
        "stages_ms": results,
    }

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
        f"\nSaved profile to: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()