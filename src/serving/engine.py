import json
from pathlib import Path

import faiss
import joblib
import numpy as np

from src.data.load import load_behaviors, load_news
from src.data.parse import parse_history, parse_impressions
from src.data.split import chronological_split
from src.ranking.baselines import (
    build_article_category_map,
    build_popularity_counts,
)
from src.ranking.features import (
    build_article_subcategory_map,
    build_candidate_features,
    build_user_ranking_context,
)
from src.retrieval.user_embeddings import (
    build_article_embedding_map,
)


BEHAVIORS_PATH = Path(
    "data/raw/train/behaviors.tsv"
)

NEWS_PATH = Path(
    "data/raw/train/news.tsv"
)

EMBEDDINGS_PATH = Path(
    "artifacts/embeddings/article_embeddings.npy"
)

ARTICLE_IDS_PATH = Path(
    "artifacts/embeddings/article_ids.json"
)

FAISS_INDEX_PATH = Path(
    "artifacts/retrieval/article.index"
)

RANKER_PATH = Path(
    "artifacts/models/ranker.joblib"
)

VALIDATION_DATE = "2019-11-14"


class RecommendationEngine:
    """
    Frozen two-stage recommendation pipeline.

    Stage 1:
        FAISS candidate retrieval.

    Stage 2:
        learned reranking.

    Unknown/cold users receive a popularity fallback.
    """

    def __init__(self) -> None:

        print("Loading serving artifacts...")

        # --------------------------------------------------
        # Data
        # --------------------------------------------------

        behaviors = load_behaviors(
            BEHAVIORS_PATH
        )

        news = load_news(
            NEWS_PATH
        )

        train, _ = chronological_split(
            behaviors,
            validation_date=VALIDATION_DATE,
        )

        # --------------------------------------------------
        # Article embeddings
        # --------------------------------------------------

        self.article_embeddings = np.load(
            EMBEDDINGS_PATH
        )

        with ARTICLE_IDS_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            self.article_ids = json.load(
                file
            )

        self.article_embedding_map = (
            build_article_embedding_map(
                self.article_ids,
                self.article_embeddings,
            )
        )

        # --------------------------------------------------
        # Reconstruct exact article ordering used when
        # the frozen FAISS index was created.
        # --------------------------------------------------

        train_article_ids = (
            self._collect_observed_articles(
                train
            )
        )

        self.retrieval_article_ids = [
            news_id
            for news_id in self.article_ids
            if news_id in train_article_ids
        ]

        # --------------------------------------------------
        # Load FAISS ONCE
        # --------------------------------------------------

        self.index = faiss.read_index(
            str(FAISS_INDEX_PATH)
        )

        if (
            self.index.ntotal
            != len(self.retrieval_article_ids)
        ):
            raise RuntimeError(
                "FAISS index size does not match "
                "retrieval article ID mapping."
            )

        if (
            self.index.d
            != self.article_embeddings.shape[1]
        ):
            raise RuntimeError(
                "FAISS index dimension does not match "
                "article embeddings."
            )

        # --------------------------------------------------
        # Load ranker ONCE
        # --------------------------------------------------

        self.ranker = joblib.load(
            RANKER_PATH
        )

        # --------------------------------------------------
        # Ranking metadata
        # --------------------------------------------------

        self.article_categories = (
            build_article_category_map(
                news
            )
        )

        self.article_subcategories = (
            build_article_subcategory_map(
                news
            )
        )

        self.popularity = (
            build_popularity_counts(
                train
            )
        )

        # --------------------------------------------------
        # Final known user state at deployment cutoff
        # --------------------------------------------------

        self.user_histories = (
            self._build_latest_user_histories(
                train
            )
        )

        # --------------------------------------------------
        # Cold-user fallback
        # --------------------------------------------------

        self.popularity_ranking = sorted(
            self.retrieval_article_ids,
            key=lambda news_id: (
                -self.popularity[news_id],
                news_id,
            ),
        )

        print(
            f"FAISS vectors:       "
            f"{self.index.ntotal:,}"
        )

        print(
            f"Known users:         "
            f"{len(self.user_histories):,}"
        )

        print(
            f"Embedding dimension: "
            f"{self.index.d}"
        )

        print(
            "Recommendation engine ready."
        )

    @staticmethod
    def _collect_observed_articles(
        behaviors,
    ) -> set[str]:

        observed = set()

        for row in behaviors.itertuples(
            index=False
        ):
            observed.update(
                parse_history(
                    row.history
                )
            )

            candidates, _ = (
                parse_impressions(
                    row.impressions
                )
            )

            observed.update(
                candidates
            )

        return observed

    @staticmethod
    def _build_latest_user_histories(
        train_behaviors,
    ) -> dict[str, str]:
        """
        Construct the latest known user state at the
        end of each user's training-period activity.

        The current impression's clicked articles are
        appended to its pre-impression history.
        """

        latest_histories = {}

        ordered = train_behaviors.sort_values(
            "time"
        )

        for row in ordered.itertuples(
            index=False
        ):

            history_ids = parse_history(
                row.history
            )

            candidates, labels = (
                parse_impressions(
                    row.impressions
                )
            )

            current_clicks = [
                news_id
                for news_id, label
                in zip(candidates, labels)
                if label == 1
            ]

            final_history = (
                history_ids
                + current_clicks
            )

            # Remove duplicates while preserving order.
            final_history = list(
                dict.fromkeys(
                    final_history
                )
            )

            latest_histories[
                row.user_id
            ] = " ".join(
                final_history
            )

        return latest_histories

    def _popularity_fallback(
        self,
        user_id: str,
        k: int,
    ) -> dict:

        recommendations = [
            {
                "news_id": news_id,
                "score": float(
                    self.popularity[
                        news_id
                    ]
                ),
            }
            for news_id
            in self.popularity_ranking[:k]
        ]

        return {
            "user_id": user_id,
            "strategy": (
                "popularity_fallback"
            ),
            "recommendations": (
                recommendations
            ),
        }

    def _retrieve_candidates(
        self,
        user_embedding: np.ndarray,
        exclude_ids: set[str],
        candidate_k: int,
    ) -> list[str]:

        query = np.asarray(
            user_embedding,
            dtype=np.float32,
        )

        query = query / np.linalg.norm(
            query
        )

        query = np.ascontiguousarray(
            query.reshape(1, -1)
        )

        search_k = min(
            self.index.ntotal,
            candidate_k
            + len(exclude_ids),
        )

        _, indices = self.index.search(
            query,
            search_k,
        )

        retrieved = []

        for index in indices[0]:

            if index < 0:
                continue

            news_id = (
                self.retrieval_article_ids[
                    index
                ]
            )

            if news_id in exclude_ids:
                continue

            retrieved.append(
                news_id
            )

            if len(retrieved) == candidate_k:
                break

        return retrieved

    def recommend(
        self,
        user_id: str,
        k: int = 10,
        candidate_k: int = 100,
    ) -> dict:

        if k <= 0:
            raise ValueError(
                "k must be greater than 0."
            )

        if candidate_k < k:
            raise ValueError(
                "candidate_k must be >= k."
            )

        history = self.user_histories.get(
            user_id
        )

        if not history:
            return self._popularity_fallback(
                user_id,
                k,
            )

        context = build_user_ranking_context(
            history,
            self.article_embedding_map,
            self.article_categories,
            self.article_subcategories,
        )

        if context is None:
            return self._popularity_fallback(
                user_id,
                k,
            )

        history_ids = set(
            parse_history(
                history
            )
        )

        candidate_ids = (
            self._retrieve_candidates(
                context.user_embedding,
                exclude_ids=history_ids,
                candidate_k=candidate_k,
            )
        )

        feature_rows = []

        valid_candidate_ids = []

        for news_id in candidate_ids:

            features = (
                build_candidate_features(
                    news_id=news_id,
                    context=context,
                    article_embedding_map=(
                        self.article_embedding_map
                    ),
                    article_categories=(
                        self.article_categories
                    ),
                    article_subcategories=(
                        self.article_subcategories
                    ),
                    popularity=(
                        self.popularity
                    ),
                )
            )

            if features is None:
                continue

            feature_rows.append(
                features
            )

            valid_candidate_ids.append(
                news_id
            )

        if not feature_rows:
            return self._popularity_fallback(
                user_id,
                k,
            )

        X_candidates = np.vstack(
            feature_rows
        ).astype(
            np.float32,
            copy=False,
        )

        scores = self.ranker.predict_proba(
            X_candidates
        )[:, 1]

        order = np.argsort(
            -scores,
            kind="stable",
        )

        recommendations = []

        for rank_index in order[:k]:

            recommendations.append(
                {
                    "news_id": (
                        valid_candidate_ids[
                            rank_index
                        ]
                    ),
                    "score": float(
                        scores[
                            rank_index
                        ]
                    ),
                }
            )

        return {
            "user_id": user_id,
            "strategy": "personalized",
            "recommendations": (
                recommendations
            ),
        }