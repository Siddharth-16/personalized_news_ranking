from pathlib import Path

import faiss
import numpy as np


class FaissArticleRetriever:
    """
    Exact cosine-similarity retrieval over article embeddings.

    Article and user embeddings are L2-normalized, allowing
    IndexFlatIP inner product to act as cosine similarity.
    """

    def __init__(
        self,
        article_ids: list[str],
        article_embeddings: np.ndarray,
    ):
        if article_embeddings.ndim != 2:
            raise ValueError(
                "article_embeddings must be two-dimensional."
            )

        if len(article_ids) != len(article_embeddings):
            raise ValueError(
                "article_ids and article_embeddings "
                "must have the same length."
            )

        embeddings = np.asarray(
            article_embeddings,
            dtype=np.float32,
        )

        embeddings = np.ascontiguousarray(embeddings)

        # Normalize defensively even though they were normalized
        # during article encoding.
        norms = np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True,
        )

        if np.any(norms == 0):
            raise ValueError(
                "Article embeddings contain zero vectors."
            )

        embeddings = embeddings / norms

        self.article_ids = article_ids
        self.embedding_dim = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(
            self.embedding_dim
        )

        self.index.add(embeddings)

    def retrieve(
        self,
        user_embedding: np.ndarray,
        k: int,
        exclude_ids: set[str] | None = None,
    ) -> tuple[list[str], list[float]]:
        """
        Retrieve Top-K articles for a user.

        Articles already present in the user's history can
        optionally be excluded.
        """

        if k <= 0:
            raise ValueError(
                "k must be greater than 0."
            )

        query = np.asarray(
            user_embedding,
            dtype=np.float32,
        )

        if query.ndim != 1:
            raise ValueError(
                "user_embedding must be one-dimensional."
            )

        if len(query) != self.embedding_dim:
            raise ValueError(
                "User and article embedding dimensions differ."
            )

        norm = np.linalg.norm(query)

        if norm == 0:
            raise ValueError(
                "user_embedding cannot be a zero vector."
            )

        query = query / norm
        query = np.ascontiguousarray(
            query.reshape(1, -1)
        )

        exclude_ids = exclude_ids or set()

        # Retrieve additional items so removing articles already
        # seen by the user still leaves approximately K results.
        search_k = min(
            self.index.ntotal,
            k + len(exclude_ids),
        )

        scores, indices = self.index.search(
            query,
            search_k,
        )

        retrieved_ids = []
        retrieved_scores = []

        for index, score in zip(
            indices[0],
            scores[0],
        ):
            if index < 0:
                continue

            news_id = self.article_ids[index]

            if news_id in exclude_ids:
                continue

            retrieved_ids.append(news_id)
            retrieved_scores.append(float(score))

            if len(retrieved_ids) == k:
                break

        return retrieved_ids, retrieved_scores

    def save_index(
        self,
        path: str | Path,
    ) -> None:
        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        faiss.write_index(
            self.index,
            str(path),
        )