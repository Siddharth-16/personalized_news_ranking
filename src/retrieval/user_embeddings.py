import numpy as np

from src.data.parse import parse_history


def build_article_embedding_map(
    article_ids: list[str],
    article_embeddings: np.ndarray,
) -> dict[str, np.ndarray]:
    if len(article_ids) != len(article_embeddings):
        raise ValueError(
            "article_ids and article_embeddings must have the same length."
        )

    return {
        news_id: embedding
        for news_id, embedding in zip(
            article_ids,
            article_embeddings,
        )
    }


def build_user_embedding(
    history: str,
    article_embedding_map: dict[str, np.ndarray],
) -> np.ndarray | None:
    """
    Build a user representation by mean-pooling embeddings
    of articles in the user's click history.

    Returns None when no usable history is available.
    """

    history_ids = parse_history(history)

    history_embeddings = [
        article_embedding_map[news_id]
        for news_id in history_ids
        if news_id in article_embedding_map
    ]

    if not history_embeddings:
        return None

    user_embedding = np.mean(
        history_embeddings,
        axis=0,
    ).astype(np.float32)

    norm = np.linalg.norm(user_embedding)

    if norm == 0:
        return None

    return user_embedding / norm