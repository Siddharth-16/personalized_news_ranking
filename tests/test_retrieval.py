import numpy as np
import pytest

from src.retrieval.faiss_retriever import (
    FaissArticleRetriever,
)
from src.retrieval.user_embeddings import (
    build_article_embedding_map,
    build_user_embedding,
)


def test_build_article_embedding_map():
    article_ids = [
        "N1",
        "N2",
    ]

    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    embedding_map = (
        build_article_embedding_map(
            article_ids,
            embeddings,
        )
    )

    assert set(embedding_map) == {
        "N1",
        "N2",
    }

    np.testing.assert_array_equal(
        embedding_map["N1"],
        embeddings[0],
    )


def test_article_embedding_map_rejects_length_mismatch():
    article_ids = [
        "N1",
    ]

    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        build_article_embedding_map(
            article_ids,
            embeddings,
        )


def test_build_user_embedding_mean_pools_and_normalizes():
    embedding_map = {
        "N1": np.asarray(
            [1.0, 0.0],
            dtype=np.float32,
        ),
        "N2": np.asarray(
            [0.0, 1.0],
            dtype=np.float32,
        ),
    }

    user_embedding = build_user_embedding(
        "N1 N2",
        embedding_map,
    )

    expected = np.asarray(
        [1.0, 1.0],
        dtype=np.float32,
    )

    expected /= np.linalg.norm(
        expected
    )

    np.testing.assert_allclose(
        user_embedding,
        expected,
        atol=1e-6,
    )

    assert np.linalg.norm(
        user_embedding
    ) == pytest.approx(
        1.0
    )


def test_build_user_embedding_handles_empty_history():
    embedding_map = {
        "N1": np.asarray(
            [1.0, 0.0],
            dtype=np.float32,
        ),
    }

    assert (
        build_user_embedding(
            None,
            embedding_map,
        )
        is None
    )


def test_build_user_embedding_ignores_unknown_articles():
    embedding_map = {
        "N1": np.asarray(
            [1.0, 0.0],
            dtype=np.float32,
        ),
    }

    user_embedding = build_user_embedding(
        "UNKNOWN N1",
        embedding_map,
    )

    np.testing.assert_allclose(
        user_embedding,
        np.asarray(
            [1.0, 0.0],
            dtype=np.float32,
        ),
    )


def test_faiss_retriever_returns_nearest_articles():
    article_ids = [
        "N1",
        "N2",
        "N3",
    ]

    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )

    retriever = FaissArticleRetriever(
        article_ids,
        embeddings,
    )

    user_embedding = np.asarray(
        [1.0, 0.0],
        dtype=np.float32,
    )

    retrieved_ids, scores = (
        retriever.retrieve(
            user_embedding,
            k=2,
        )
    )

    assert len(retrieved_ids) == 2
    assert len(scores) == 2

    assert retrieved_ids[0] == "N1"


def test_faiss_retriever_excludes_seen_articles():
    article_ids = [
        "N1",
        "N2",
        "N3",
    ]

    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )

    retriever = FaissArticleRetriever(
        article_ids,
        embeddings,
    )

    retrieved_ids, _ = (
        retriever.retrieve(
            np.asarray(
                [1.0, 0.0],
                dtype=np.float32,
            ),
            k=2,
            exclude_ids={"N1"},
        )
    )

    assert "N1" not in retrieved_ids
    assert len(retrieved_ids) == 2


def test_faiss_retriever_rejects_invalid_k():
    retriever = FaissArticleRetriever(
        ["N1"],
        np.asarray(
            [[1.0, 0.0]],
            dtype=np.float32,
        ),
    )

    with pytest.raises(ValueError):
        retriever.retrieve(
            np.asarray(
                [1.0, 0.0],
                dtype=np.float32,
            ),
            k=0,
        )