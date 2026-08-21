import pytest
from pydantic import ValidationError

from src.serving.app import (
    RecommendationResponse,
)


def test_personalized_response_contract():
    response = RecommendationResponse(
        user_id="U1",
        strategy="personalized",
        recommendations=[
            {
                "news_id": "N1",
                "score": 0.75,
            },
            {
                "news_id": "N2",
                "score": 0.50,
            },
        ],
    )

    assert response.user_id == "U1"
    assert response.strategy == "personalized"

    assert (
        len(response.recommendations)
        == 2
    )

    assert (
        response.recommendations[0].news_id
        == "N1"
    )


def test_popularity_fallback_response_contract():
    response = RecommendationResponse(
        user_id="UNKNOWN",
        strategy="popularity_fallback",
        recommendations=[
            {
                "news_id": "N1",
                "score": 100.0,
            }
        ],
    )

    assert (
        response.strategy
        == "popularity_fallback"
    )


def test_response_contract_requires_news_id():
    with pytest.raises(
        ValidationError
    ):
        RecommendationResponse(
            user_id="U1",
            strategy="personalized",
            recommendations=[
                {
                    "score": 0.5,
                }
            ],
        )