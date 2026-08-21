from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from pydantic import BaseModel

from src.serving.engine import (
    RecommendationEngine,
)


class Recommendation(BaseModel):
    news_id: str
    score: float


class RecommendationResponse(BaseModel):
    user_id: str
    strategy: str
    recommendations: list[Recommendation]


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):

    app.state.engine = (
        RecommendationEngine()
    )

    yield

    app.state.engine = None


app = FastAPI(
    title="Personalized News Ranking Engine",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:

    return {
        "status": "ok"
    }


@app.get(
    "/recommend/{user_id}",
    response_model=RecommendationResponse,
)
def recommend(
    user_id: str,
    request: Request,
) -> dict:

    engine: RecommendationEngine = (
        request.app.state.engine
    )

    return engine.recommend(
        user_id=user_id,
        k=10,
        candidate_k=100,
    )