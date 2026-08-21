# Serving and Inference Configuration

## Recommendation Service

The recommendation API serves the frozen two-stage ranking pipeline:

1. Mean-pool pretrained article embeddings into a user representation.
2. Retrieve Top-100 candidates with FAISS `IndexFlatIP`.
3. Generate ranking features for the retrieved candidates.
4. Score candidates with the trained `HistGradientBoostingClassifier`.
5. Return the Top-10 ranked articles.

The FastAPI service loads the article embeddings, FAISS index, ranking model,
metadata, and user state once during application startup.

## Running the Service

Use:

```bash
./scripts/run_server.sh
```
