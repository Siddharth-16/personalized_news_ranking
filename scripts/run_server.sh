#!/usr/bin/env bash

set -euo pipefail

# Limit native numerical-library threading.
#
# The recommendation API handles request-level concurrency itself.
# Allowing scikit-learn/OpenMP/BLAS to create multiple native threads
# per request caused CPU oversubscription and severe tail-latency
# degradation under concurrent load.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

python -m uvicorn src.serving.app:app \
    --host 127.0.0.1 \
    --port 8000