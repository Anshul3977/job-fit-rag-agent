# Single image for Hugging Face Spaces (port 7860) and Render ($PORT).
# Local compose overrides PORT to 8741.

FROM node:22-alpine AS frontend
WORKDIR /src
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim AS runtime
# Render's free web instance has 512 MB RAM. One process, one worker, and
# capped native thread pools keep MiniLM + Chroma inside that. A scored fit
# on this image measured about 280 MiB RSS with the model loaded.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2 \
    WEB_CONCURRENCY=1 \
    PORT=7860 \
    ENV=prod \
    DEBUG_ROUTES=false \
    EMBEDDING_CACHE_DIR=/data/models \
    CHROMA_DIR=/data/chroma \
    DATABASE_URL=sqlite:////data/jobfit.db
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY eval ./eval
COPY --from=frontend /src/dist ./frontend/dist
EXPOSE 7860
# Render and Spaces inject PORT. Fall back to 7860 for Hugging Face.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
