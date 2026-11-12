# Single image for Hugging Face Spaces (port 7860) and Render ($PORT).
# Local compose overrides PORT to 8741.

FROM node:22-alpine AS frontend
WORKDIR /src
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
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
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
