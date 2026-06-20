# ---------- Stage 1: dependency builder ----------
FROM python:3.11-slim AS builder

# Copy uv binary from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /build

# Copy dependency manifest only — maximises cache reuse when code changes
COPY pyproject.toml uv.lock ./

# Install all runtime deps into .venv; skip project install (no build backend)
RUN uv sync --frozen --no-dev --no-install-project

# ---------- Stage 2: runtime ----------
FROM python:3.11-slim AS runtime

# Non-root user
RUN useradd -m -u 1000 app

WORKDIR /app

# Virtualenv from builder — no uv or build tools in final image
COPY --from=builder /build/.venv /app/.venv

# Application code only — notebooks, tests, .git excluded via .dockerignore
COPY main.py ./
COPY src/ ./src/

# Persistent data directory — mount as volume at /app/data
RUN mkdir -p /app/data && chown app:app /app/data

USER app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Default DB_PATH points inside the volume mount; override via --env-file if needed
    DB_PATH=/app/data/portfolio.db

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
