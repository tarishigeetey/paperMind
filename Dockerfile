# ── Stage 1: Builder ──────────────────────────────────────────────
# Uses UV's official image to install dependencies
# Like a Maven build stage — compiles and packages
FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS base

WORKDIR /app

# Copy dependency files first — Docker layer caching
# If pyproject.toml hasn't changed, this layer is cached
# Like Maven's dependency:resolve before copying source
COPY pyproject.toml uv.lock ./

# UV performance flags:
# UV_COMPILE_BYTECODE=1  → generates .pyc files → faster startup
# UV_LINK_MODE=copy      → silences hard-link warnings in Docker
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Install only production dependencies (--no-dev)
# --frozen means use exact versions from uv.lock — no surprises
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=/app/uv.lock \
    --mount=type=bind,source=pyproject.toml,target=/app/pyproject.toml \
    uv sync --frozen --no-dev

# Copy source code AFTER dependencies
# This way code changes don't invalidate the dependency cache layer
COPY src /app/src

# ── Stage 2: Final image ──────────────────────────────────────────
# Slim Python image — no UV, no build tools, just the app
# Final image is much smaller than if we used the builder image
FROM python:3.12.8-slim AS final

EXPOSE 8000

# PYTHONUNBUFFERED=1 → logs appear immediately, not buffered
# Critical for seeing logs in Docker — without it logs show up late
ENV PYTHONUNBUFFERED=1
ARG VERSION=0.1.0
ENV APP_VERSION=$VERSION

WORKDIR /app

# Copy ONLY the installed app from the builder stage
# No UV, no build tools, no cache — clean minimal image
COPY --from=base /app /app

# Add the venv to PATH so Python finds our packages
ENV PATH="/app/.venv/bin:$PATH"

# Start the server with 4 workers
# Like deploying a Spring Boot jar with multiple threads
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]