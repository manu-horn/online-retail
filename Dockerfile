# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.12.15@sha256:62f8c047d0a0e9ece6b53fc63df902585a67a47a7f318ddec4a37db586edc8e3 AS uv
FROM python:3.12-slim-bookworm@sha256:44cf1793efa7c7a476bb0f2cff81443f4a072c90c1cb127e2924933daa3e8c64 AS base

# XGBoost requires the OpenMP runtime on Debian slim.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/cache \
    OPENBLAS_NUM_THREADS=2 \
    OMP_NUM_THREADS=2

FROM base AS builder
# Fail before large downloads if the daemon's filesystem cannot hold the build.
RUN python -c "import shutil; free = shutil.disk_usage('/').free; assert free >= 4 * 1024**3, f'Docker build needs at least 4 GiB free; available: {free / 1024**3:.2f} GiB. Free space or move Docker storage.'"
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

FROM base AS inference
ARG VCS_REF=unknown
LABEL org.opencontainers.image.title="Online Retail batch inference" \
      org.opencontainers.image.revision=$VCS_REF
RUN groupadd --gid 10001 app && useradd --uid 10001 --gid app --no-create-home app
COPY --from=builder /app/.venv /app/.venv
COPY scripts/score.py scripts/package_model.py ./scripts/
ENV PATH="/app/.venv/bin:$PATH"
USER 10001:10001
ENTRYPOINT ["python", "scripts/score.py"]
CMD ["--help"]
