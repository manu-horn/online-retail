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

# Reuse the inference builder, adding only the locked training dependency group.
FROM builder AS training-builder
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --group training --no-editable

FROM builder AS serving-builder
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --group serving --no-editable

FROM base AS runtime
RUN groupadd --gid 10001 app && useradd --uid 10001 --gid app --no-create-home app
ENV PATH="/app/.venv/bin:$PATH"
USER 10001:10001

FROM runtime AS training
ARG VCS_REF=unknown
LABEL org.opencontainers.image.title="Online Retail fixed-fit training" \
      org.opencontainers.image.revision=$VCS_REF
ENV ONLINE_RETAIL_CODE_REVISION=$VCS_REF \
    MLFLOW_ENABLE_TELEMETRY=false \
    MLFLOW_DISABLE_AGENT_HINT=1
COPY --from=training-builder /app/.venv /app/.venv
COPY scripts/reproduce_fit.py ./scripts/
COPY configs/tracking.toml ./configs/
# Preserve source/configuration for MLflow provenance archives.
COPY src/ ./src/
COPY pyproject.toml uv.lock ./
ENTRYPOINT ["python", "scripts/reproduce_fit.py"]
CMD ["--help"]

# SageMaker adapters use root as required by the managed container contract.
FROM runtime AS sagemaker-training
USER 0:0
ARG VCS_REF=unknown
LABEL org.opencontainers.image.title="Online Retail SageMaker fixed-fit training" \
      org.opencontainers.image.revision=$VCS_REF
ENV ONLINE_RETAIL_CODE_REVISION=$VCS_REF
COPY --from=builder /app/.venv /app/.venv
ENTRYPOINT ["python", "-m", "online_retail.sagemaker"]
CMD ["train"]

FROM runtime AS sagemaker-inference
USER 0:0
ARG VCS_REF=unknown
LABEL org.opencontainers.image.title="Online Retail SageMaker batch inference" \
      org.opencontainers.image.revision=$VCS_REF
COPY --from=serving-builder /app/.venv /app/.venv
EXPOSE 8080
ENTRYPOINT ["python", "-m", "online_retail.sagemaker"]
CMD ["serve"]

# Keep inference last: a build without --target still produces the small scoring image.
FROM runtime AS inference
ARG VCS_REF=unknown
LABEL org.opencontainers.image.title="Online Retail batch inference" \
      org.opencontainers.image.revision=$VCS_REF
COPY --from=builder /app/.venv /app/.venv
COPY scripts/score.py scripts/package_model.py ./scripts/
ENTRYPOINT ["python", "scripts/score.py"]
CMD ["--help"]
