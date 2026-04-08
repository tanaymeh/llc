FROM python:3.13-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy
ENV PATH="/app/.venv/bin:$PATH"
ARG UV_VERSION=0.10.11

RUN apt-get update && apt-get install -y --no-install-recommends ripgrep \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir "uv==${UV_VERSION}" ast-grep-cli

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY llc ./llc
RUN uv sync --frozen --no-dev

ENTRYPOINT ["python", "-m", "llc"]
