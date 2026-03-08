FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy
ENV PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY main.py config.py models.py ./
COPY agent ./agent
COPY commands ./commands
COPY ui ./ui

ENTRYPOINT ["python", "/app/main.py"]
