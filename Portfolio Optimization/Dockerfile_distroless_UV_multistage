FROM python:3.11-slim AS build-env
WORKDIR /app

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy your application files
COPY app.py .
COPY src/ src/
COPY pyproject.toml .
COPY uv.lock .
# Create virtual environment and sync dependencies with UV
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN uv sync --frozen --no-cache

FROM gcr.io/distroless/python3
WORKDIR /app

COPY --from=build-env /opt/venv /opt/venv
COPY --from=build-env /app /app

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app/.venv/lib/python3.11/site-packages:/app"

ENTRYPOINT ["/usr/bin/python3.11", "/app/app.py"]