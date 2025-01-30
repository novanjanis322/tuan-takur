FROM python:3.11-slim AS build-env
WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY gke_deployment/app.py .
COPY src/ src/
COPY gke_deployment/pyproject.toml .
COPY gke_deployment/uv.lock .

RUN python -m venv /opt/venv \
&& . /opt/venv/bin/activate \
&& uv sync --frozen --no-cache \
&& find /opt/venv -name "*.pyc" -delete \
&& find /opt/venv -name "__pycache__" -delete

FROM gcr.io/distroless/python3
WORKDIR /app

COPY --from=build-env /opt/venv /opt/venv
COPY --from=build-env /app /app

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app/.venv/lib/python3.11/site-packages:/app"
EXPOSE 8080
ENTRYPOINT ["/usr/bin/python3.11", "/app/app.py"]