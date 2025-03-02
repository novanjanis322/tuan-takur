FROM python:3.11-slim AS build-env
WORKDIR /app

COPY API/ .
COPY src/utils/validators.py src/utils/validators.py

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "app.py"]
