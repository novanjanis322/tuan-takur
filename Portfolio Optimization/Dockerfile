FROM python:3.9-slim
FROM gurobi/python:latest

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p /app/data/raw /app/data/processed /app/data/cache

# Copy requirements first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY src/ ./src/
COPY data/raw/LQ45_benchmark.csv ./data/raw/

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Run the web service using Granian
CMD exec granian --interface asgi --host 0.0.0.0 --port $PORT --workers 1 src.app:app