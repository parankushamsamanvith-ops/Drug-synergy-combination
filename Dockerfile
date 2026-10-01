# Multi-stage production Dockerfile for SynRes-AI Drug Synergy Predictor
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Install system dependencies (for building C extensions if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency specifications first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code, data, and models
COPY src/ ./src/
COPY models/ ./models/
COPY data/signatures/ ./data/signatures/
COPY data/benchmarks/ ./data/benchmarks/
COPY static/ ./static/
COPY backend.py .
COPY app.py .

# Expose port (default 8000 for FastAPI / Web UI)
EXPOSE 8000

# Command to launch the full-stack FastAPI application
CMD ["sh", "-c", "uvicorn backend:app --host 0.0.0.0 --port ${PORT:-8000}"]
