# =====================================================================
# Production Dockerfile for OmniRAG Service
# =====================================================================

FROM python:3.12-slim

# Install system dependencies for psycopg2 and curl health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-warm FastEmbed model cache into Docker layer to avoid cold-start delays
RUN python3 -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5')"

# Copy application files
COPY . .

# Expose API port
EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start production server
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
