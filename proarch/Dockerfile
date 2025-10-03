# Use Python 3.12 slim image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONHASHSEED=random

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Change ownership of the working directory
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Copy requirements first for better caching
COPY --chown=appuser:appuser requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Add user bin to PATH
ENV PATH="/home/appuser/.local/bin:$PATH"

# Copy source code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser data/ ./data/
COPY --chown=appuser:appuser .env* ./
COPY --chown=appuser:appuser frontend.py ./

# Create data directories
RUN mkdir -p data/vector_store data/evaluation data/raw data/reference

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - API server
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
