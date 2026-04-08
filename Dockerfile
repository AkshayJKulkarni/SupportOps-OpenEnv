# Dockerfile for SupportOps OpenEnv - Hugging Face Spaces
FROM python:3.11-slim

# Set environment variables for production
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace \
    PORT=7860 \
    HF_HOME=/workspace/.cache

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash --user-group --uid 1001 app \
    && mkdir -p /workspace \
    && chown -R app:app /workspace

# Switch to non-root user
USER app

# Set working directory
WORKDIR /workspace

# Copy requirements first for better Docker layer caching
COPY --chown=app:app requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy application code
COPY --chown=app:app . .

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Expose port for Hugging Face Spaces
EXPOSE $PORT

# Start the FastAPI application with uvicorn
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--loop", "uvloop"]
