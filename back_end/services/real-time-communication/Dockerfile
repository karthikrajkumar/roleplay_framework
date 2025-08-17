FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY services/real-time-communication/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY services/real-time-communication/ .

# Copy shared library (Note: shared library should be copied from build context)
# The shared directory should be copied at build time using build context
# Example: docker build -f services/real-time-communication/Dockerfile --build-arg SHARED_PATH=./shared .
ARG SHARED_PATH=./shared
COPY ${SHARED_PATH} ./shared

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Set environment variables
ENV PYTHONPATH=/app:/app/shared
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "main.py"]
