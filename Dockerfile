# ==============================================================================
# Multi-Stage Dockerfile for Django Model Monitoring Dashboard
# Stage 1: Build dependency wheels in an isolated virtual environment
# Stage 2: Pristine runtime image running as unprivileged non-root user
# ==============================================================================

# ----------------- Stage 1: Dependency Builder -----------------
FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

COPY requirements.txt .

RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ----------------- Stage 2: Production Runtime -----------------
FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8000

# Security: Create non-root system user and group (UID 1001)
RUN groupadd -g 1001 appuser && \
    useradd -u 1001 -g appuser -s /bin/bash -m appuser

WORKDIR /app

# Copy isolated virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application source code (filtered strictly via .dockerignore)
COPY . /app/

# Ensure writable data, artifact, and static directories exist with non-root ownership
RUN mkdir -p /app/data /app/saved_models /app/saved_datasets /app/staticfiles && \
    chown -R appuser:appuser /app /opt/venv

# Run container as unprivileged user
USER appuser

EXPOSE 8000

# Container Healthcheck (verifies Django serves HTTP 200 on port 8000)
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/').read()" || exit 1

# Production WSGI Server via Gunicorn (concurrency: 2 workers, 2 threads)
CMD ["sh", "-c", "python manage.py migrate --noinput && gunicorn core_project.wsgi:application --bind 0.0.0.0:8000 --workers 2 --threads 2 --timeout 60"]
