# ============================================================
# CPU downloader image
# Role: kaggle API download + multithreaded acceleration + unzip
# Base image: python:3.11-slim (CPU is enough; no GPU needed)
# ============================================================
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps: aria2c multithreaded download, unzip, curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    aria2 \
    unzip \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install the kaggle CLI
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir kaggle

# Working directory
WORKDIR /data/kaggle

# Default permissions (can be overridden by the entrypoint)
RUN chmod -R 777 /data/kaggle 2>/dev/null || true

# Download script is mounted to /scripts via a volume
# Default command (can be overridden by docker-compose command)
CMD ["bash", "/scripts/download.sh"]
