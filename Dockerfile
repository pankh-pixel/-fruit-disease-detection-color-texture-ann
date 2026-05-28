# ============================================================
#  Fruit Disease Detection — Docker Image
#  Base: python:3.10-slim  (CPU-only TensorFlow build)
#  For GPU support swap base to tensorflow/tensorflow:2.13.0-gpu
# ============================================================

FROM python:3.10-slim

LABEL maintainer="your.email@example.com"
LABEL description="Fruit Disease Detection — Color, Texture & ANN Pipeline"
LABEL version="1.0.0"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create runtime directories and fix ownership
RUN mkdir -p data/raw data/processed models/checkpoints logs && \
    chown -R appuser:appuser /app

USER appuser

# Expose MLflow tracking port (optional)
EXPOSE 5000

# Default command: show help
CMD ["python", "train.py", "--help"]
