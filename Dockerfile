FROM python:3.12-slim

# Set environment variables to reduce image size and improve performance
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONHASHSEED=random

# Install build dependencies for numpy and pandas (slim doesn't include these)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libatlas-base-dev \
    libjpeg-dev \
    libpng-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and install core scientific packages first for cache efficiency
RUN pip install --no-cache-dir --upgrade pip==24.0 setuptools==70.0.0 wheel==0.43.0

# Pre-install heavy scientific libs to avoid rebuilding from source
# Using EXACTLY the same versions as in your main Dockerfile
RUN pip install --no-cache-dir \
    numpy==1.26.0 --only-binary=:all: \
    scikit-learn==1.6.1 --only-binary=:all:

# Copy only requirements to leverage layer caching
COPY requirements.txt .

# Install remaining project-specific dependencies
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Create directories for models and data
RUN mkdir -p Application/trained_models Application/data

# Now copy the full source code
COPY . .

# Default command runs the training script
CMD ["uvicorn", "Application.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]