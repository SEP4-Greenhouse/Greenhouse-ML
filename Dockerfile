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
RUN pip install --no-cache-dir \
    numpy==1.26.0 --only-binary=:all: \
    scikit-learn==1.6.1 --only-binary=:all:



# Copy only requirements to leverage layer caching
COPY requirements.txt .

# Install remaining project-specific dependencies
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Create directory for models if it doesn't exist
RUN mkdir -p Application/trained_models

# Now copy the full source code
COPY . .

# Health check to ensure the service is running properly
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Expose application port
EXPOSE 8000

# Run application with 1 worker per CPU
CMD ["uvicorn", "Application.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]