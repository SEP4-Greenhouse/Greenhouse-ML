FROM python:3.12-slim

# === Environment settings ===
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONHASHSEED=random
ENV PYTHONPATH=/app  # 👈 Ensures modules are importable

# === System dependencies ===
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libatlas-base-dev \
    libjpeg-dev \
    libpng-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# === Set working directory ===
WORKDIR /app

# === Install requirements ===
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel

# === Critical packages first ===
RUN pip install --prefer-binary --no-cache-dir \
    numpy==1.26.0 \
    pandas==2.2.2 \
    scikit-learn==1.6.1 \
    matplotlib \
    joblib

# === Optional project-specific dependencies ===
RUN pip install --no-cache-dir -r requirements.txt || true

# === Copy codebase ===
COPY . .

# === Train model INSIDE container ===
RUN rm -f Application/trained_models/*.pkl && \
    mkdir -p Application/trained_models && \
    python Application/training/training_models/train_randomForest.py && \
    ls -lh Application/trained_models

# === Expose port and start server ===
EXPOSE 8000
CMD ["uvicorn", "Application.main:app", "--host", "0.0.0.0", "--port", "8000"]
