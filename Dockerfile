FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONHASHSEED=random

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libatlas-base-dev \
    libjpeg-dev \
    libpng-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel

# 🔥 Install critical libs from binary wheels
RUN pip install --prefer-binary --no-cache-dir \
    numpy==1.26.0 \
    pandas==2.2.2 \
    scikit-learn==1.6.1 \
    matplotlib \
    joblib

# Optional: other app-specific deps
RUN pip install --no-cache-dir -r requirements.txt || true

COPY . .

EXPOSE 8000

CMD ["uvicorn", "Application.main:app", "--host", "0.0.0.0", "--port", "8000"]
