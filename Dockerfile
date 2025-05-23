FROM python:3.12-slim-bullseye

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install pip tools and setuptools first
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install scientific packages separately with binary preference
RUN pip install --no-cache-dir --prefer-binary numpy==1.26.0 pandas==2.0.3 scikit-learn==1.6.1

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "Application.main:app", "--host", "0.0.0.0", "--port", "8000"]