college\semi 4\Sep4\Greenhouse-ML\README.md
# Greenhouse ML API

Machine learning service for predicting optimal watering times for greenhouse plants.

## Features

- 🌱 Predicts hours until next watering is needed
- 📊 Uses Random Forest regression model
- 🔄 Handles various sensor data inputs
- 🚀 Packaged as a Docker container
- ✅ 98% accurate predictions (R²: 0.98)

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Python 3.12+ (for local development)

### Running the Service

```bash
# Using Docker
docker compose up

# Using Python locally
pip install -r [requirements.txt](http://_vscodecontentref_/3)
uvicorn Application.main:app --reload