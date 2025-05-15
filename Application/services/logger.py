# Application/services/logger.py

import os
import requests
from Application.schema.predict import SensorData, PredictionResult


# Backend URL to send prediction logs to (must be set in your .env file)
BACKEND_API_URL = os.getenv("BACKEND_PREDICTION_LOG_URL")  # Example: http://backend:5000/api/predictions

def log_prediction(sensor: SensorData, result: PredictionResult):
    """
    Sends the prediction result and sensor data to the backend server
    for storage in the central database.

    Parameters:
    - sensor: SensorData containing type, value, timestamp
    - result: PredictionResult containing status, suggestion, trend

    The backend must expose an endpoint that accepts this payload.
    """
    payload = {
        "timestamp": result.timestamp.isoformat(),
        "sensorType": sensor.sensorType,
        "value": sensor.value,
        "status": result.status,
        "suggestion": result.suggestion,
        "trendAnalysis": result.trendAnalysis
    }

    # Attempt to send log to backend
    try:
        response = requests.post(BACKEND_API_URL, json=payload)
        response.raise_for_status()
    except Exception as e:
        print(f"[ERROR] Failed to send prediction to backend: {e}")
