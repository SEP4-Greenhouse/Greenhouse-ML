import os
import httpx
from typing import List
from datetime import datetime
from Application.schema.predict import SensorHistoryEntry, PredictionResult, SensorData

BACKEND_API_URL = os.getenv("BACKEND_PREDICTION_LOG_URL", "http://backend:5001/api/predictions")
BACKEND_HISTORY_URL = os.getenv("BACKEND_HISTORY_URL", "http://backend:5001/api/sensors/history")

async def fetch_history(sensor_type: str, limit: int = 20) -> List[SensorHistoryEntry]:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(BACKEND_HISTORY_URL, params={"sensorType": sensor_type, "limit": limit})
            response.raise_for_status()
            return [SensorHistoryEntry(**item) for item in response.json()]
    except Exception as e:
        print(f"[ML] Failed to fetch history: {e}")
        return []

async def log_prediction(sensor: SensorData, result: PredictionResult) -> bool:
    payload = {
        "timestamp": result.timestamp.isoformat(),
        "sensorType": sensor.sensorType,
        "value": sensor.value,
        "status": result.status,
        "suggestion": result.suggestion,
        "trendAnalysis": result.trendAnalysis
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(BACKEND_API_URL, json=payload)
            response.raise_for_status()
            return True
    except Exception as e:
        print(f"[ML] Failed to log prediction: {e}")
        return False
