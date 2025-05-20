import asyncio
import os
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from Application.schema.predict import PredictionRequest, SensorData
from Application.services.predictor import analyze_prediction
import httpx

BACKEND_LATEST_DATA_URL = os.getenv("BACKEND_LATEST_SENSOR_URL", "http://backend:5001/api/ml/latest-data")

scheduler = AsyncIOScheduler()

async def fetch_all_sensor_data():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(BACKEND_LATEST_DATA_URL)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        print(f"[SCHEDULER] Failed to fetch latest sensor data: {e}")
        return []

async def run_scheduled_predictions():
    sensor_batches = await fetch_all_sensor_data()

    for entry in sensor_batches:
        try:
            request = PredictionRequest(
                current=SensorData(**entry["current"]),
                history=entry.get("history", [])
            )
            result = await analyze_prediction(request)
            print(f"[SCHEDULER] Prediction complete: {request.current.sensorType} → {result.status}")
        except Exception as e:
            print(f"[SCHEDULER] Error processing sensor: {e}")

def start_scheduler():
    scheduler.add_job(run_scheduled_predictions, "interval", minutes=5)
    scheduler.start()
