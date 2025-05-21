from fastapi import APIRouter
from datetime import datetime
from Application.schema.predict import PredictionRequest, PredictionResult, SensorData
from Application.services.predictor import analyze_prediction

router = APIRouter()

@router.post("/predict", response_model=PredictionResult)
async def predict_from_backend(payload: dict):
    readings = payload["mlSensorReadings"]
    growth_stage = payload["plantGrowthStage"]

    # Build SensorData directly
    sensor_data = SensorData(
        temperature=next(r["value"] for r in readings if r["sensorName"].lower() == "temperature"),
        humidity=next(r["value"] for r in readings if r["sensorName"].lower() == "humidity"),
        soil_moisture=next(r["value"] for r in readings if "soilmoisture" in r["sensorName"].lower()),
        light=next(r["value"] for r in readings if r["sensorName"].lower() == "light"),
        hour_of_day=datetime.now().hour,
        growthStage=growth_stage
    )

    request = PredictionRequest(current=sensor_data)
    result = await analyze_prediction(request)
    return result
