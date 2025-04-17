from fastapi import FastAPI
from datetime import datetime
from schemas import SensorData, PredictionResult

app = FastAPI()

@app.post("/predict", response_model=PredictionResult)
async def predict(data: SensorData):
    # Mock logic
    if data.sensorType == "Temperature" and data.value > 30:
        status = "warning"
        suggestion = "Activate cooling"
    elif data.sensorType == "Humidity" and data.value < 30:
        status = "warning"
        suggestion = "Start irrigation"
    else:
        status = "normal"
        suggestion = "No action needed"

    return PredictionResult(
        timestamp=datetime.utcnow(),
        status=status,
        suggestion=suggestion
    )
