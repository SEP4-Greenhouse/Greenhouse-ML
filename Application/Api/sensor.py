from fastapi import APIRouter
from Application.schema.predict import SensorData, PredictionResult, PredictionRequest
from Application.services.predictor import analyze_prediction

router = APIRouter()

@router.post("/sensor", response_model=PredictionResult)
def receive_sensor_data(data: SensorData):
    """
    Endpoint to receive real-time sensor input, analyze it, and return a prediction.
    This endpoint does not log or save data directly. Logging must be handled by the backend.
    """
    # Wrap sensor reading in the unified prediction format
    request = PredictionRequest(current=data)

    # Run ML analysis on the current sensor data
    prediction = analyze_prediction(request)

    # ⛔ DO NOT log prediction here — backend will handle saving
    return prediction
