# Application/Api/sensor.py

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from Application.schema.predict import SensorData, PredictionResult, PredictionRequest
from Application.services.predictor import analyze_prediction
from Application.services.logger import log_prediction
from Application.database import get_db

# Initialize FastAPI router for handling sensor-related API routes
router = APIRouter()

@router.post("/sensor", response_model=PredictionResult)
def receive_sensor_data(data: SensorData, db: Session = Depends(get_db)):
    """
    Endpoint: POST /sensor
    Accepts current sensor readings, runs prediction logic, and logs the result.

    Workflow:
    1. Wrap the received SensorData into a PredictionRequest.
    2. Use the ML logic (analyze_prediction) to generate a result.
    3. Log the prediction to the backend database (via SQLAlchemy).
    
    Parameters:
    - data: Incoming sensor data (type, value, timestamp) from the frontend or device.
    - db: SQLAlchemy database session (injected automatically by FastAPI).

    Returns:
    - A structured prediction result with timestamp, status, suggestion, and optional trend info.
    """
    # Wrap the sensor input in a standard prediction request model
    request = PredictionRequest(current=data)

    # Analyze the incoming sensor data using your ML logic
    prediction = analyze_prediction(request)

    # Log the result to the backend database for tracking and frontend usage
    log_prediction(data, prediction, db)

    # Return prediction back to frontend or caller
    return prediction
