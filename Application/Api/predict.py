# Application/Api/predict.py

from fastapi import APIRouter
from Application.schema.predict import PredictionRequest, PredictionResult
from Application.services.predictor import analyze_prediction

# Create router for prediction endpoints
router = APIRouter()
#
@router.post("/predict", response_model=PredictionResult)
def predict(request: PredictionRequest):
    """
    Predict endpoint that takes sensor data with optional history,
    processes it through the ML logic, and returns the result.

    This route is used when sensor + history is passed explicitly.
    """
    return analyze_prediction(request)
