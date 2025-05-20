from fastapi import APIRouter
from Application.schema.predict import PredictionRequest, PredictionResult
from Application.services.predictor import analyze_prediction

router = APIRouter()

@router.post("/predict", response_model=PredictionResult)
async def predict(request: PredictionRequest):
    """
    ML prediction endpoint: receives sensor data, processes it,
    and returns prediction. Fetches history if not provided.
    """
    return await analyze_prediction(request)
