from fastapi import APIRouter
from app.schemas.predict import PredictionRequest, PredictionResult
from app.services.predictor import analyze_prediction

router = APIRouter()

@router.post("/predict", response_model=PredictionResult)
def predict(request: PredictionRequest):
    return analyze_prediction(request)
