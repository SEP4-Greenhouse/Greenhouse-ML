from datetime import datetime
from Application.schema.predict import PredictionRequest, PredictionResult
from Application.services.ml_model import predict_hours_until_watering

async def analyze_prediction(request: PredictionRequest) -> PredictionResult:
    current = request.current

    predicted_hours = predict_hours_until_watering(current, history=[])

    return PredictionResult(
        timestamp=datetime.utcnow(),
        predictedHoursUntilWatering=round(predicted_hours, 2)
    )
