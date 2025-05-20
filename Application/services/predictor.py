from Application.schema.predict import PredictionRequest, PredictionResult
from Application.services.backend_client import fetch_history, log_prediction
from Application.services.ml_model import predict_action
from datetime import datetime, timezone

def calculate_trend(history, current_value):
    if not history:
        return None
    avg = sum(h.value for h in history) / len(history)
    return "Decreasing" if current_value < avg else "Stable or Increasing"

async def analyze_prediction(request: PredictionRequest) -> PredictionResult:
    current = request.current
    history = request.history

    if history is None:
        history = await fetch_history(current.sensorType, limit=20)

    # 🔁 Uses real ML model
    suggestion = predict_action(current.value, history)

    # Derive status based on suggestion
    status = "warning" if "irrigation" in suggestion.lower() else "normal"

    trend = calculate_trend(history, current.value)

    result = PredictionResult(
        timestamp=datetime.now(timezone.utc),
        status=status,
        suggestion=suggestion,
        trendAnalysis=trend
    )

    await log_prediction(current, result)
    return result
