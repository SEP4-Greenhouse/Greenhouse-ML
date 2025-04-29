from datetime import datetime
from app.schemas.predict import PredictionRequest, PredictionResult

def analyze_prediction(request: PredictionRequest) -> PredictionResult:
    data = request.current
    history = request.history

    if data.sensorType.lower() == "temperature" and data.value > 30:
        status = "warning"
        suggestion = "Activate cooling"
    elif data.sensorType.lower() == "humidity" and data.value < 30:
        status = "warning"
        suggestion = "Start irrigation"
    else:
        status = "normal"
        suggestion = "No action needed"

    trend_analysis = None
    if history:
        values = [entry.value for entry in history]
        values.append(data.value)
        if len(values) >= 2:
            change = values[-1] - values[0]
            hours = (data.timestamp - history[0].timestamp).total_seconds() / 3600
            if hours != 0:
                rate = change / hours
                if rate < -2:
                    trend_analysis = f"Humidity dropping {abs(rate):.2f}% per hour"
                elif rate > 2:
                    trend_analysis = f"Humidity rising {abs(rate):.2f}% per hour"
                else:
                    trend_analysis = "Humidity stable"

    return PredictionResult(
        timestamp=datetime.utcnow(),
        status=status,
        suggestion=suggestion,
        trendAnalysis=trend_analysis
    )
