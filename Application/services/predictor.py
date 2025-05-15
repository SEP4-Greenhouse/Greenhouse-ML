# Application/services/predictor.py

from datetime import datetime
import os
from Application.schema.predict import PredictionRequest, PredictionResult

# Default humidity threshold for triggering irrigation
WATER_THRESHOLD = float(os.getenv("WATER_THRESHOLD", 30.0))  # Can be overridden via .env

def analyze_prediction(request: PredictionRequest) -> PredictionResult:
    """
    Analyze incoming sensor data and return a structured prediction result.

    Logic:
    - If sensorType is 'temperature' and value > 30°C: trigger cooling
    - If sensorType is 'humidity' and value < WATER_THRESHOLD: trigger irrigation
    - Otherwise: status is normal

    If historical data is included, calculate the rate of change
    and return a trend analysis.

    Args:
        request (PredictionRequest): Contains current sensor reading and optional history

    Returns:
        PredictionResult: Includes status, suggestion, and optional trendAnalysis
    """

    data = request.current
    history = request.history

    # Decision logic
    if data.sensorType.lower() == "temperature" and data.value > 30:
        status = "warning"
        suggestion = "Activate cooling"
    elif data.sensorType.lower() == "humidity" and data.value < WATER_THRESHOLD:
        status = "warning"
        suggestion = "Start irrigation"
    else:
        status = "normal"
        suggestion = "No action needed"

    # Optional: Trend analysis based on historical values
    trend_analysis = None
    if history:
        values = [entry.value for entry in history] + [data.value]
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
