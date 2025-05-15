from datetime import datetime
from Application.services.predictor import analyze_prediction

from Application.schema.predict import PredictionRequest, PredictionResult
import os

# Set the humidity threshold for triggering irrigation (default: 30%)
WATER_THRESHOLD = float(os.getenv("WATER_THRESHOLD", 30.0))

def analyze_prediction(request: PredictionRequest) -> PredictionResult:
    """
    Analyze incoming sensor data and return a structured prediction result.

    - For temperature: Trigger cooling if value > 30°C.
    - For humidity: Trigger irrigation if value < threshold (default 30%).
    - Optional: If history is included, compute rate of change over time to
      provide trend analysis.
    """

    data = request.current       # Current sensor reading
    history = request.history    # Optional list of past readings

    # Rule-based decision logic
    if data.sensorType.lower() == "temperature" and data.value > 30:
        status = "warning"
        suggestion = "Activate cooling"
    elif data.sensorType.lower() == "humidity" and data.value < WATER_THRESHOLD:
        status = "warning"
        suggestion = "Start irrigation"
    else:
        status = "normal"
        suggestion = "No action needed"

    # Analyze historical trend if available
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

    # Return prediction result object with status and optional trend
    return PredictionResult(
        timestamp=datetime.utcnow(),
        status=status,
        suggestion=suggestion,
        trendAnalysis=trend_analysis
    )
