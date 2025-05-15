# tests/test_predictor.py

from datetime import datetime, timedelta
from Application.schema.predict import PredictionRequest, SensorData, SensorHistoryEntry
from Application.services.predictor import analyze_prediction


def test_humidity_warning():
    """
    Should return a warning when humidity is below threshold.
    """
    data = SensorData(sensorType="humidity", value=25.0, timestamp=datetime.utcnow())
    request = PredictionRequest(current=data)
    result = analyze_prediction(request)

    assert result.status == "warning"
    assert "irrigation" in result.suggestion.lower()


def test_temperature_warning():
    """
    Should return a warning when temperature is above 30°C.
    """
    data = SensorData(sensorType="temperature", value=35.0, timestamp=datetime.utcnow())
    request = PredictionRequest(current=data)
    result = analyze_prediction(request)

    assert result.status == "warning"
    assert "cooling" in result.suggestion.lower()


def test_normal_condition():
    """
    Should return normal when humidity is above threshold and temperature is not too high.
    """
    data = SensorData(sensorType="humidity", value=45.0, timestamp=datetime.utcnow())
    request = PredictionRequest(current=data)
    result = analyze_prediction(request)

    assert result.status == "normal"
    assert "no action" in result.suggestion.lower()


def test_trend_analysis_drop():
    """
    Should detect a decreasing humidity trend based on historical values.
    """
    now = datetime.utcnow()
    history = [
        SensorHistoryEntry(value=50.0, timestamp=now - timedelta(hours=2)),
        SensorHistoryEntry(value=40.0, timestamp=now - timedelta(hours=1))
    ]
    current = SensorData(sensorType="humidity", value=30.0, timestamp=now)
    request = PredictionRequest(current=current, history=history)
    result = analyze_prediction(request)

    assert result.trendAnalysis is not None
    assert "dropping" in result.trendAnalysis.lower()
