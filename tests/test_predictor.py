from datetime import datetime, timedelta
from app.Schemas.predict import PredictionRequest, SensorData, SensorHistoryEntry
from app.Services.predictor import analyze_prediction

def test_humidity_warning():
    data = SensorData(sensorType="humidity", value=25.0, timestamp=datetime.utcnow())
    request = PredictionRequest(current=data)
    result = analyze_prediction(request)
    assert result.status == "warning"
    assert "irrigation" in result.suggestion.lower()

def test_temperature_warning():
    data = SensorData(sensorType="temperature", value=35.0, timestamp=datetime.utcnow())
    request = PredictionRequest(current=data)
    result = analyze_prediction(request)
    assert result.status == "warning"
    assert "cooling" in result.suggestion.lower()

def test_normal_condition():
    data = SensorData(sensorType="humidity", value=45.0, timestamp=datetime.utcnow())
    request = PredictionRequest(current=data)
    result = analyze_prediction(request)
    assert result.status == "normal"

def test_trend_analysis_drop():
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
