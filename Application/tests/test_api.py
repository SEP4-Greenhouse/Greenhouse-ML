import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


from fastapi.testclient import TestClient
from Application.main import app
from datetime import datetime, timezone

client = TestClient(app)


def test_api_ml_predict():
    """Test the ML prediction API endpoint."""
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "plantGrowthStage": "Vegetative Stage",
        "timeSinceLastWateringInHours": 6.0,
        "mlSensorReadings": [
            {"SensorName": "Temperature", "Unit": "°C", "Value": 24.5},
            {"SensorName": "Soil Humidity", "Unit": "%", "Value": 40.0},
            {"SensorName": "Air Humidity", "Unit": "%", "Value": 55.0},
            {"SensorName": "Light", "Unit": "lux", "Value": 200.0}
        ]
    }

    response = client.post("/api/ml/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "timestamp" in data
    assert "predictedHoursUntilWatering" in data
    assert isinstance(data["predictedHoursUntilWatering"], float)
    assert data["predictedHoursUntilWatering"] > 0
