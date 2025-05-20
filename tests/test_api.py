from fastapi.testclient import TestClient
from Application.main import app
from datetime import datetime, timezone


client = TestClient(app)

def test_api_ml_predict():
    payload = {
        "current": {
            "sensorType": "Temperature",
            "value": 23.5,
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        "history": [
            {"value": 22.0, "timestamp": datetime.now(timezone.utc).isoformat()},
            {"value": 21.0, "timestamp": datetime.now(timezone.utc).isoformat()}
        ]
    }

    response = client.post("/api/ml/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "suggestion" in data
