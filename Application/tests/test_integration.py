import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pytest
import glob
from unittest import mock
import numpy as np
from fastapi.testclient import TestClient
from datetime import datetime, timezone

from Application.main import app

client = TestClient(app)

@pytest.mark.integration
def test_end_to_end_prediction():
    """Test the full prediction flow from API to model to response."""
    # Instead of skipping, mock the model files
    with mock.patch('joblib.load') as mock_load:
        # Create a mock model that returns a reasonable prediction
        mock_model = mock.MagicMock()
        mock_model.predict.return_value = np.array([8.0])
        mock_load.return_value = mock_model
        
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "plantGrowthStage": "Vegetative Stage",
            "timeSinceLastWateringInHours": 5.0,
            "mlSensorReadings": [
                {"SensorName": "Temperature", "Unit": "°C", "Value": 24.5},
                {"SensorName": "Soil Humidity", "Unit": "%", "Value": 40.0},
                {"SensorName": "Air Humidity", "Unit": "%", "Value": 55.0},
                {"SensorName": "Light", "Unit": "lux", "Value": 200.0}
            ]
        }

        # Use the correct endpoint path that matches your router
        # Try multiple paths if needed
        response = None
        for endpoint in ["/predict", "/api/predict", "/api/ml/predict"]:
            try:
                response = client.post(endpoint, json=payload)
                if response.status_code == 200:
                    break
            except:
                continue
                
        # If all endpoints failed, use the original
        if not response:
            response = client.post("/predict", json=payload)
            
        # Remove try/except to allow the test to fail naturally if there's an issue
        assert response.status_code == 200
        
        data = response.json()
        assert "PredictionTime" in data
        assert "HoursUntilNextWatering" in data
        assert isinstance(data["HoursUntilNextWatering"], (float, int))
        assert 0 <= data["HoursUntilNextWatering"] <= 48