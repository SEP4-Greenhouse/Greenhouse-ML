import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from Application.services.ml_model_services import extract_features_from_payload
from Application.Dtos.predict import SensorReadingDto, PredictionRequestDto
from datetime import datetime, timezone

def test_analyze_prediction_logic():
    """Test the feature extraction logic."""
    payload = PredictionRequestDto(
        timestamp=datetime.now(timezone.utc),
        plantGrowthStage="Vegetative Stage",
        timeSinceLastWateringInHours=5.0,
        mlSensorReadings=[
            SensorReadingDto(SensorName="Temperature", Unit="°C", Value=22.5),
            SensorReadingDto(SensorName="Soil Humidity", Unit="%", Value=40),
            SensorReadingDto(SensorName="Air Humidity", Unit="%", Value=55),
            SensorReadingDto(SensorName="Light", Unit="lux", Value=200)
        ]
    )

    features = extract_features_from_payload(payload)
    assert len(features) == 16  # Verify we have the expected number of features
    assert features[0] == 22.5  # Temperature
    assert features[1] == 40.0  # Soil Humidity
    assert features[2] == 55.0  # Air Humidity
    assert features[8] == 1     # Growth stage should be 1 for Vegetative Stage