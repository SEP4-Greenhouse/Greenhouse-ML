import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pytest
from datetime import datetime, timezone
from Application.Dtos.predict import SensorReadingDto, PredictionRequestDto
from Application.services.ml_model_services import analyze_prediction

@pytest.mark.asyncio
async def test_analyze_prediction_simple():
    """Test the prediction service with simple input values - integration test."""
    # Skip if running in CI environment without model files
    if os.environ.get('CI'):
        pytest.skip("Skipping integration test in CI environment")
    
    payload = PredictionRequestDto(
        timestamp=datetime.now(timezone.utc),
        plantGrowthStage="Vegetative Stage",
        timeSinceLastWateringInHours=12.0,
        mlSensorReadings=[
            SensorReadingDto(SensorName="Temperature", Unit="°C", Value=22.0),
            SensorReadingDto(SensorName="Soil Humidity", Unit="%", Value=38.0),
            SensorReadingDto(SensorName="Air Humidity", Unit="%", Value=60.0),
            SensorReadingDto(SensorName="Light", Unit="lux", Value=250.0)
        ]
    )
    
    # This test uses the actual model, so it's an integration test
    # It will be skipped if model files aren't available
    try:
        result = await analyze_prediction(payload)
        
        # We can't assert exact values since it depends on the model
        # but we can check the result structure and reasonable ranges
        assert result is not None
        assert hasattr(result, "PredictionTime")
        assert hasattr(result, "HoursUntilNextWatering")
        assert result.HoursUntilNextWatering > 0
        assert result.HoursUntilNextWatering < 72  # Reasonable upper bound
    except FileNotFoundError:
        pytest.skip("No model file found, skipping integration test")