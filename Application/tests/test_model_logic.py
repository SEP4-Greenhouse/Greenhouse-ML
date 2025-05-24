import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from datetime import datetime, timezone
from Application.Dtos.predict import SensorReadingDto, PredictionRequestDto
from Application.services.ml_model_services import analyze_prediction

@pytest.mark.asyncio
async def test_analyze_prediction_logic():
    """Test that the ML prediction logic produces valid results."""
    payload = PredictionRequestDto(
        timestamp=datetime.now(timezone.utc),
        plantGrowthStage="Vegetative Stage",
        timeSinceLastWateringInHours=5.0,
        mlSensorReadings=[
            SensorReadingDto(SensorName="Soil Humidity", Unit="%", Value=30),
            SensorReadingDto(SensorName="Temperature", Unit="°C", Value=25),
            SensorReadingDto(SensorName="Air Humidity", Unit="%", Value=55),
            SensorReadingDto(SensorName="Light", Unit="lux", Value=300)
        ]
    )

    result = await analyze_prediction(payload)

    assert result is not None
    assert hasattr(result, "HoursUntilNextWatering")
    assert hasattr(result, "PredictionTime")
    assert isinstance(result.HoursUntilNextWatering, float)
    assert result.HoursUntilNextWatering >= 0
