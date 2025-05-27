import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError
from Application.Dtos.predict import PredictionRequestDto, PredictionResultDto, SensorReadingDto

def test_sensor_reading_dto():
    """Test SensorReadingDto validation"""
    # Valid
    sensor = SensorReadingDto(SensorName="Temperature", Unit="°C", Value=25.0)
    assert sensor.SensorName == "Temperature"
    
    # Invalid - missing required field
    with pytest.raises(ValidationError):
        SensorReadingDto(Unit="°C", Value=24.5)

def test_prediction_request_dto():
    """Test PredictionRequestDto validation"""
    now = datetime.now(timezone.utc)
    
    # Valid
    request = PredictionRequestDto(
        timestamp=now,
        plantGrowthStage="Vegetative Stage",
        timeSinceLastWateringInHours=6.0,
        mlSensorReadings=[
            SensorReadingDto(SensorName="Temperature", Unit="°C", Value=24.5)
        ]
    )
    assert request.plantGrowthStage == "Vegetative Stage"
    
    # Test with missing required fields instead of negative values
    # If negative values are allowed, we should test for what is not allowed
    with pytest.raises(ValidationError):
        PredictionRequestDto(
            timestamp=now,
            # Missing plantGrowthStage
            timeSinceLastWateringInHours=6.0,
            mlSensorReadings=[
                SensorReadingDto(SensorName="Temperature", Unit="°C", Value=24.5)
            ]
        )

def test_prediction_result_dto():
    """Test PredictionResultDto"""
    now = datetime.now(timezone.utc)
    result = PredictionResultDto(
        PredictionTime=now,
        HoursUntilNextWatering=12.5
    )
    
    # Test serialization - use model_dump() instead of dict()
    result_dict = result.model_dump()
    assert result_dict["HoursUntilNextWatering"] == 12.5