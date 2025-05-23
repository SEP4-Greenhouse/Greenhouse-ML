import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from unittest import mock
from datetime import datetime, timezone
from Application.Dtos.predict import SensorReadingDto, PredictionRequestDto
from Application.services.ml_model_services import analyze_prediction


@pytest.mark.asyncio
async def test_temperature_warning():
    """Test prediction with high temperature values."""
    payload = PredictionRequestDto(
        timestamp=datetime.now(timezone.utc),
        plantGrowthStage="Vegetative Stage",  # Match exact value from training
        timeSinceLastWateringInHours=4.0,
        mlSensorReadings=[
            SensorReadingDto(SensorName="Temperature", Unit="°C", Value=36.0),
            SensorReadingDto(SensorName="Soil Humidity", Unit="%", Value=35),
            SensorReadingDto(SensorName="Air Humidity", Unit="%", Value=55),
            SensorReadingDto(SensorName="Light", Unit="lux", Value=180)
        ]
    )
    result = await analyze_prediction(payload)
    assert isinstance(result.predictedHoursUntilWatering, float)
    assert result.predictedHoursUntilWatering > 0
    assert hasattr(result, "modelVersion")


@pytest.mark.asyncio
async def test_humidity_warning():
    """Test prediction with low soil humidity values."""
    payload = PredictionRequestDto(
        timestamp=datetime.now(timezone.utc),
        plantGrowthStage="Vegetative Stage",  # Match exact value from training
        timeSinceLastWateringInHours=6.0,
        mlSensorReadings=[
            SensorReadingDto(SensorName="Soil Humidity", Unit="%", Value=25),
            SensorReadingDto(SensorName="Temperature", Unit="°C", Value=22),
            SensorReadingDto(SensorName="Air Humidity", Unit="%", Value=60),
            SensorReadingDto(SensorName="Light", Unit="lux", Value=200)
        ]
    )
    result = await analyze_prediction(payload)
    assert isinstance(result.predictedHoursUntilWatering, float)
    assert result.predictedHoursUntilWatering >= 0
    assert hasattr(result, "modelVersion")

@pytest.mark.asyncio
async def test_missing_sensor_data():
    """Test prediction with missing sensor readings."""
    payload = PredictionRequestDto(
        timestamp=datetime.now(timezone.utc),
        plantGrowthStage="Vegetative Stage",
        timeSinceLastWateringInHours=5.0,
        mlSensorReadings=[
            # Missing Temperature
            SensorReadingDto(SensorName="Soil Humidity", Unit="%", Value=40),
            SensorReadingDto(SensorName="Air Humidity", Unit="%", Value=55),
            SensorReadingDto(SensorName="Light", Unit="lux", Value=200)
        ]
    )
    result = await analyze_prediction(payload)
    assert isinstance(result.predictedHoursUntilWatering, float)
    assert result.predictedHoursUntilWatering >= 0


@pytest.mark.asyncio
async def test_different_growth_stages():
    """Test prediction with different plant growth stages."""
    stages = ["Seedling Stage", "Vegetative Stage", "Flowering Stage"]
    
    for stage in stages:
        payload = PredictionRequestDto(
            timestamp=datetime.now(timezone.utc),
            plantGrowthStage=stage,
            timeSinceLastWateringInHours=5.0,
            mlSensorReadings=[
                SensorReadingDto(SensorName="Temperature", Unit="°C", Value=25),
                SensorReadingDto(SensorName="Soil Humidity", Unit="%", Value=40),
                SensorReadingDto(SensorName="Air Humidity", Unit="%", Value=55),
                SensorReadingDto(SensorName="Light", Unit="lux", Value=200)
            ]
        )
        result = await analyze_prediction(payload)
        assert isinstance(result.predictedHoursUntilWatering, float)
        assert result.predictedHoursUntilWatering >= 0
        assert hasattr(result, "modelVersion")


@pytest.mark.asyncio
async def test_invalid_inputs():
    """Test prediction with invalid inputs."""
    # Test with extremely high values
    payload = PredictionRequestDto(
        timestamp=datetime.now(timezone.utc),
        plantGrowthStage="Vegetative Stage",
        timeSinceLastWateringInHours=1000.0,  # Unreasonably high
        mlSensorReadings=[
            SensorReadingDto(SensorName="Temperature", Unit="°C", Value=100),  # Unreasonably high
            SensorReadingDto(SensorName="Soil Humidity", Unit="%", Value=40),
            SensorReadingDto(SensorName="Air Humidity", Unit="%", Value=55),
            SensorReadingDto(SensorName="Light", Unit="lux", Value=200)
        ]
    )
    result = await analyze_prediction(payload)
    assert isinstance(result.predictedHoursUntilWatering, float)
    # Even with extreme inputs, we should get a value


@pytest.mark.asyncio
async def test_model_loading_error():
    """Test error handling when model can't be loaded."""
    # Mock the model loading to simulate error
    with mock.patch('joblib.load', side_effect=Exception("Simulated error")):
        payload = PredictionRequestDto(
            timestamp=datetime.now(timezone.utc),
            plantGrowthStage="Vegetative Stage",
            timeSinceLastWateringInHours=5.0,
            mlSensorReadings=[
                SensorReadingDto(SensorName="Temperature", Unit="°C", Value=25),
                SensorReadingDto(SensorName="Soil Humidity", Unit="%", Value=40),
                SensorReadingDto(SensorName="Air Humidity", Unit="%", Value=55),
                SensorReadingDto(SensorName="Light", Unit="lux", Value=200)
            ]
        )
        result = await analyze_prediction(payload)
        # Should use fallback model
        assert result.predictedHoursUntilWatering > 0  # Changed from < 0
        assert "fallback" in result.modelVersion.lower()  # Check that it used fallback