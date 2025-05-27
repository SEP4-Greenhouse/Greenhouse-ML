import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pytest
import time
from unittest import mock
import numpy as np
from datetime import datetime, timezone

from Application.Dtos.predict import PredictionRequestDto, SensorReadingDto
from Application.services.ml_model_services import analyze_prediction

@pytest.fixture
def standard_payload():
    """Standard test payload for predictions"""
    return PredictionRequestDto(
        timestamp=datetime.now(timezone.utc),
        plantGrowthStage="Vegetative Stage",
        timeSinceLastWateringInHours=5.0,
        mlSensorReadings=[
            SensorReadingDto(SensorName="Temperature", Unit="°C", Value=25.0),
            SensorReadingDto(SensorName="Soil Humidity", Unit="%", Value=40.0),
            SensorReadingDto(SensorName="Air Humidity", Unit="%", Value=55.0),
            SensorReadingDto(SensorName="Light", Unit="lux", Value=200.0)
        ]
    )

@pytest.mark.asyncio
async def test_basic_prediction(standard_payload):
    """Test basic prediction functionality"""
    with mock.patch('joblib.load') as mock_load:
        mock_model = mock.MagicMock()
        mock_model.predict.return_value = np.array([8.5])
        mock_load.return_value = mock_model
        
        result = await analyze_prediction(standard_payload)
        
        assert result.HoursUntilNextWatering == 8.5

@pytest.mark.asyncio
async def test_model_fallback(standard_payload):
    """Test fallback when model loading fails"""
    with mock.patch('joblib.load', side_effect=Exception("Failed to load")):
        with mock.patch('pickle.load', side_effect=Exception("Failed to load with pickle")):
            result = await analyze_prediction(standard_payload)
            
            # Should still get a fallback prediction
            assert result is not None
            assert result.HoursUntilNextWatering > 0

@pytest.mark.asyncio
async def test_extreme_values():
    """Test with extreme input values"""
    payload = PredictionRequestDto(
        timestamp=datetime.now(timezone.utc),
        plantGrowthStage="Vegetative Stage",
        timeSinceLastWateringInHours=100.0,  # Very long time
        mlSensorReadings=[
            SensorReadingDto(SensorName="Temperature", Unit="°C", Value=45.0),  # Very hot
            SensorReadingDto(SensorName="Soil Humidity", Unit="%", Value=5.0),  # Very dry
        ]
    )
    
    with mock.patch('joblib.load') as mock_load:
        mock_model = mock.MagicMock()
        mock_model.predict.return_value = np.array([2.0])
        mock_load.return_value = mock_model
        
        result = await analyze_prediction(payload)
        
        assert 0 <= result.HoursUntilNextWatering <= 24  # Reasonable range
@pytest.mark.asyncio
@pytest.mark.performance
async def test_prediction_performance(standard_payload):
    """Test prediction performance"""
    with mock.patch('joblib.load') as mock_load:
        mock_model = mock.MagicMock()
        mock_model.predict.return_value = np.array([8.0])
        mock_load.return_value = mock_model
        
        start_time = time.time()
        await analyze_prediction(standard_payload)
        elapsed = time.time() - start_time
        
        # Prediction should be fast
        assert elapsed < 0.5  # 500ms is generous for a mocked test