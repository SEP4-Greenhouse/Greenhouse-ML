import pytest
from datetime import datetime, timezone
from Application.schema.predict import SensorData, PredictionRequest, SensorHistoryEntry
from Application.services.predictor import analyze_prediction

@pytest.mark.asyncio
async def test_analyze_prediction_simple():
    current = SensorData(
        sensorType="Temperature",
        value=22.5,
        timestamp=datetime.now(timezone.utc)
    )

    history = [
        SensorHistoryEntry(value=21.0, timestamp=datetime.now(timezone.utc)),
        SensorHistoryEntry(value=20.5, timestamp=datetime.now(timezone.utc))
    ]

    request = PredictionRequest(current=current, history=history)
    result = await analyze_prediction(request)

    assert result.status in ["normal", "warning"]
    assert result.suggestion in ["Start irrigation", "No action needed", "Unable to predict"]
    assert result.timestamp is not None
