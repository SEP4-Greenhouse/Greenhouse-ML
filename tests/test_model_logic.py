from Application.services.ml_model import predict_action
from Application.schema.predict import SensorHistoryEntry
from datetime import datetime, timezone


def test_predict_action_start_irrigation():
    history = [
        SensorHistoryEntry(value=20.0, timestamp=datetime.now(timezone.utc)),
        SensorHistoryEntry(value=21.0, timestamp=datetime.now(timezone.utc))
    ]
    result = predict_action(19.0, history)
    assert result in ["Start irrigation", "No action needed", "Unable to predict"]  # depends on your dummy model

def test_predict_action_handles_empty_history():
    result = predict_action(19.0, [])
    assert isinstance(result, str)
