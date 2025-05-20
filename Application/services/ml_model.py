import joblib
import os
import numpy as np
from Application.schema.predict import SensorHistoryEntry

MODEL_PATH = os.getenv("MODEL_PATH", "Application/ml_model/model.pkl")
model = joblib.load(MODEL_PATH)

# Mapping numeric prediction to label
label_map = {
    1: "Start irrigation",
    0: "No action needed"
}

def predict_action(current_value: float, history: list[SensorHistoryEntry]) -> str:
    try:
        avg = np.mean([entry.value for entry in history]) if history else current_value
        features = np.array([[current_value, avg]])
        prediction = model.predict(features)[0]
        return label_map.get(prediction, "Unknown")
    except Exception as e:
        print(f"[ML_MODEL] Prediction error: {e}")
        return "Unable to predict"
