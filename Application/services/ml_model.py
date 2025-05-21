import joblib
import os
import glob
import numpy as np
from Application.schema.predict import SensorData

# === Resolve latest model file ===
MODEL_DIR = "Application/trained_models"

def get_latest_model(pattern: str) -> str:
    files = sorted(glob.glob(os.path.join(MODEL_DIR, pattern)), key=os.path.getmtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No model file matching pattern '{pattern}' found in '{MODEL_DIR}'")
    return files[0]

# Load the most recent model
MODEL_PATH = get_latest_model("tuned_model_*.pkl")
model = joblib.load(MODEL_PATH)

# === Prediction logic ===
def predict_hours_until_watering(current: SensorData) -> float:

    try:
        avg_soil_moisture = (
            sum(h.value for h in history) / len(history)
            if history else current.soil_moisture
        )

        is_daytime = 1 if 6 <= current.hour_of_day <= 18 else 0

        features = np.array([[ 
            current.soil_moisture,
            current.temperature,
            current.humidity,
            current.light,
            current.hour_of_day,
            is_daytime,
            0  # Placeholder for moisture_drop_rate
        ]])

        prediction = model.predict(features)[0]
        return float(prediction)

    except Exception as e:
        print(f"[ML_MODEL] Prediction error: {e}")
        return -1.0
