import os
import glob
import traceback
import joblib
from datetime import datetime
from Application.Dtos.predict import PredictionRequestDto, PredictionResultDto

# Constants
MODEL_DIR = os.environ.get("MODEL_DIR", "Application/trained_models")

async def analyze_prediction(payload: PredictionRequestDto) -> PredictionResultDto:
    """Analyze sensor data and predict hours until watering is needed."""
    try:
        # Find the most recent model file
        model_files = glob.glob(os.path.join(MODEL_DIR, "reg_model_*.pkl"))
        if not model_files:
            print(f"[ML_MODEL] No model files found in {MODEL_DIR}")
            return create_fallback_prediction(payload, "no_model_found")

        model_path = max(model_files, key=os.path.getctime)
        print(f"[ML_MODEL] Using model: {model_path}")

        try:
            # Try to load the model
            model = joblib.load(model_path)
        except ModuleNotFoundError as e:
            print(f"[ML_MODEL] Prediction error: {str(e)}")
            print(f"[ML_MODEL] Error details: {traceback.format_exc()}")

            # Special handling for numpy._core issue - create a simple model on the fly
            return create_fallback_model_prediction(payload, model_path)
        except Exception as e:
            print(f"[ML_MODEL] Error loading model: {str(e)}")
            print(f"[ML_MODEL] Error details: {traceback.format_exc()}")
            return create_fallback_prediction(payload, f"model_load_error_{type(e).__name__}")

        # Extract features for prediction
        features = extract_features_from_payload(payload)

        # Make prediction
        try:
            prediction = model.predict([features])[0]
            return PredictionResultDto(
                PredictionTime=datetime.utcnow(),
                HoursUntilNextWatering=float(prediction)
            )
        except Exception as e:
            print(f"[ML_MODEL] Prediction failed: {str(e)}")
            print(f"[ML_MODEL] Error details: {traceback.format_exc()}")
            return create_fallback_prediction(payload, f"prediction_error_{type(e).__name__}")

    except Exception as e:
        print(f"[ML_MODEL] Unexpected error: {str(e)}")
        print(f"[ML_MODEL] Error details: {traceback.format_exc()}")
        return create_fallback_prediction(payload, f"unexpected_error_{type(e).__name__}")

def extract_features_from_payload(payload: PredictionRequestDto) -> list:
    sensor_dict = {reading.SensorName: reading.Value for reading in payload.mlSensorReadings}
    features = [
        sensor_dict.get("Temperature", 25.0),
        sensor_dict.get("Soil Humidity", 40.0),
        sensor_dict.get("Air Humidity", 50.0),
        sensor_dict.get("Light", 200.0),
        payload.timeSinceLastWateringInHours,
    ]
    growth_stage_map = {
        "Seedling": 0,
        "Seedling Stage": 0,
        "Vegetative": 1,
        "Vegetative Stage": 1,
        "Flowering": 2,
        "Flowering Stage": 2
    }
    features.append(growth_stage_map.get(payload.plantGrowthStage, 1))
    return features

def create_fallback_model_prediction(payload: PredictionRequestDto, model_path: str) -> PredictionResultDto:
    print("[ML_MODEL] Creating simple fallback model")
    sensor_dict = {reading.SensorName: reading.Value for reading in payload.mlSensorReadings}
    soil_humidity = sensor_dict.get("Soil Humidity", 50)
    if soil_humidity < 20:
        hours = 4.0
    elif soil_humidity < 30:
        hours = 12.0
    elif soil_humidity < 40:
        hours = 24.0
    elif soil_humidity < 50:
        hours = 36.0
    else:
        hours = 48.0
    temperature = sensor_dict.get("Temperature", 25)
    if temperature > 30:
        hours *= 0.7
    elif temperature < 15:
        hours *= 1.3
    if payload.plantGrowthStage in ["Seedling", "Seedling Stage"]:
        hours *= 0.9
    elif payload.plantGrowthStage in ["Flowering", "Flowering Stage"]:
        hours *= 1.1
    return PredictionResultDto(
        PredictionTime=datetime.utcnow(),
        HoursUntilNextWatering=float(hours)
    )

def create_fallback_prediction(payload: PredictionRequestDto, reason: str) -> PredictionResultDto:
    print(f"[ML_MODEL] Using fallback prediction due to: {reason}")
    sensor_dict = {reading.SensorName: reading.Value for reading in payload.mlSensorReadings}
    soil_humidity = sensor_dict.get("Soil Humidity", 50)
    if soil_humidity < 20:
        hours = 4.0
    elif soil_humidity < 30:
        hours = 12.0
    elif soil_humidity < 40:
        hours = 24.0
    elif soil_humidity < 60:
        hours = 36.0
    else:
        hours = 48.0
    if reason == "no_model_found":
        print("[ML_MODEL] No model available, using extended fallback logic")
        temperature = sensor_dict.get("Temperature", 25)
        if temperature > 30:
            hours *= 0.8
        elif temperature < 15:
            hours *= 1.2
        air_humidity = sensor_dict.get("Air Humidity", 50)
        if air_humidity < 30:
            hours *= 0.9
        elif air_humidity > 70:
            hours *= 1.1
        light = sensor_dict.get("Light", 200)
        if light > 800:
            hours *= 0.9
        elif light < 100:
            hours *= 1.1
        if payload.plantGrowthStage in ["Seedling", "Seedling Stage"]:
            hours *= 0.8
        elif payload.plantGrowthStage in ["Vegetative", "Vegetative Stage"]:
            hours *= 1.0
        elif payload.plantGrowthStage in ["Flowering", "Flowering Stage"]:
            hours *= 1.1
        if payload.timeSinceLastWateringInHours > 48:
            hours *= 0.8
    hours = max(min(hours, 72.0), 1.0)
    return PredictionResultDto(
        PredictionTime=datetime.utcnow(),
        HoursUntilNextWatering=float(hours)
    )