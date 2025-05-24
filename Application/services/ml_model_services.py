import os
import glob
import traceback
import joblib
import pickle
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
        model_version = os.path.basename(model_path)  # Store version but don't return it
        print(f"[ML_MODEL] Using model: {model_path}")

        try:
            # Try to load the model with normal joblib first
            model = joblib.load(model_path)
        except ModuleNotFoundError as e:
            print(f"[ML_MODEL] Prediction error: {str(e)}")
            print(f"[ML_MODEL] Error details: {traceback.format_exc()}")

            # Try alternate loading methods if it's a numpy._core issue
            if "numpy._core" in str(e):
                try:
                    print(f"[ML_MODEL] Trying alternate model loading with pickle")
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f, encoding='latin1')
                    print(f"[ML_MODEL] Successfully loaded with alternate method")
                    # Continue to prediction with the loaded model
                    features = extract_features_from_payload(payload)
                    prediction = model.predict([features])[0]
                    print(f"[ML_API] Successful prediction: {prediction:.2f} hours using pickle_compatible_{model_version}")
                    return PredictionResultDto(
                        PredictionTime=datetime.utcnow(),
                        HoursUntilNextWatering=float(prediction)
                    )
                except Exception as pickle_error:
                    print(f"[ML_MODEL] Alternate loading also failed: {str(pickle_error)}")
                    print(f"[ML_MODEL] Error details: {traceback.format_exc()}")
                    # Fall back to rule-based prediction
                    return create_fallback_model_prediction(payload, model_path)
            else:
                # For other module errors, use fallback
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
            print(f"[ML_API] Successful prediction: {prediction:.2f} hours using model {model_version}")
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
    """Extract and compute features to match the 16 features expected by the model."""
    sensor_dict = {reading.SensorName: reading.Value for reading in payload.mlSensorReadings}
    
    # Extract base features from payload
    temperature = sensor_dict.get("Temperature", 25.0)
    soil_humidity = sensor_dict.get("Soil Humidity", 40.0)
    air_humidity = sensor_dict.get("Air Humidity", 50.0)
    light = sensor_dict.get("Light", 200.0)
    co2 = sensor_dict.get("CO2", 400.0)
    pir = sensor_dict.get("PIR", 0.0)
    proximity = sensor_dict.get("Proximity", 0.0)
    time_since_watering = payload.timeSinceLastWateringInHours
    
    # Map growth stage to numeric values
    growth_stage_map = {
        "Seedling": 0,
        "Seedling Stage": 0,
        "Vegetative": 1,
        "Vegetative Stage": 1,
        "Flowering": 2,
        "Flowering Stage": 2
    }
    growth_stage = growth_stage_map.get(payload.plantGrowthStage, 1)
    
    # Create engineered features to match what was likely used in training
    # Feature interactions
    temp_soil = temperature * soil_humidity / 100.0
    temp_air = temperature * air_humidity / 100.0
    light_temp = light * temperature / 1000.0
    soil_air = soil_humidity * air_humidity / 100.0
    time_soil = time_since_watering * soil_humidity / 100.0
    
    # Squared features
    temp_squared = temperature * temperature / 100.0
    soil_squared = soil_humidity * soil_humidity / 100.0
    
    # Create the full feature list (16 features)
    features = [
        temperature,           # 1. Temperature
        soil_humidity,         # 2. Soil Humidity
        air_humidity,          # 3. Air Humidity
        light,                 # 4. Light
        co2,                   # 5. CO2
        pir,                   # 6. PIR
        proximity,             # 7. Proximity
        time_since_watering,   # 8. Hours since last watering
        growth_stage,          # 9. Plant growth stage
        temp_soil,             # 10. Temperature × Soil Humidity interaction
        temp_air,              # 11. Temperature × Air Humidity interaction  
        light_temp,            # 12. Light × Temperature interaction
        soil_air,              # 13. Soil Humidity × Air Humidity interaction
        time_soil,             # 14. Time × Soil Humidity interaction
        temp_squared,          # 15. Temperature squared
        soil_squared           # 16. Soil Humidity squared
    ]
    
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
    
    model_name = os.path.basename(model_path)
    print(f"[ML_API] Fallback prediction: {hours:.2f} hours (fallback_{model_name})")
    
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
    
    print(f"[ML_API] Fallback prediction: {hours:.2f} hours (fallback_{reason})")
    
    return PredictionResultDto(
        PredictionTime=datetime.utcnow(),
        HoursUntilNextWatering=float(hours)
    )