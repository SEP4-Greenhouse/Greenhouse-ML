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
            # Just basic prediction with the loaded model
            prediction = model.predict([features])[0]
            
            # Return a successful prediction
            model_version = os.path.basename(model_path).replace("reg_model_", "").replace(".pkl", "")
            return PredictionResultDto(
                timestamp=datetime.now(),
                predictedHoursUntilWatering=float(prediction),
                modelVersion=model_version
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
    """Extract features from the request payload."""
    # Map sensor readings to a dictionary for easier access
    sensor_dict = {reading.SensorName: reading.Value for reading in payload.mlSensorReadings}
    
    # Extract features in the order expected by the model
    features = [
        sensor_dict.get("Temperature", 25.0),  # Default to reasonable values if missing
        sensor_dict.get("Soil Humidity", 40.0),
        sensor_dict.get("Air Humidity", 50.0),
        sensor_dict.get("Light", 200.0),
        payload.timeSinceLastWateringInHours,
    ]
    
    # For plant growth stage, we'd normally one-hot encode
    # But for fallback, we'll use a numerical representation
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
    """Create a simple decision tree model as fallback when main model fails."""
    print("[ML_MODEL] Creating simple fallback model")
    
    # Map sensor readings to a dictionary for easier access
    sensor_dict = {reading.SensorName: reading.Value for reading in payload.mlSensorReadings}
    
    # Extract soil humidity - main factor for watering
    soil_humidity = sensor_dict.get("Soil Humidity", 50)
    
    # Simple decision tree logic
    if soil_humidity < 20:
        hours = 4.0  # Very dry - water soon
    elif soil_humidity < 30:
        hours = 12.0
    elif soil_humidity < 40:
        hours = 24.0
    elif soil_humidity < 50:
        hours = 36.0
    else:
        hours = 48.0
    
    # Adjust based on temperature
    temperature = sensor_dict.get("Temperature", 25)
    if temperature > 30:
        hours *= 0.7  # Higher temperature - water more frequently
    elif temperature < 15:
        hours *= 1.3  # Lower temperature - water less frequently
    
    # Adjust based on growth stage
    if payload.plantGrowthStage == "Seedling Stage" or payload.plantGrowthStage == "Seedling":
        hours *= 0.9  # Seedlings need more frequent watering
    elif payload.plantGrowthStage == "Flowering Stage" or payload.plantGrowthStage == "Flowering":
        hours *= 1.1  # Flowering plants may need slightly less frequent watering
    
    model_version = os.path.basename(model_path).replace("reg_model_", "").replace(".pkl", "")
    return PredictionResultDto(
        timestamp=datetime.now(),
        predictedHoursUntilWatering=float(hours),
        modelVersion=f"fallback_{model_version}"
    )

def create_fallback_prediction(payload: PredictionRequestDto, reason: str) -> PredictionResultDto:
    """Create a fallback prediction when the model cannot be used."""
    print(f"[ML_MODEL] Using fallback prediction due to: {reason}")
    
    # Map sensor readings to a dictionary for easier access
    sensor_dict = {reading.SensorName: reading.Value for reading in payload.mlSensorReadings}
    
    # Get soil humidity (most important factor)
    soil_humidity = sensor_dict.get("Soil Humidity", 50)
    
    # Set baseline hours based on soil humidity
    if soil_humidity < 20:
        hours = 4.0  # Very dry - water soon
    elif soil_humidity < 30:
        hours = 12.0
    elif soil_humidity < 40:
        hours = 24.0
    elif soil_humidity < 60:
        hours = 36.0
    else:
        hours = 48.0
    
    # If this is specifically a "no model found" case, add more adjustments
    if reason == "no_model_found":
        print("[ML_MODEL] No model available, using extended fallback logic")
        
        # Consider temperature
        temperature = sensor_dict.get("Temperature", 25)
        if temperature > 30:
            hours *= 0.8  # Hot - water more frequently
        elif temperature < 15:
            hours *= 1.2  # Cold - water less frequently
            
        # Consider air humidity
        air_humidity = sensor_dict.get("Air Humidity", 50)
        if air_humidity < 30:
            hours *= 0.9  # Dry air - plants lose more water
        elif air_humidity > 70:
            hours *= 1.1  # Humid air - plants lose less water
            
        # Consider light intensity
        light = sensor_dict.get("Light", 200)
        if light > 800:
            hours *= 0.9  # Bright conditions - more evaporation
        elif light < 100:
            hours *= 1.1  # Dim conditions - less evaporation
            
        # Consider growth stage
        if payload.plantGrowthStage == "Seedling" or payload.plantGrowthStage == "Seedling Stage":
            hours *= 0.8  # Seedlings need more frequent watering
        elif payload.plantGrowthStage == "Vegetative" or payload.plantGrowthStage == "Vegetative Stage":
            hours *= 1.0  # Normal watering frequency
        elif payload.plantGrowthStage == "Flowering" or payload.plantGrowthStage == "Flowering Stage":
            hours *= 1.1  # Flowering plants need slightly less frequent watering
            
        # Consider time since last watering
        if payload.timeSinceLastWateringInHours > 48:
            hours *= 0.8  # Long time since last watering - prioritize watering
    
    # Ensure hours is within reasonable bounds
    hours = max(min(hours, 72.0), 1.0)
        
    return PredictionResultDto(
        timestamp=datetime.now(),
        predictedHoursUntilWatering=float(hours),
        modelVersion=f"fallback_{reason}"
    )