from datetime import datetime
import numpy as np
import traceback

# Define constants for feature names - updated to match training data
CORE_SENSORS = ["Soil Humidity", "Temperature", "Air Humidity", "Light"]
ENGINEERED_FEATURES = [
    "soil_dryness_index",
    "temp_soil_product",
    "light_humidity_ratio",
    "hourOfDay",
    "is_daytime"
]
TRAINING_FEATURES = CORE_SENSORS + ENGINEERED_FEATURES  # Will add one-hot encoded features dynamically

def map_backend_readings_to_features(data: dict, encoder=None):
    """
    Maps API/backend data format to the format expected by the ML model.
    
    Args:
        data: Dictionary containing sensor readings and metadata
        encoder: Optional one-hot encoder for categorical variables
        
    Returns:
        tuple: (features_list, feature_names_list)
    """
    try:
        # Extract sensor readings from JSON format
        sensor_readings = {}
        if "mlSensorReadings" in data:
            for reading in data["mlSensorReadings"]:
                sensor_name = reading["SensorName"]
                sensor_value = float(reading["Value"])
                sensor_readings[sensor_name] = sensor_value
        
        # Extract timestamp and time features
        try:
            timestamp_str = data.get("timestamp", "")
            if isinstance(timestamp_str, str) and timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            else:
                timestamp = datetime.now()
                
            hour_of_day = timestamp.hour
            is_daytime = 1.0 if 6 <= hour_of_day < 20 else 0.0
            
        except Exception as e:
            print(f"Error parsing timestamp: {e}")
            timestamp = datetime.now()
            hour_of_day = timestamp.hour
            is_daytime = 1.0
        
        # Extract core sensor readings with reasonable defaults
        temperature = sensor_readings.get("Temperature", 25.0)  # Default room temperature
        soil_humidity = sensor_readings.get("Soil Humidity", 50.0)  # Middle value
        air_humidity = sensor_readings.get("Air Humidity", 50.0)  # Middle value
        light = sensor_readings.get("Light", 200.0)  # Moderate light level
        
        # Get plant growth stage
        plant_stage = data.get("plantGrowthStage", "Unknown")
        
        # Get time since last watering
        time_since_last_watering = float(data.get("timeSinceLastWateringInHours", 24.0))  # Default 1 day
        
        # Create engineered features
        soil_dryness_index = 100.0 - soil_humidity
        temp_soil_product = temperature * soil_humidity / 100.0
        light_humidity_ratio = light / max(soil_humidity, 1.0)  # Avoid division by zero
        
        # Process categorical variables using one-hot encoding
        stage_features = []
        stage_feature_names = []
        
        # Encode plant stage if encoder is available
        if encoder is not None:
            try:
                # One-hot encode plant growth stage
                stage_encoded = encoder.transform([[plant_stage]])
                
                # Check if it's already an array or needs to be converted
                if hasattr(stage_encoded, 'toarray'):
                    stage_vector = stage_encoded.toarray()[0]
                else:
                    # It's already an array
                    stage_vector = stage_encoded[0]
                    
                stage_feature_names = list(encoder.get_feature_names_out(["plantGrowthStage"]))
                stage_features = list(stage_vector)
            except Exception as e:
                print(f"Error encoding plant stage: {e}")
                stage_features = []
                stage_feature_names = []
        
        # Core features - order must match CORE_SENSORS
        core_features = [
            soil_humidity,  # First in training data CSV
            temperature,    # Second in training data CSV
            air_humidity, 
            light
        ]
        
        # Engineered features
        engineered_features = [
            soil_dryness_index,
            temp_soil_product,
            light_humidity_ratio,
            hour_of_day,
            is_daytime
        ]
        
        # Combine all features in the exact order expected by the model
        features = core_features + engineered_features + stage_features
        feature_names = CORE_SENSORS + ENGINEERED_FEATURES + stage_feature_names
        
        return features, feature_names
        
    except Exception as e:
        print(f"Error in mapping features: {e}")
        print(traceback.format_exc())
        
        # Return a safe fallback with empty features
        dummy_core = [50.0, 25.0, 50.0, 200.0]  # More reasonable defaults for core sensors
        dummy_engineered = [50.0, 12.5, 4.0, 12.0, 1.0]  # Reasonable defaults for engineered features
        dummy_categorical = []
        
        if encoder is not None:
            try:
                dummy_categorical = [0.0] * len(encoder.get_feature_names_out(["plantGrowthStage"]))
                stage_feature_names = list(encoder.get_feature_names_out(["plantGrowthStage"]))
            except Exception as e:
                print(f"Error creating dummy categorical features: {e}")
                dummy_categorical = [0.0]
                stage_feature_names = ["plantGrowthStage_Unknown"]
        else:
            dummy_categorical = [0.0]
            stage_feature_names = ["plantGrowthStage_Unknown"]
            
        dummy_features = dummy_core + dummy_engineered + dummy_categorical
        dummy_feature_names = CORE_SENSORS + ENGINEERED_FEATURES + stage_feature_names
        
        return dummy_features, dummy_feature_names