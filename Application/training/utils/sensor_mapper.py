from datetime import datetime
from Application.schema.predict import SensorData

def map_backend_readings(data: dict) -> SensorData:
    sensor_map = {entry["sensorName"].lower(): entry["value"] for entry in data.get("mlSensorReadings", [])}
    timestamp = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))

    return SensorData(
        temperature=sensor_map.get("temperature", 0.0),
        humidity=sensor_map.get("humidity", 0.0),
        soil_moisture=sensor_map.get("soilmoisture", 0.0),
        light=sensor_map.get("light", 0.0),
        hour_of_day=timestamp.hour,
        growthStage=data.get("plantGrowthStage", "Unknown")
    )
