from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional

# Represents a single sensor reading (e.g., humidity, temperature)
class SensorData(BaseModel):
    temperature: float
    soil_moisture: float
    humidity: float
    light: float
    hour_of_day: int
    growthStage: str  # if used



# Represents the output returned by ML
class PredictionResult(BaseModel):
    timestamp: datetime
    predictedHoursUntilWatering: float

# Represents a single entry in the sensor history
class PredictionRequest(BaseModel):
    current: SensorData
