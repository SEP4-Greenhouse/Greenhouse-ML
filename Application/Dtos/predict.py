from pydantic import BaseModel, Field
from datetime import datetime
from typing import List

class SensorReadingDto(BaseModel):
    SensorName: str
    Unit: str
    Value: float

class PredictionRequestDto(BaseModel):
    timestamp: datetime = Field(..., example="2024-06-10T12:34:56Z")
    plantGrowthStage: str = Field(..., example="Seedling")
    timeSinceLastWateringInHours: float = Field(..., example=12.5)
    mlSensorReadings: List[SensorReadingDto] = Field(
        ...,
        example=[
            {"SensorName": "Temperature", "Unit": "°C", "Value": 0.0},
            {"SensorName": "Soil Humidity", "Unit": "%", "Value": 0.0},
            {"SensorName": "Air Humidity", "Unit": "%", "Value": 0.0},
            {"SensorName": "CO2", "Unit": "%", "Value": 0.0},
            {"SensorName": "Light", "Unit": "lux", "Value": 0.0},
            {"SensorName": "PIR", "Unit": "None", "Value": 0.0},
            {"SensorName": "Proximity", "Unit": "None", "Value": 0.0}
        ]
    )

class PredictionResultDto(BaseModel):
    predictionTime: datetime = Field(..., alias="predictionTime")
    hoursUntilNextWatering: float = Field(..., alias="hoursUntilNextWatering")
    modelVersion: str = Field(None, exclude=True)

    class Config:
        allow_population_by_field_name = True
