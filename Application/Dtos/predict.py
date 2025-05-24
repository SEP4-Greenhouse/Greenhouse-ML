from pydantic import BaseModel, Field
from datetime import datetime
from typing import List

class SensorReadingDto(BaseModel):
    SensorName: str
    Unit: str
    Value: float

class PredictionRequestDto(BaseModel):
    timestamp: datetime = Field(..., json_schema_extra={"example": "2024-06-10T12:34:56Z"})
    plantGrowthStage: str = Field(..., json_schema_extra={"example": "Seedling"})
    timeSinceLastWateringInHours: float = Field(..., json_schema_extra={"example": 12.5})
    mlSensorReadings: List[SensorReadingDto] = Field(
        ...,
        json_schema_extra={
            "example": [
                {"SensorName": "Temperature", "Unit": "°C", "Value": 0.0},
                {"SensorName": "Soil Humidity", "Unit": "%", "Value": 0.0},
                {"SensorName": "Air Humidity", "Unit": "%", "Value": 0.0},
                {"SensorName": "CO2", "Unit": "%", "Value": 0.0},
                {"SensorName": "Light", "Unit": "lux", "Value": 0.0},
                {"SensorName": "PIR", "Unit": "None", "Value": 0.0},
                {"SensorName": "Proximity", "Unit": "None", "Value": 0.0}
            ]
        }
    )

class PredictionResultDto(BaseModel):
    PredictionTime: datetime
    HoursUntilNextWatering: float
    modelVersion: str = Field(None, exclude=True)  # This field won't be included in JSON responses
    
    class Config:
        populate_by_name = True  # Updated from allow_population_by_field_name for Pydantic v2

class PredictionResponseDto(BaseModel):
    """DTO for API responses - matching C# naming exactly"""
    PredictionTime: datetime = Field(..., json_schema_extra={"example": "2024-06-10T12:34:56Z"})
    HoursUntilNextWatering: float = Field(..., json_schema_extra={"example": 24.5})
    
    class Config:
        populate_by_name = True