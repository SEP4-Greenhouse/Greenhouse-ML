from pydantic import BaseModel
from datetime import datetime

class SensorData(BaseModel):
    sensorType: str
    value: float
    timestamp: datetime

class PredictionResult(BaseModel):
    timestamp: datetime
    status: str
    suggestion: str
