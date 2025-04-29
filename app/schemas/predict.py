from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional

class SensorData(BaseModel):
    sensorType: str
    value: float
    timestamp: datetime

class SensorHistoryEntry(BaseModel):
    value: float
    timestamp: datetime

class PredictionRequest(BaseModel):
    current: SensorData
    history: Optional[List[SensorHistoryEntry]] = None

class PredictionResult(BaseModel):
    timestamp: datetime
    status: str
    suggestion: str
    trendAnalysis: Optional[str] = None
