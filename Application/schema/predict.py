from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional

# Represents a single sensor reading (e.g., humidity, temperature)
class SensorData(BaseModel):
    sensorType: str           # Type of sensor (e.g., "humidity", "temperature")
    value: float              # Measured value
    timestamp: datetime       # Time when the reading was taken

# Represents a past reading in the sensor history (used for trend analysis)
class SensorHistoryEntry(BaseModel):
    value: float              # Historical value
    timestamp: datetime       # Corresponding timestamp

# Represents the full request to the prediction endpoint
# Includes the current sensor reading and optional history
class PredictionRequest(BaseModel):
    current: SensorData                         # The current live reading
    history: Optional[List[SensorHistoryEntry]] = None  # Past readings (optional)

# Response returned by the ML analysis endpoint
class PredictionResult(BaseModel):
    timestamp: datetime         # Time when prediction was generated
    status: str                 # "normal" or "warning" based on value
    suggestion: str             # Action recommended (e.g., "Start irrigation")
    trendAnalysis: Optional[str] = None  # Summary of changes over time (optional)
